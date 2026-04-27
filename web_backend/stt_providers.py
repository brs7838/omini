"""Pluggable STT (speech-to-text) backends for the voice pipeline.

Two options:

* ``sarvam`` — cloud call to Sarvam's `/speech-to-text` endpoint. Keeps
  the GPU free of a second big model so OmniVoice (TTS) has sole tenancy
  and cuDNN doesn't have to rechoose kernels between turns. Network
  round-trip is in the ~300-600 ms range; fine for a chat pipeline that
  already waits on LLM streaming.
* ``whisper`` — local faster-whisper ``large-v3-turbo`` on cuda:0. Faster
  and free but shares the GPU with OmniVoice, which we've found causes
  inter-turn audio drift (cuDNN + allocator churn).

Selection mirrors the LLM provider pattern: explicit override > persisted
config file at project root > env var > default (``sarvam``).
"""
from __future__ import annotations

import io
import json
import os
import wave
from dataclasses import dataclass
from typing import Any, Optional

import httpx
import numpy as np


SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"
SARVAM_DEFAULT_MODEL = "saaras:v3"
SARVAM_DEFAULT_LANGUAGE = "hi-IN"

WHISPER_DEFAULT_MODEL = "large-v3-turbo"

# Persisted STT choice. Sits next to provider_config.json at project root so
# the launcher and backend both see the same value without importing code.
STT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "stt_config.json",
)


def load_stt_config(path: str = STT_CONFIG_PATH) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return {}
    except Exception as e:
        print(f"[stt_providers] config read error: {e}", flush=True)
        return {}
    if not isinstance(data, dict):
        return {}
    return data


def save_stt_config(provider: str, model: Optional[str] = None,
                    path: str = STT_CONFIG_PATH) -> bool:
    try:
        payload = {"provider": provider, "model": model}
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
        return True
    except Exception as e:
        print(f"[stt_providers] config save error: {e}", flush=True)
        return False


@dataclass
class STTChoice:
    """Resolved STT selection. ``model`` is provider-specific (e.g.
    saarika:v2.5 for Sarvam, large-v3-turbo for Whisper). ``uses_local_gpu``
    tells the model store whether to load Whisper at boot."""
    name: str                       # "sarvam" | "whisper"
    model: str
    uses_local_gpu: bool


def resolve_stt(override_name: Optional[str] = None,
                override_model: Optional[str] = None) -> STTChoice:
    persisted = load_stt_config()
    persisted_name = (persisted.get("provider") or "").strip().lower() or None
    persisted_model = persisted.get("model") or None

    env_name = os.environ.get("STT_PROVIDER", "").strip().lower() or None

    name = (override_name or persisted_name or env_name or "sarvam").lower()
    if name not in ("sarvam", "whisper"):
        raise RuntimeError(f"Unknown STT_PROVIDER={name!r}. Expected 'sarvam' or 'whisper'.")

    if override_model is not None:
        model = override_model
    elif persisted_name == name and persisted_model:
        model = persisted_model
    else:
        model = SARVAM_DEFAULT_MODEL if name == "sarvam" else WHISPER_DEFAULT_MODEL

    return STTChoice(
        name=name,
        model=model,
        uses_local_gpu=(name == "whisper"),
    )


def _pcm_to_wav_bytes(audio_16k: np.ndarray, sample_rate: int = 16000) -> bytes:
    """Wrap a mono float32/int16 buffer into an in-memory WAV blob for the
    Sarvam uploader. Float input in [-1, 1] is clipped and converted to
    int16; already-int16 input is used as-is."""
    if audio_16k.dtype != np.int16:
        pcm = np.clip(audio_16k, -1.0, 1.0)
        pcm = (pcm * 32767.0).astype(np.int16)
    else:
        pcm = audio_16k
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


class SarvamSTT:
    """Cloud Sarvam /speech-to-text client. One HTTP client instance kept
    alive across turns so TLS/keep-alive costs are paid once."""

    def __init__(self, api_key: str, model: str = SARVAM_DEFAULT_MODEL,
                 language_code: str = SARVAM_DEFAULT_LANGUAGE):
        if not api_key:
            raise RuntimeError(
                "STT_PROVIDER=sarvam but SARVAM_API_KEY is not set. "
                "Add it to .env or export it in your shell."
            )
        self.name = "sarvam"
        self.model = model
        self.language_code = language_code
        self._headers = {"api-subscription-key": api_key}
        self._client = httpx.AsyncClient(timeout=15.0)

    async def transcribe(self, audio_16k: np.ndarray) -> str:
        wav_bytes = _pcm_to_wav_bytes(audio_16k, sample_rate=16000)
        files = {"file": ("audio.wav", wav_bytes, "audio/wav")}
        data = {"model": self.model, "language_code": self.language_code}
        try:
            r = await self._client.post(
                SARVAM_STT_URL,
                files=files,
                data=data,
                headers=self._headers,
            )
        except Exception as e:
            print(f"[SarvamSTT] request error: {type(e).__name__}: {e}", flush=True)
            return ""
        if r.status_code != 200:
            print(f"[SarvamSTT] HTTP {r.status_code}: {r.text[:200]}", flush=True)
            return ""
        try:
            body = r.json()
        except Exception as e:
            print(f"[SarvamSTT] JSON parse error: {e}", flush=True)
            return ""
        transcript = (body.get("transcript") or "").strip()
        return transcript

    async def aclose(self):
        try:
            await self._client.aclose()
        except Exception:
            pass


class LocalWhisperSTT:
    """Thin adapter around an already-loaded faster-whisper WhisperModel.
    ``transcribe`` runs in a threadpool via ``asyncio.to_thread`` so the
    event loop isn't blocked."""

    def __init__(self, whisper_model: Any, model_name: str = WHISPER_DEFAULT_MODEL,
                 language: str = "hi"):
        self.name = "whisper"
        self.model = model_name
        self.language = language
        self._m = whisper_model

    async def transcribe(self, audio_16k: np.ndarray) -> str:
        import asyncio
        segments, _ = await asyncio.to_thread(
            self._m.transcribe,
            audio_16k,
            beam_size=5,
            language=self.language,
            condition_on_previous_text=False,
        )
        return "".join(s.text for s in segments).strip()

    async def aclose(self):
        # Weights are owned by _ModelStore; nothing to close per-session.
        pass


def build_stt(choice: STTChoice, whisper_model: Any = None):
    """Factory: hand back a usable STT client for the resolved choice.
    Whisper branch expects a pre-loaded ``faster_whisper.WhisperModel``;
    caller passes ``None`` for Sarvam."""
    if choice.name == "sarvam":
        api_key = os.environ.get("SARVAM_API_KEY", "").strip()
        return SarvamSTT(api_key=api_key, model=choice.model)
    if choice.name == "whisper":
        if whisper_model is None:
            raise RuntimeError(
                "STT_PROVIDER=whisper requires the Whisper model to be loaded "
                "first. Call _ModelStore.load_whisper()."
            )
        return LocalWhisperSTT(whisper_model=whisper_model, model_name=choice.model)
    raise RuntimeError(f"Unknown STT provider {choice.name!r}")
