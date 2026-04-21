"""
TTS Lab — isolated FastAPI server for pitch/quality experiments.

NO LLM, NO STT, NO WebSocket, NO Asterisk. Just OmniVoice + preset phrases.

Endpoints:
    GET  /            → single-page UI
    GET  /api/voices  → list of available voice reference files
    GET  /api/phrases → preset Hindi phrases
    POST /api/synth   → {voice, text, num_step, guidance_scale, t_shift} → audio/wav

Run:
    python -m tts_lab.server
    # then open http://localhost:8100
"""

from __future__ import annotations

import io
import os
import sys
import time
import wave
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

REPO_ROOT = Path(__file__).resolve().parents[1]
VOICES_DIR = REPO_ROOT / "assets" / "voices"
STATIC_DIR = Path(__file__).resolve().parent / "static"

sys.path.insert(0, str(REPO_ROOT))

from omnivoice.models.omnivoice import OmniVoice, OmniVoiceGenerationConfig  # noqa: E402

from tts_lab.phrases_hindi import ALL as PHRASES  # noqa: E402


class SynthRequest(BaseModel):
    voice: str
    text: str
    num_step: int = 12
    guidance_scale: float = 2.0
    t_shift: float = 0.1


class LabState:
    def __init__(self) -> None:
        self.model: Optional[OmniVoice] = None
        self.device: str = "cpu"
        self.current_voice: Optional[str] = None
        self.voice_prompt = None

    def load_model(self) -> None:
        if self.model is not None:
            return
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        print(f"[lab] Loading OmniVoice on {device} dtype={dtype}...", flush=True)
        t0 = time.perf_counter()
        try:
            self.model = OmniVoice.from_pretrained(
                "k2-fsa/OmniVoice",
                device_map=device,
                dtype=dtype,
                load_asr=False,
            )
            self.device = device
            print(f"[lab] Loaded on {device} in {time.perf_counter() - t0:.1f}s", flush=True)
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            print(f"[lab] GPU load failed: {e}. Falling back to CPU.", flush=True)
            torch.cuda.empty_cache()
            self.model = OmniVoice.from_pretrained(
                "k2-fsa/OmniVoice",
                device_map="cpu",
                dtype=torch.float32,
                load_asr=False,
            )
            self.device = "cpu"
            print(f"[lab] Loaded on CPU in {time.perf_counter() - t0:.1f}s", flush=True)

    def ensure_voice(self, voice_filename: str) -> None:
        if self.current_voice == voice_filename and self.voice_prompt is not None:
            return
        voice_path = VOICES_DIR / voice_filename
        if not voice_path.exists():
            raise HTTPException(status_code=404, detail=f"Voice not found: {voice_filename}")
        print(f"[lab] Cloning voice: {voice_filename}", flush=True)
        t0 = time.perf_counter()
        self.voice_prompt = self.model.create_voice_clone_prompt(
            ref_audio=str(voice_path), preprocess_prompt=True
        )
        self.current_voice = voice_filename
        print(f"[lab] Clone ready in {time.perf_counter() - t0:.2f}s", flush=True)


state = LabState()

app = FastAPI(title="TTS Lab")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup() -> None:
    state.load_model()


@app.get("/api/voices")
def api_voices() -> dict:
    if not VOICES_DIR.exists():
        return {"voices": []}
    voices = sorted(
        p.name for p in VOICES_DIR.iterdir()
        if p.suffix.lower() in {".mp3", ".wav", ".flac", ".ogg"}
    )
    return {"voices": voices, "default": "ravi_sir_8s.wav" if "ravi_sir_8s.wav" in voices else (voices[0] if voices else None)}


@app.get("/api/phrases")
def api_phrases() -> dict:
    return {"phrases": PHRASES}


@app.get("/api/status")
def api_status() -> dict:
    return {
        "device": state.device,
        "model_loaded": state.model is not None,
        "current_voice": state.current_voice,
    }


def _audio_to_wav_bytes(audio: np.ndarray, sr: int = 24000) -> bytes:
    pcm = np.clip(audio, -1.0, 1.0)
    pcm_int16 = (pcm * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm_int16.tobytes())
    return buf.getvalue()


@app.post("/api/synth")
def api_synth(req: SynthRequest) -> Response:
    if state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    state.ensure_voice(req.voice)

    cfg = OmniVoiceGenerationConfig(
        num_step=req.num_step,
        guidance_scale=req.guidance_scale,
        t_shift=req.t_shift,
    )
    t0 = time.perf_counter()
    try:
        out = state.model.generate(
            text=req.text,
            voice_clone_prompt=state.voice_prompt,
            language="hi",
            generation_config=cfg,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Synth failed: {e}")
    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    audio = out[0]
    if hasattr(audio, "cpu"):
        audio = audio.cpu().numpy()
    audio = np.asarray(audio, dtype=np.float32)
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    duration_s = len(audio) / 24000.0

    print(
        f"[synth] voice={req.voice} ns={req.num_step} "
        f"in_chars={len(req.text)} synth_ms={elapsed_ms} "
        f"audio_s={duration_s:.2f} peak={peak:.3f}",
        flush=True,
    )

    wav_bytes = _audio_to_wav_bytes(audio, sr=24000)
    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={
            "X-Synth-Ms": str(elapsed_ms),
            "X-Audio-Seconds": f"{duration_s:.3f}",
            "X-Peak": f"{peak:.4f}",
            "X-Device": state.device,
            "X-Num-Step": str(req.num_step),
        },
    )


@app.get("/")
def ui_root() -> FileResponse:
    idx = STATIC_DIR / "index.html"
    if not idx.exists():
        raise HTTPException(status_code=404, detail="UI missing")
    return FileResponse(str(idx))


if __name__ == "__main__":
    import uvicorn
    host = os.environ.get("TTS_LAB_HOST", "127.0.0.1")
    port = int(os.environ.get("TTS_LAB_PORT", "8100"))
    uvicorn.run(app, host=host, port=port, log_level="info")
