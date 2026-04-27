from __future__ import annotations
import asyncio
import collections
from typing import Optional, Dict, List, Any, cast
import sys
import os
import traceback
import httpx
import json
import random
import re
import numpy as np
import torch
import time
from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from scipy import signal
import hashlib

# Ensure imports work when run from within web_backend or parent dir
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from campaigns import load_campaign, build_system_prompt  # type: ignore
from stt_providers import (  # type: ignore
    STTChoice, resolve_stt, build_stt, LocalWhisperSTT, SarvamSTT,
)


_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


def _strip_think_streaming(chunk: str, state: dict) -> str:
    """Strip <think>...</think> blocks across a streaming content feed.

    Reasoning-capable LLMs (Sarvam-M / Sarvam-30B, Qwen-thinking, etc.) can
    wrap their chain-of-thought in <think>…</think> and emit it token-by-token
    in the same `content` stream as the final answer. Our TTS picks up any
    text that contains end-of-sentence punctuation, so without a filter the
    voters would hear the model's English deliberation before the real answer.

    `state` is a per-turn dict the caller initialises as
    `{"in_think": False, "tail": ""}`. `tail` holds up to ~8 chars carried over
    from the previous chunk so that an opening/closing tag split across chunk
    boundaries (e.g. "<thi" then "nk>") is detected reliably. The small
    delay it introduces (holding back up to 8 chars until we know they're not
    the start of a tag) is imperceptible in a voice pipeline."""
    text = state.get("tail", "") + (chunk or "")
    state["tail"] = ""
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        if not state["in_think"]:
            open_idx = text.find(_THINK_OPEN, i)
            if open_idx == -1:
                # No opening tag in view. Emit everything except the last few
                # chars, which could be the start of "<think>" split across
                # chunks. Held chars carry over as tail.
                hold = len(_THINK_OPEN) - 1  # 6
                if n - i > hold:
                    out.append(text[i : n - hold])
                    state["tail"] = text[n - hold :]
                else:
                    state["tail"] = text[i:]
                break
            out.append(text[i:open_idx])
            i = open_idx + len(_THINK_OPEN)
            state["in_think"] = True
        else:
            close_idx = text.find(_THINK_CLOSE, i)
            if close_idx == -1:
                # Inside think block, closing tag not yet arrived. Drop
                # everything except the last few chars (possible partial
                # "</think>") so we can detect it in the next chunk.
                hold = len(_THINK_CLOSE) - 1  # 7
                if n - i > hold:
                    state["tail"] = text[n - hold :]
                else:
                    state["tail"] = text[i:]
                break
            i = close_idx + len(_THINK_CLOSE)
            state["in_think"] = False
    return "".join(out)

# --- Configuration (Hardcoded for Vaani Web) ---
LLM_URL = "http://127.0.0.1:11434/v1/chat/completions"
LLM_MODEL = "gemma3:4b"
SILENCE_CHUNKS = 12             # ~1s at 4096-sample frames (native-rate). Old value of 1 (~85ms) was splitting
                                # one continuous sentence into multiple turns whenever the user paused to breathe,
                                # so the LLM answered each fragment separately. 1s lets a speaker finish a thought.
MAX_BUFFER_CHUNKS = 250
BARGE_IN_FRAMES = 8             # Ensure AI stops immediately when user interrupts (fast barge-in)
TTS_CHUNK_SAMPLES = 6000        # 250ms at 24kHz — stream small chunks for low-latency playback

# Short acknowledgement phrases synthesized per voice so llm_worker can emit
# one immediately at turn start. Must be brief (<1s synthesized) so they
# finish playing before real TTS lands, and written with a terminal Devanagari
# danda so OmniVoice's prosody planner treats each as a complete utterance.
# Mix of bare acknowledgements and "go on" style invitations, picked at random
# by llm_worker — keeps backchannels from sounding like a stuck record.
_BACKCHANNEL_PHRASES = [
    "अच्छा।", "हम्म।", "ठीक है।", "जी।", "हाँ।",
    "अच्छा, आप बोलिए।", "हम्म, बोलिए।", "जी, बोलिए।", "हाँ, बोलिए।",
]

# How smart-backchannel gating behaves. Tuned to feel human:
#   * MIN_GAP stops AI from "अच्छा" every single turn — real listeners don't.
#   * PROBABILITY <1.0 means even when eligible, AI sometimes stays silent.
#   * MIN_UTTERANCE_LEN skips backchannel on tiny "ok"/"haan" style turns
#     where an immediate AI reply already lands fast enough.
BACKCHANNEL_MIN_GAP_SEC = 5.0
BACKCHANNEL_PROBABILITY = 0.0
BACKCHANNEL_MIN_UTTERANCE_LEN = 12

# Utterance coalescing window. Whisper/Sarvam flush a transcript as soon as
# SILENCE_CHUNKS of silence elapse (~1s). But real speakers often pause
# mid-sentence ("arre yaar … kaise ho … ye baat galat hai") — each pause
# triggers a fresh flush, and the old code fed every fragment to the LLM
# separately, so the AI replied to the tail scrap instead of the whole thought.
# Fix: hold the transcript in a buffer for COALESCE_SEC; if more transcripts
# land, concatenate and restart the timer. Only enqueue to llm_queue when the
# user has been genuinely quiet for the full window. ChatGPT / Gemini voice
# mode behave the same way.
COALESCE_SEC = 1.0

# Use absolute path for reliability
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF_AUDIO_PATH = os.path.join(ROOT_DIR, "assets", "voices", "ravi_sir_8s.wav")

def clean_text(text):
    """Zero-Tolerance Tag Guard: Validates and whitelists only supported OmniVoice acoustic tags.
    Strips away ANY other bracketed content to prevent hallucinations.
    """
    # 1. Supported Tags Whitelist (Lowercase for matching)
    ALLOWED_TAGS = ["laughter", "sigh", "sniff", "dissatisfaction-hnn"]
    
    # 2. Extract and Validate all bracketed tags [...]
    def validate_and_protect(match):
        content = match.group(1).lower().strip()
        mapping = {
            "हँसी": "laughter", "हंसी": "laughter", "haha": "laughter",
            "आह": "sigh", "sigh": "sigh",
            "स्निफ़": "sniff", "sniff": "sniff",
            "हम्म": "dissatisfaction-hnn", "हूँ": "dissatisfaction-hnn", "hnn": "dissatisfaction-hnn"
        }
        normalized = mapping.get(content, content)
        if normalized in ALLOWED_TAGS:
            return f"[{normalized}]"
        return "" # Delete non-whitelisted brackets
    
    # Apply the validator
    text = re.sub(r'\[(.*?)\]', validate_and_protect, text)

    # 3. Standard Text Cleaning
    tag_map = {}
    def hide_tag(match):
        placeholder = f"§§{len(tag_map)}§§"
        tag_map[placeholder] = match.group(0)
        return placeholder
    
    text = re.sub(r'\[.*?\]', hide_tag, text)
    text = re.sub(r'#+\s*', '', text)
    text = text.replace('**', '').replace('*', '')
    text = re.sub(r'[\U00010000-\U0010ffff]', '', text)
    # text = re.sub(r'[a-zA-Z]', '', text) # DELETED: Removing Roman characters was causing Hinglish silence.
    
    # --- OmniVoice Native Hindi Punctuation Formatting ---
    # To prevent "breathlessness" and "fluctuations" in long paragraphs:
    # 1. OmniVoice Hindi expects Devanagari Danda for full stops.
    text = text.replace('.', '।')
    # 2. Exclamation marks often cause extreme overacting/fluctuating pitch. Map to Danda.
    text = text.replace('!', '।')
    # 3. Commas need explicit trailing spaces to guarantee a clean structural pause.
    text = text.replace(',', ', ')
    text = text.replace('?', '? ')
    text = text.replace('-', ' ')
    # 4. Collapse extra spaces
    text = re.sub(r'\s+', ' ', text)
    
    # 4. Final Cleanup: Remove any stray brackets that didn't form a valid tag (before restoring placeholders)
    # This prevents the AI from saying "k" or click-sounds for single brackets
    text = re.sub(r'[\[\]]', '', text)

    # Restore original tags
    for placeholder, original in tag_map.items():
        text = text.replace(placeholder, original)
    
    return text.strip()

def synth_backchannels(tts_model, voice_prompt) -> list[bytes]:
    """Synthesize the standard backchannel phrases against a given voice
    prompt and return a list of raw int16 PCM bytes at 24 kHz (ready to emit
    via audio_chunk). Runs synchronously on the calling thread — callers that
    need to avoid blocking the event loop should wrap this in asyncio.to_thread.
    """
    out: list[bytes] = []
    for phrase in _BACKCHANNEL_PHRASES:
        try:
            audio = tts_model.generate(
                text=phrase,
                voice_clone_prompt=voice_prompt,
                language="hi",
                generation_config=OmniVoiceGenerationConfig(
                    num_step=12, guidance_scale=2.0, t_shift=0.1,
                    audio_chunk_threshold=600.0,
                    postprocess_output=False,
                ),
            )[0]
            if hasattr(audio, "cpu"):
                audio = audio.cpu().numpy()
            pcm = (np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0) * 32767.0).astype(np.int16)
            out.append(pcm.tobytes())
        except Exception as bc_err:
            print(f"[Engine] backchannel synth failed for {phrase!r}: {bc_err}", flush=True)
    return out


class _ModelStore:
    """Singleton: loads heavy GPU models ONCE, reused across all WebSocket
    connections. Whisper is now lazy — only loaded when the active STT
    provider is 'whisper'. When STT is 'sarvam' (default), the GPU holds
    only OmniVoice, which keeps the CUDA allocator clean and stops cuDNN
    from rechoosing convolution kernels between TTS turns. That allocator
    churn was the root of the inter-turn pitch drift."""
    _instance = None
    _loaded: bool
    asr_model: Any
    _whisper_name: Optional[str]
    tts_model: Any
    voice_prompt: Any
    backchannel_pcm: List[Any]
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loaded = False
            cls._instance.asr_model = None
            cls._instance._whisper_name = None
        return cls._instance

    def load(self):
        if self._loaded:
            print("[Engine] Models already loaded, reusing.")
            return
        try:
            print("[Engine] Loading TTS (one-time)...")
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            # Force deterministic CUDA kernels globally. Paired with
            # CUBLAS_WORKSPACE_CONFIG=:4096:8 (set in main.py before torch
            # import) and per-generate seed pinning in tts_worker, this is
            # what makes the diffusion sampler reproduce the same audio for
            # the same text across turns. warn_only=True so any op without a
            # deterministic kernel logs instead of crashing the worker.
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception as det_err:
                print(f"[Engine] use_deterministic_algorithms failed (continuing): {det_err}")
            self.tts_model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cuda:0", dtype=torch.float16, load_asr=False)
            self.voice_prompt = self.tts_model.create_voice_clone_prompt(ref_audio=REF_AUDIO_PATH, preprocess_prompt=True)
            # Pre-synth short backchannels so the llm_worker can fire one
            # instantly at the start of a turn without waiting for an
            # OmniVoice round-trip. These kill the perceived 2-3s silence
            # between user-stops-talking and AI-starts-talking — the real
            # LLM reply still takes its full time, but the user hears
            # *something* within ~50ms of their turn ending. This cache is
            # tied to the DEFAULT voice (Ravi); when a session selects a
            # different voice, WebAssistant regenerates its own copy so the
            # backchannel matches the chosen speaker.
            self.backchannel_pcm = synth_backchannels(self.tts_model, self.voice_prompt)
            print(f"[Engine] Pre-synth {len(self.backchannel_pcm)} default backchannels ready.")
            self._loaded = True
            print("[Engine] TTS Loaded Successfully.")
        except Exception as e:
            print(f"[Engine] CRITICAL FAILURE during loading: {e}")
            raise e

    def load_whisper(self, model_name: str = "large-v3-turbo"):
        """Lazily load faster-whisper. Called only when STT provider is
        switched to 'whisper' (at boot or via /stt/switch). Safe to call
        repeatedly — re-uses the loaded instance if the model name matches."""
        if self.asr_model is not None and getattr(self, "_whisper_name", None) == model_name:
            return self.asr_model
        # Loading a different model than what's cached — drop the old one
        # so VRAM doesn't double up.
        if self.asr_model is not None:
            try:
                del self.asr_model
            except Exception:
                pass
            self.asr_model = None
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        from faster_whisper import WhisperModel
        print(f"[Engine] Loading Whisper ({model_name}) on cuda:1 ...", flush=True)
        self.asr_model = WhisperModel(model_name, device="cuda", device_index=1, compute_type="int8_float16")
        self._whisper_name = model_name
        print("[Engine] Whisper loaded.", flush=True)
        return self.asr_model

    def unload_whisper(self):
        """Free the Whisper GPU weights when user switches back to Sarvam.
        Keeps VRAM lean and gets the allocator back to its cleaner state."""
        if self.asr_model is None:
            return
        try:
            del self.asr_model
        except Exception:
            pass
        self.asr_model = None
        self._whisper_name: Optional[str] = None
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        print("[Engine] Whisper unloaded.", flush=True)

_models = _ModelStore()

class WebAssistant:
    def __init__(self, event_handler=None, llm_url=None, llm_model=None,
                 llm_headers=None, llm_extras=None, get_provider=None,
                 get_campaign=None, get_stt=None, input_sampling_rate=24000):
        self.event_handler = event_handler
        self.llm_url = llm_url or LLM_URL
        self.llm_model = llm_model or LLM_MODEL
        # Extra request shape for the active provider — auth header and any
        # body-level fields like Sarvam's top-level `max_tokens`. Default is
        # empty so the legacy Ollama path stays untouched.
        self.llm_headers = dict(llm_headers or {})
        self.llm_extras = dict(llm_extras or {})
        # Optional callable the engine consults at the start of each LLM turn
        # so a /provider/switch mid-session takes effect on the next turn
        # without having to tear the WebSocket down.
        self.get_provider = get_provider
        # Same pattern for the campaign config — edited live via /campaign,
        # re-read per turn so a PUT takes effect without reopening the socket.
        self.get_campaign = get_campaign
        # And for STT — /stt/switch updates app.state.stt; asr_worker re-reads
        # before each transcription so the next utterance picks up the switch
        # without reopening the socket.
        self.get_stt = get_stt
        self.input_sampling_rate = input_sampling_rate
        _models.load()
        self.tts_model = _models.tts_model
        # asr_model is only populated when Whisper is the active STT provider.
        self.asr_model = _models.asr_model
        self.active_voice_metadata = {
            "name": "Vaani",
            "gender": "male",
            "age": "30",
            "about": "intelligent, witty, and helpful local assistant",
            "dialogues": "Main Ravi hoon, aapka smart local assistant.",
            "persona": "intelligent, witty, and helpful",
            "prompt": "ravi",
            "style": "authoritative, Indian, professional"
        }
        self.voice_prompt = _models.voice_prompt
        # Pre-synth short acknowledgement PCMs. llm_worker plays one right as a
        # turn starts so the user hears *something* in ~50ms instead of waiting
        # 1.5-3s for LLM+TTS. Mirrors human backchanneling in conversation.
        # Tied to the currently-loaded voice — regenerated by set_voice_prompt
        # on switch so the "अच्छा" isn't in a different speaker than the reply.
        self.backchannels: list[bytes] = list(getattr(_models, "backchannel_pcm", []) or [])
        # Monotonic counter — a late-finishing regen task for an older voice
        # must not overwrite a newer voice's backchannels if two switches
        # happen quickly.
        self._backchannel_gen = 0
        # Smart-backchannel gating. Tracks the last time we actually emitted
        # a backchannel so llm_worker can enforce a minimum gap between them.
        # Without this, every user turn got an "अच्छा" and it started sounding
        # robotic / nagging within a few exchanges.
        self.last_backchannel_time = 0.0
        # Utterance coalescer. ASR flushes on every ~1s pause, but a real
        # sentence can contain 2-3 sub-second pauses; each flush would otherwise
        # become its own LLM turn. These fields hold the pending fragments and
        # the timer that commits them to llm_queue once the user truly stops.
        self._pending_user_text = ""
        self._coalesce_task: asyncio.Task | None = None
        self.initial_prompt = "Natural Hindi/Hinglish: नमस्ते, क्या हाल है? मैं ठीक हूँ। Hello, how are you?"
        # self.llm_model = LLM_MODEL # REMOVED: Now set from __init__
        self.is_running = True
        self.is_speaking = False
        self.speaking_start_time = 0.0    # Track when AI starts speaking for barge-in grace period
        self.interrupt_event = asyncio.Event()
        self.history = []
        self.asr_queue = asyncio.Queue()
        self.llm_queue = asyncio.Queue()
        self.tts_queue = asyncio.Queue()

    async def emit(self, msg_type, data):
        if msg_type == "status":
            print(f"[Status] -> {data}")
            if data == "speaking":
                self.speaking_start_time = time.time()

        if self.event_handler:
            # Guard: a closed or broken WebSocket will raise here. If we let it
            # propagate, the enclosing worker task dies silently inside
            # asyncio.gather — the next turn then finds the queue has no reader
            # and stalls forever. Log + swallow so the worker survives.
            try:
                await self.event_handler(msg_type, data)
            except Exception as e:
                print(f"[emit] {msg_type} dropped: {type(e).__name__}: {e}")

    def _coalesce_user_text(self, fragment: str) -> None:
        """Queue a transcript fragment and (re)start the debounce timer.

        Each ASR flush calls this instead of llm_queue.put(). We append to a
        buffer and schedule a COALESCE_SEC wait; any fragment landing before
        the timer fires cancels it and starts a new one, so rapid-fire flushes
        from mid-sentence pauses merge into a single LLM turn.
        """
        fragment = fragment.strip()
        if not fragment:
            return
        if self._pending_user_text:
            # Join with a space; ASR transcripts don't always end in punctuation,
            # and the LLM handles whitespace-separated clauses fine.
            self._pending_user_text = f"{self._pending_user_text} {fragment}"
        else:
            self._pending_user_text = fragment
        # Cancel any in-flight debounce and start a fresh one.
        if self._coalesce_task and not self._coalesce_task.done():
            self._coalesce_task.cancel()
        self._coalesce_task = asyncio.create_task(self._flush_coalesced())

    async def _flush_coalesced(self) -> None:
        """Wait COALESCE_SEC then commit the buffered user text to llm_queue.

        Cancelled by _coalesce_user_text() whenever another fragment arrives,
        so we only fire after the user has been silent for the full window.
        """
        try:
            await asyncio.sleep(COALESCE_SEC)
        except asyncio.CancelledError:
            return
        text = self._pending_user_text.strip()
        self._pending_user_text = ""
        if text:
            print(f"[Engine] Putting to llm_queue: {text!r}", flush=True)
            await self.llm_queue.put(text)

    def set_voice_prompt(self, voice_prompt) -> None:
        """Swap the active voice AND clear/regenerate backchannels to match.
        Without the regen, the pre-synth "अच्छा" keeps playing in the old
        (default Ravi) voice even after the user selected someone else —
        audible as a brief wrong-speaker glitch before each reply.

        The actual synth runs in a background thread so callers don't block
        their WebSocket loop; backchannel emission is disabled until the new
        batch is ready. A monotonic gen counter guards against a slower
        regen from an older switch landing after a newer one."""
        self.voice_prompt = voice_prompt
        self._backchannel_gen += 1
        gen = self._backchannel_gen
        # Clear immediately so the llm_worker's `if self.backchannels` gate
        # skips emission until the new-voice batch arrives. A brief gap
        # (2-3 turns with no backchannel) is far better than emitting the
        # old voice's "अच्छा" right before a new-voice reply.
        self.backchannels = []

        async def _regen():
            try:
                pcms = await asyncio.to_thread(synth_backchannels, self.tts_model, voice_prompt)
                if self._backchannel_gen == gen:
                    self.backchannels = pcms
                    print(f"[Engine] Regenerated {len(pcms)} backchannels for new voice (gen={gen}).", flush=True)
            except Exception as e:
                print(f"[Engine] backchannel regen failed (gen={gen}): {e}", flush=True)

        try:
            asyncio.create_task(_regen())
        except RuntimeError:
            # No running loop (shouldn't happen from WS handler, but defensively):
            # skip — backchannels will stay empty until next switch.
            pass

    async def asr_worker(self):
        print("[ASR] Worker started.")
        buffer = []
        pre_speech_buffer = collections.deque(maxlen=15) # Approx ~300-500ms of history
        silence_count = 0
        barge_in_count = 0
        is_speech_active = False
        while self.is_running:
            try:
                chunk_bytes = await self.asr_queue.get()
                audio_float32 = np.frombuffer(chunk_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                rms = np.sqrt(np.mean(audio_float32 ** 2))
                
                # Adaptive Threshold: Be less sensitive while speaking to ignore echo
                threshold = 0.08 if self.is_speaking else 0.025
                is_speech = rms > threshold

                if self.is_speaking and is_speech:
                    # GRACE PERIOD: Ignore barge-in during the first 1.5s of speech 
                    # to prevent noise/echo from skipping AI's turn immediately.
                    if time.time() - self.speaking_start_time < 1.5:
                        continue 
                        
                    barge_in_count += 1
                    if barge_in_count >= BARGE_IN_FRAMES:
                        await self.emit("log", f"[Barge-in] DETECTED (Energy: {rms:.4f})")
                        self.interrupt_event.set()
                        # Flush queues
                        while not self.tts_queue.empty(): self.tts_queue.get_nowait()
                        while not self.llm_queue.empty(): self.llm_queue.get_nowait()
                        await self.emit("status", "interrupted")
                        self.is_speaking = False
                        barge_in_count = 0
                        # Important: let it break here or proceed to consume
                else:
                    barge_in_count = 0

                if is_speech:
                    if not is_speech_active:
                        # Just started speaking: attach the recent trailing audio so we don't miss first faint sounds
                        buffer.extend(list(pre_speech_buffer))
                    is_speech_active = True
                    silence_count = 0
                    buffer.append(chunk_bytes)
                elif is_speech_active:
                    silence_count += 1
                    buffer.append(chunk_bytes)
                else:
                    pre_speech_buffer.append(chunk_bytes)

                if is_speech_active and silence_count >= SILENCE_CHUNKS and len(buffer) > 3:
                    audio_bytes = b"".join(buffer)
                    audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                    
                    # Calculate Resample Ratio dynamically (target 16000)
                    # For 24k -> 16k: ratio 2/3. For 8k -> 16k: ratio 2/1.
                    from fractions import Fraction
                    ratio = Fraction(16000, self.input_sampling_rate)
                    audio_16k = signal.resample_poly(audio_np, ratio.numerator, ratio.denominator)
                    
                    # Safe Volume Normalization
                    # If the speech is clear but quiet, we boost it to standard volume.
                    # This prevents Whisper from hallucinating on low-volume waveforms.
                    peak = np.max(np.abs(audio_16k))
                    if peak > 0.05:
                        audio_16k = audio_16k / peak
                    elif peak > 0.01:
                        audio_16k = audio_16k * (0.1 / peak)

                    # Route through the active STT provider. sarvam (default)
                    # hits Sarvam's /speech-to-text endpoint; whisper uses the
                    # locally-loaded faster-whisper model. Pulled per-turn via
                    # self.get_stt so a /stt/switch takes effect immediately.
                    stt = None
                    if self.get_stt is not None:
                        try:
                            stt = self.get_stt()
                        except Exception as _e:
                            print(f"[ASR] get_stt error: {_e}", flush=True)
                    if stt is None and self.asr_model is not None:
                        stt = LocalWhisperSTT(whisper_model=self.asr_model)
                    if stt is None:
                        await self.emit("log", "[ASR] No STT provider configured; dropping utterance")
                        text = ""
                    else:
                        # Whisper's mel-spec wants ≥200ms of context or it
                        # hallucinates on very short utterances. Sarvam cloud
                        # handles short clips natively, so skip the 400ms of
                        # padding there — it was turning every turn into an
                        # extra ~200ms of Sarvam server time for no accuracy
                        # gain.
                        if isinstance(stt, LocalWhisperSTT):
                            padding = np.zeros(int(16000 * 0.2), dtype=np.float32)
                            audio_to_stt = np.concatenate([padding, audio_16k, padding])
                        else:
                            audio_to_stt = audio_16k
                        try:
                            text = await stt.transcribe(audio_to_stt)
                        except Exception as _e:
                            print(f"[ASR] transcribe error: {type(_e).__name__}: {_e}", flush=True)
                            text = ""
                    
                    # Smarter Junk Filter: Only drop if the ENTIRE transcript is a junk word
                    junk_patterns = ["watching", "subscribe", "thank you", "thanks for", "please like"]
                    is_pure_junk = len(text.split()) <= 2 and any(p in text.lower() for p in junk_patterns)
                    if is_pure_junk: text = ""

                    if text and len(text) >= 1:
                        await self.emit("log", f"[ASR] Speech: {text}")
                        await self.emit("transcript", {"role": "user", "text": text})
                        # Queue for coalesced dispatch instead of firing the
                        # LLM turn immediately — lets mid-sentence pauses
                        # merge into one request.
                        self._coalesce_user_text(text)

                        # Barge-in Interruption
                        self.interrupt_event.set()
                        await self.emit("status", "interrupted")
                        self.is_speaking = False # Reset state immediately so next turn is sensitive
                        # Drain remaining TTS to stop immediately
                        while not self.tts_queue.empty(): self.tts_queue.get_nowait()
                    buffer = []
                    silence_count = 0
                    is_speech_active = False
            except Exception as e:
                print(f"ASR Error: {e}")

    async def llm_worker(self):
        print("[LLM] Worker started.", flush=True)
        while self.is_running:
            try:
                async with httpx.AsyncClient(timeout=60.0) as client:
                    print("[LLM] HEARTBEAT: Ready and waiting for turns...", flush=True)
                    while self.is_running:
                        user_text = await self.llm_queue.get()
                        try:
                            # Fresh turn: clear any stale interrupt flag
                            self.interrupt_event.clear()
                            print(f"[LLM] dequeued turn: {user_text!r}", flush=True)
                            
                            # Per-turn provider refresh
                            p = None
                            if self.get_provider is not None:
                                try:
                                    p = self.get_provider()
                                    if p is not None:
                                        self.llm_url = p.url
                                        self.llm_model = p.model
                                        self.llm_headers = p.headers()
                                        self.llm_extras = p.payload_extras()
                                except Exception as _e:
                                    print(f"[LLM] get_provider error: {_e}", flush=True)

                            # Per-turn campaign refresh
                            campaign = None
                            if self.get_campaign is not None:
                                try:
                                    campaign = self.get_campaign()
                                except Exception:
                                    pass
                            if campaign is None:
                                campaign = load_campaign()
                            
                            system_prompt = build_system_prompt(
                                campaign,
                                voice_name=self.active_voice_metadata.get("name", "AI"),
                                voice_gender=self.active_voice_metadata.get("gender", "male"),
                                voice_metadata=self.active_voice_metadata,
                            )
                            
                            print(f"[LLM] SYSTEM PROMPT REPR (len={len(system_prompt)}): {repr(system_prompt)[:160]}...", flush=True)

                            user_msg: Dict[str, Any] = {"role": "user", "content": user_text}
                            messages: List[Dict[str, Any]] = self.history[-30:] + [user_msg]
                            
                            payload: Dict[str, Any]
                            if p is not None:
                                payload = p.build_payload(system_prompt, messages=messages)
                            else:
                                payload = {"messages": messages, "stream": True, "model": self.llm_model}
                            
                            if self.llm_extras:
                                payload.update(self.llm_extras)
                            
                            # Enforce conversational brevity for KCR mode
                            voice_id_check = self.active_voice_metadata.get("id", "").lower()
                            voice_name_check = self.active_voice_metadata.get("name", "").lower()
                            if "kcr" in voice_id_check or "kcr" in voice_name_check:
                                opts = payload.get("options", {})
                                if isinstance(opts, dict):
                                    opts["num_predict"] = 128
                                    payload["options"] = opts
                                else:
                                    payload["max_tokens"] = 128

                            sentence = ""
                            full_response = ""
                            think_state = {"in_think": False, "tail": ""}
                            await self.emit("status", "thinking")

                            # Smart backchannel gating
                            now = time.time()
                            gap_ok = (now - self.last_backchannel_time) >= BACKCHANNEL_MIN_GAP_SEC
                            len_ok = len(user_text.strip()) >= BACKCHANNEL_MIN_UTTERANCE_LEN
                            roll_ok = random.random() < BACKCHANNEL_PROBABILITY
                            if bool(self.backchannels) and not self.interrupt_event.is_set() and gap_ok and len_ok and roll_ok:
                                try:
                                    bc = random.choice(self.backchannels)
                                    self.is_speaking = True
                                    await self.emit("status", "speaking")
                                    await self.emit("audio_chunk", bc)
                                    self.last_backchannel_time = now
                                except Exception as bc_err:
                                    print(f"[LLM] backchannel emit failed: {bc_err}", flush=True)

                            sentence_end_re = re.compile(r"(\n\n)")
                            print(f"[LLM] Starting stream request to: {self.llm_url} (model={self.llm_model})", flush=True)
                            async with client.stream("POST", self.llm_url, json=payload, headers=self.llm_headers or None) as response:
                                response = cast(httpx.Response, response)
                                if response.status_code != 200:
                                    err_body = await response.aread()
                                    print(f"[LLM] API ERROR (status={response.status_code}) url={self.llm_url}: {err_body.decode(errors='ignore')}", flush=True)
                                    await self.emit("log", f"[LLM] API Error: {response.status_code}")
                                    await self.emit("status", "idle")
                                    continue

                                async for line in response.aiter_lines():
                                    if not line.strip(): continue
                                    if self.interrupt_event.is_set():
                                        await self.emit("status", "interrupted")
                                        break
                                    
                                    if line.startswith("data: "):
                                        if "[DONE]" in line: break
                                        try:
                                            body = json.loads(line[6:])
                                        except json.JSONDecodeError:
                                            continue
                                    else:
                                        continue

                                    choices = body.get("choices") or []
                                    if not choices: continue
                                    delta = choices[0].get("delta") or {}
                                    raw_chunk = delta.get("content") or ""
                                    chunk = _strip_think_streaming(raw_chunk, think_state)
                                    if not chunk: continue
                                    
                                    sentence += chunk
                                    full_response += chunk
                                    await self.emit("llm_chunk", chunk)
                                    
                                    while True:
                                        m = sentence_end_re.search(sentence)
                                        if not m: break
                                        end = m.end()
                                        part = sentence[:end].strip()
                                        sentence = sentence[end:]
                                        if part and not self.interrupt_event.is_set():
                                            cleaned = clean_text(part)
                                            if cleaned:
                                                await self.tts_queue.put(cleaned)

                            # End of stream flush
                            if think_state["in_think"]:
                                await self.emit("log", "[LLM] Model response was entirely reasoning block.")
                            if not think_state["in_think"] and think_state["tail"]:
                                sentence += think_state["tail"]
                                full_response += think_state["tail"]
                                await self.emit("llm_chunk", think_state["tail"])
                            
                            tail = sentence.strip()
                            if tail and not self.interrupt_event.is_set():
                                cleaned = clean_text(tail)
                                if cleaned:
                                    await self.tts_queue.put(cleaned)
                            
                            await self.tts_queue.put("__END_RESPONSE__")
                            await self.emit("transcript", {"role": "ai", "text": full_response.strip()})
                            
                            if full_response.strip():
                                self.history.append({"role": "user", "content": user_text})
                                self.history.append({"role": "assistant", "content": full_response})
                                if len(self.history) > 60: self.history = self.history[-60:]
                            else:
                                await self.emit("log", "[LLM] Empty response from model")
                                
                        except Exception as e:
                            print(f"LLM Turn Error: {type(e).__name__}: {e}")
                            traceback.print_exc()
            except Exception as outer_e:
                print(f"LLM Outer Worker Error: {outer_e}")
                traceback.print_exc()
                await asyncio.sleep(2)

    async def tts_worker(self):
        """Lab-parity TTS worker.

        Mirrors tts_lab/server.py's api_synth exactly: explicit
        guidance_scale/t_shift, no seed pinning, no peak normalize, no beep,
        no audio_chunk_threshold override. The lab produces stable pitch with
        this path, so any drift left after matching it is upstream (LLM text
        or a real OmniVoice non-determinism) — the SHA log below tells us
        which.
        """
        import hashlib

        print("[TTS] Worker started.")
        while self.is_running:
            try:
                text = await self.tts_queue.get()
                if self.interrupt_event.is_set() or text == "": continue
                if text == "__END_RESPONSE__":
                    self.is_speaking = False
                    await self.emit("status", "idle")
                    continue

                cleaned_text = clean_text(text)
                if not cleaned_text: continue

                text_sha = hashlib.sha1(cleaned_text.encode("utf-8")).hexdigest()[:10]
                # Dynamic prosody: pick guidance_scale by text shape so the
                # voice doesn't sound monotonic turn after turn. Lower scale
                # → looser prosody, more natural/expressive; higher scale →
                # tighter, more committed-sounding delivery. 2.0 is the lab
                # default; we bias up for long replies (hold structure across
                # a paragraph) and down for short acks / emotive tags.
                tl = len(cleaned_text)
                has_emote = any(tag in cleaned_text for tag in ("[laughter]", "[sigh]", "[sniff]", "[dissatisfaction-hnn]"))
                if has_emote:
                    guidance = 1.6
                elif tl < 40:
                    guidance = 1.8
                elif tl > 120:
                    guidance = 2.2
                else:
                    guidance = 2.0
                print(f"[TTS] Synthesizing (text_sha={text_sha} guidance={guidance}): {cleaned_text}")
                try:
                    def _lab_generate():
                        # Pin every RNG right before sampling so the diffusion
                        # sampler (flow-matching noise init, any dropout, any
                        # multinomial decode) starts from the same state on
                        # every turn. Without this, same text → different
                        # audio → audible pitch drift.
                        torch.manual_seed(42)
                        torch.cuda.manual_seed_all(42)
                        np.random.seed(42)
                        import random as _py_random
                        _py_random.seed(42)
                        return self.tts_model.generate(
                            text=cleaned_text,
                            voice_clone_prompt=self.voice_prompt,
                            language="hi",
                            generation_config=OmniVoiceGenerationConfig(
                                num_step=12,
                                guidance_scale=guidance,
                                t_shift=0.1,
                                # Disable internal _generate_chunked: default
                                # audio_chunk_duration is 15s, so any reply
                                # whose synth exceeds 15s gets split into
                                # chunks that each run an independent diffusion
                                # pass — audible as mid-sentence pitch switches.
                                # Our replies are capped to ~30s of audio, well
                                # under one-pass memory budget, so we force
                                # single-pass by raising the threshold.
                                audio_chunk_threshold=600.0,
                                # CRITICAL: disable remove_silence post-proc.
                                # Default True triggers pydub.split_on_silence
                                # with mid_sil=500ms, which trims any pause
                                # ≥0.5s (commas / full stops in longer LLM
                                # replies all produce these) and re-concats
                                # the remaining segments with no crossfade.
                                # If the iterative sampler landed on slightly
                                # different pitch on either side of a comma,
                                # the concat exposes it as a hard jump — the
                                # exact "pitch changes after ','" drift. The
                                # lab never hits it because its preset phrases
                                # are too short to produce 500 ms pauses.
                                postprocess_output=False,
                            ),
                        )
                    t0 = time.perf_counter()
                    audio_list = await asyncio.to_thread(_lab_generate)
                    synth_ms = int((time.perf_counter() - t0) * 1000)
                    audio_24k = audio_list[0]
                    if hasattr(audio_24k, "cpu"):
                        audio_24k = audio_24k.cpu().numpy()
                    audio_24k = np.asarray(audio_24k, dtype=np.float32)

                    peak = float(np.max(np.abs(audio_24k))) if len(audio_24k) else 0.0
                    duration_s = len(audio_24k) / 24000.0
                    audio_sha = hashlib.sha1(audio_24k.tobytes()).hexdigest()[:10]
                    print(
                        f"[TTS] Synth OK text_sha={text_sha} audio_sha={audio_sha} "
                        f"synth_ms={synth_ms} samples={len(audio_24k)} "
                        f"audio_s={duration_s:.2f} peak={peak:.3f}",
                        flush=True,
                    )
                    await self.emit(
                        "log",
                        f"[TTS] Synth OK sha={audio_sha} {synth_ms}ms {duration_s:.2f}s peak={peak:.2f}",
                    )

                    pcm_int16 = (np.clip(audio_24k, -1.0, 1.0) * 32767.0).astype(np.int16)
                    pcm_bytes = pcm_int16.tobytes()

                    self.is_speaking = True
                    await self.emit("status", "speaking")

                    if not self.interrupt_event.is_set():
                        await self.emit("audio_chunk", pcm_bytes)
                except Exception as tts_err:
                    import traceback
                    print(f"[TTS] Gen Fail: {tts_err}")
                    await self.emit("log", f"[ERROR] TTS Failed: {tts_err}")
                    traceback.print_exc()

            except Exception as e:
                print(f"[TTS] Loop Error: {e}")

    async def run(self):
        self.tasks = [
            asyncio.create_task(self.asr_worker(), name="asr_worker"),
            asyncio.create_task(self.llm_worker(), name="llm_worker"),
            asyncio.create_task(self.tts_worker(), name="tts_worker"),
        ]
        try:
            await asyncio.gather(*self.tasks)
        except asyncio.CancelledError:
            print("[Engine] run() cancelled", flush=True)
        except BaseException as e:
            import traceback
            print(f"[Engine] run() BaseException: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
        finally:
            # Per-task postmortem so we know which worker died and why.
            for t in getattr(self, 'tasks', []):
                if t.done():
                    try:
                        exc = t.exception()
                        print(f"[Engine] task={t.get_name()} done={t.done()} cancelled={t.cancelled()} exc={type(exc).__name__ if exc else None}: {exc}", flush=True)
                    except asyncio.CancelledError:
                        print(f"[Engine] task={t.get_name()} was cancelled", flush=True)
                else:
                    print(f"[Engine] task={t.get_name()} still running (will be cancelled)", flush=True)
            await self.cleanup()

    async def cleanup(self):
        print("[Engine] Session ended.")

    def reset_session(self):
        """Reset chat history and clear queues for a fresh turn/caller."""
        print("[Engine] Resetting session history.")
        self.history = []
        self.interrupt_event.clear()
        
        # Drain queues
        while not self.asr_queue.empty(): self.asr_queue.get_nowait()
        while not self.llm_queue.empty(): self.llm_queue.get_nowait()
        while not self.tts_queue.empty(): self.tts_queue.get_nowait()

    def stop(self):
        self.is_running = False
        for task in getattr(self, 'tasks', []):
            task.cancel()
