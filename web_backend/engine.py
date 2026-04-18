import asyncio
import collections
import os
import httpx
import json
import re
import numpy as np
import torch
import time
import uuid
import statistics
from faster_whisper import WhisperModel
from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from fractions import Fraction
from scipy import signal
try:
    from silero_vad import load_silero_vad
    _SILERO_AVAILABLE = True
except Exception as _e:
    _SILERO_AVAILABLE = False
    print(f"[Engine] Silero VAD unavailable: {_e}. Will fall back to RMS threshold.")

# --- Configuration (Hardcoded for Vaani Web) ---
LLM_URL = "http://127.0.0.1:11434/v1/chat/completions"
LLM_MODEL = "gemma3:4b"         # Must match the model pre-warmed in main.py
ENERGY_THRESHOLD = 600          # Legacy (RMS fallback only)
SILENCE_CHUNKS = 3              # With Silero VAD: 3 chunks ≈ 510 ms silence before end-of-turn.
                                # Gives room for natural Hindi pauses that previously cut speech mid-sentence.
MAX_BUFFER_CHUNKS = 250
BARGE_IN_FRAMES = 2              # With Silero VAD: 2 confident speech chunks (~340 ms) to trigger barge-in.
                                # Previously 8 (≈ 1.4 s) because RMS had higher false-positive rate.
VAD_THR_IDLE = 0.5              # Silero VAD threshold when AI is idle — standard value
VAD_THR_SPEAKING = 0.7          # Higher threshold while AI speaks to reject echo / TTS bleed
TTS_CHUNK_SAMPLES = 480         # [Phase 20] 20ms at 24kHz. Universal VoIP standard for crystal clarity.

# Use absolute path for reliability
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF_AUDIO_PATH = os.path.join(ROOT_DIR, "assets", "voices", "ravi_sir.mp3")

def clean_text(text):
    """Zero-Tolerance Tag Guard: Validates and whitelists only supported OmniVoice acoustic tags.
    Strips away ANY other bracketed content to prevent hallucinations.

    Rule: acoustic tags are ENGLISH ONLY (see memory: feedback_emotional_tags_english_only).
    The reply body can be Hindi/Hinglish, but the bracketed tags must match one of the
    English canonical forms below. Hindi variants are deleted, not translated.
    """
    # 1. Supported Tags Whitelist (ENGLISH ONLY — see rule above)
    ALLOWED_TAGS = {"laughter", "sigh", "sniff", "dissatisfaction-hnn"}

    # 2. Extract and Validate all bracketed tags [...]
    def validate_and_protect(match):
        content = match.group(1).lower().strip()
        if content in ALLOWED_TAGS:
            return f"[{content}]"
        return ""  # Delete any non-whitelisted bracket (incl. Hindi/other-language variants)
    
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
    
    # 4. Final Cleanup: Remove any stray brackets that didn't form a valid tag (before restoring placeholders)
    # This prevents the AI from saying "k" or click-sounds for single brackets
    text = re.sub(r'[\[\]]', '', text)

    # Restore original tags
    for placeholder, original in tag_map.items():
        text = text.replace(placeholder, original)
    
    return text.strip()

class TurnLogger:
    """Per-utterance perf collector. Cheap: timestamps + dict.

    A single instance tracks one user turn from VAD-end through first
    audio output, so we can later attribute the latency budget to ASR /
    LLM TTFT / TTS synth / first-audio-out.
    """
    __slots__ = ("session_id", "turn_id", "t", "meta")

    def __init__(self, session_id, turn_id):
        self.session_id = session_id
        self.turn_id = turn_id
        self.t = {}      # stage_name -> perf_counter()
        self.meta = {}   # extra fields included in the [TURN] payload

    def mark(self, stage):
        self.t[stage] = time.perf_counter()

    def add(self, **kw):
        self.meta.update(kw)

    def delta_ms(self, a, b):
        if a in self.t and b in self.t:
            return int((self.t[b] - self.t[a]) * 1000)
        return None

    def summary_payload(self, models_info):
        s = {
            "asr_resample":    self.delta_ms("vad_end", "asr_resample_done"),
            "asr_decode":      self.delta_ms("asr_decode_start", "asr_decode_done"),
            "llm_queue_wait":  self.delta_ms("asr_decode_done", "llm_request_sent"),
            "llm_ttft":        self.delta_ms("llm_request_sent", "llm_first_token"),
            "llm_first_sent":  self.delta_ms("llm_request_sent", "llm_first_sentence"),
            "llm_total":       self.delta_ms("llm_request_sent", "llm_done"),
            "tts_synth_first": self.delta_ms("tts_synth_start", "tts_synth_done"),
            "first_audio_out": self.delta_ms("vad_end", "tts_first_audio_out"),
        }
        s = {k: v for k, v in s.items() if v is not None}
        return {
            "session": self.session_id,
            "turn": self.turn_id,
            **self.meta,
            "stages_ms": s,
            "models": models_info,
        }


class _ModelStore:
    """Singleton: loads heavy GPU models ONCE, reused across all WebSocket connections."""
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loaded = False
            cls._instance.device_info = {}
        return cls._instance

    def _torch_devices(self):
        if not torch.cuda.is_available():
            return []
        try:
            return [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        except Exception:
            return []

    def load(self):
        if self._loaded:
            print("[Engine] Models already loaded, reusing.")
            return

        # --- Co-locate STT + TTS on RTX 3060 (cuda:0) ---
        # Prior "multi-GPU optimized" layout put Whisper large-v3-turbo on the GTX 1650
        # (cuda:1, 4 GB). In practice that was 5-10x slower than cuda:0: 4 GB VRAM is
        # too tight for large-v3-turbo float16 + CUDA scratch, causing paging on every
        # inference (observed: 7-14 s decode). sm_75 (Turing) is also slower than
        # sm_86 (Ampere) for Whisper's attention.
        #
        # int8_float16 quant reduces VRAM by ~40% with negligible accuracy loss, so
        # Whisper (~2 GB) + OmniVoice (~4-6 GB) fit comfortably in 12 GB.
        stt_fallback_reason = None
        try:
            print("[Engine] Loading models on RTX 3060 (cuda:0, int8_float16)...")
            self.asr_model = WhisperModel(
                "large-v3-turbo",
                device="cuda",
                device_index=0,
                compute_type="int8_float16",
            )
            self.tts_model = OmniVoice.from_pretrained(
                "k2-fsa/OmniVoice",
                device_map="cuda:0",
                dtype=torch.float16,
                load_asr=False,
            )
            self.voice_prompt = self.tts_model.create_voice_clone_prompt(
                ref_audio=REF_AUDIO_PATH, preprocess_prompt=True
            )
            stt_device = "cuda:0"
            stt_compute = "int8_float16"
            tts_device_map = "cuda:0"
            tts_dtype = "float16"
            print("[Engine] GPU models loaded on cuda:0.")
        except Exception as e:
            stt_fallback_reason = f"{type(e).__name__}: {e}"[:300]
            print(f"[Engine] cuda:0 load failed ({stt_fallback_reason}). Falling back to CPU (much slower)...")
            self.asr_model = WhisperModel(
                "large-v3-turbo", device="cpu", compute_type="int8"
            )
            self.tts_model = OmniVoice.from_pretrained(
                "k2-fsa/OmniVoice",
                device_map="cpu",
                dtype=torch.float32,
                load_asr=False,
            )
            self.voice_prompt = self.tts_model.create_voice_clone_prompt(
                ref_audio=REF_AUDIO_PATH, preprocess_prompt=True
            )
            stt_device = "cpu"
            stt_compute = "int8"
            tts_device_map = "cpu"
            tts_dtype = "float32"

        # --- Warmup: 1 s of silence. On real GPU this completes in <500 ms.
        # If the model silently fell back to CPU, warmup will take several seconds and
        # we'll see it in the [BOOT] manifest (stt_warmup_ms) — much easier to diagnose
        # than staring at transcribe timings.
        warmup_ms = -1
        try:
            warmup_start = time.perf_counter()
            warmup_audio = np.zeros(16000, dtype=np.float32)
            segs, _ = self.asr_model.transcribe(
                warmup_audio, beam_size=1, language="en",
                without_timestamps=True, vad_filter=False,
            )
            _ = list(segs)  # drain generator to actually run inference
            warmup_ms = int((time.perf_counter() - warmup_start) * 1000)
        except Exception as e:
            print(f"[Engine] ASR warmup failed: {e}")

        # --- Load Silero VAD (shared across all sessions). ONNX variant is ~2 MB
        # and runs CPU at ~0.1 ms per 32 ms window — effectively free.
        self.vad_model = None
        vad_info = "rms-fallback"
        vad_warmup_ms = -1
        if _SILERO_AVAILABLE:
            try:
                vad_load_start = time.perf_counter()
                self.vad_model = load_silero_vad(onnx=True)
                # Warmup: run a few windows so the ONNX runtime caches kernels.
                dummy = torch.zeros(512, dtype=torch.float32)
                for _ in range(3):
                    _ = self.vad_model(dummy, 16000)
                vad_warmup_ms = int((time.perf_counter() - vad_load_start) * 1000)
                vad_info = "silero-v6 (onnx, CPU)"
                print(f"[Engine] Silero VAD loaded in {vad_warmup_ms}ms.")
            except Exception as e:
                print(f"[Engine] Silero VAD load failed: {e}. Falling back to RMS.")

        self._loaded = True
        self.device_info = {
            "stt_model": "large-v3-turbo",
            "stt_device": stt_device,
            "stt_compute_type": stt_compute,
            "stt_warmup_ms": warmup_ms,
            "stt_fallback_reason": stt_fallback_reason,
            "tts_model": "k2-fsa/OmniVoice",
            "tts_device_map": tts_device_map,
            "tts_dtype": tts_dtype,
            "vad": vad_info,
            "vad_warmup_ms": vad_warmup_ms,
            "torch_cuda_available": torch.cuda.is_available(),
            "torch_device_count": torch.cuda.device_count(),
            "torch_devices": self._torch_devices(),
        }
        print(f"[Engine] Models ready. ASR warmup={warmup_ms}ms on {stt_device}. VAD: {vad_info}.")

_models = _ModelStore()

class WebAssistant:
    def __init__(self, event_handler=None, input_sampling_rate=24000):
        self.event_handler = event_handler
        self.input_sampling_rate = input_sampling_rate
        _models.load()
        self.asr_model = _models.asr_model
        self.tts_model = _models.tts_model
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
        self.history = []  # Added persistent memory for context tracking
        self.voice_prompt = _models.voice_prompt
        self.initial_prompt = "Natural Hindi/Hinglish: नमस्ते, क्या हाल है? मैं ठीक हूँ। Hello, how are you?"
        self.is_running = True
        self.is_speaking = False
        self.speaking_start_time = 0    # Track when AI starts speaking for barge-in grace period
        self.start_time = time.time()  # Track session start for barge-in suppression
        self.interrupt_event = asyncio.Event()
        self.asr_queue = asyncio.Queue(maxsize=100)
        self.llm_queue = asyncio.Queue(maxsize=20)
        self.tts_queue = asyncio.Queue(maxsize=50)
        self.barge_in_lock_until = 0    # [Phase 15] Hard-Lock timer for echo immunity

        # --- Per-turn observability (see TurnLogger above) ---
        # session_id is short and stable for the lifetime of this WebSocket.
        # _turns holds in-flight TurnLogger objects keyed by turn_id; entries
        # are popped after the [TURN] summary is emitted on END_RESPONSE.
        self.session_id = uuid.uuid4().hex[:8]
        self._turn_counter = 0
        self._turns = {}            # turn_id -> TurnLogger
        self._turn_history = []     # list of stages_ms dicts for session aggregate

        # --- Silero VAD per-session state ---
        # Silero consumes exactly 512 samples @ 16 kHz per call. Our input chunks
        # don't divide evenly, so we buffer the sub-window residual here.
        self._vad_model = _models.vad_model
        self._vad_residual_16k = np.zeros(0, dtype=np.float32)

    def reset_session(self):
        """Reset session state for a new call."""
        print("[Engine] Resetting session state for new call.", flush=True)
        self.start_time = time.time()
        self.is_speaking = False
        self.speaking_start_time = 0
        self.interrupt_event.clear()
        # Drain queues
        while not self.asr_queue.empty(): self.asr_queue.get_nowait()
        while not self.llm_queue.empty(): self.llm_queue.get_nowait()
        while not self.tts_queue.empty(): self.tts_queue.get_nowait()
        # Clear VAD residual window so next call starts clean
        self._vad_residual_16k = np.zeros(0, dtype=np.float32)

    def _vad_probability(self, audio_16k_f32: np.ndarray) -> float:
        """Silero VAD over 512-sample (32 ms) windows. Returns max probability
        across all complete windows produced by this chunk (residual is carried
        into the next call). 0.0 if VAD unavailable."""
        if self._vad_model is None:
            return 0.0
        self._vad_residual_16k = np.concatenate([self._vad_residual_16k, audio_16k_f32])
        max_prob = 0.0
        while self._vad_residual_16k.size >= 512:
            window = self._vad_residual_16k[:512]
            self._vad_residual_16k = self._vad_residual_16k[512:]
            tensor = torch.from_numpy(window.copy())
            try:
                prob = float(self._vad_model(tensor, 16000).item())
            except Exception:
                prob = 0.0
            if prob > max_prob:
                max_prob = prob
        return max_prob

    async def emit(self, msg_type, data):
        if msg_type == "status":
            print(f"[Status] -> {data}")
            if data == "speaking":
                self.speaking_start_time = time.time()

        if self.event_handler:
            await self.event_handler(msg_type, data)

    async def asr_worker(self):
        print("[ASR] Worker started.")
        buffer = []
        pre_speech_buffer = collections.deque(maxlen=15) # Approx ~300-500ms of history
        silence_count = 0
        barge_in_count = 0
        is_speech_active = False

        asr_chunk_count = 0
        was_speech = False
        vad_available = self._vad_model is not None
        await self.emit(
            "log",
            f"[ASR] Worker ready. VAD={'silero' if vad_available else 'rms-fallback'} "
            f"thr_idle={VAD_THR_IDLE} thr_speaking={VAD_THR_SPEAKING} "
            f"silence_chunks={SILENCE_CHUNKS} barge_in_frames={BARGE_IN_FRAMES}"
        )

        while self.is_running:
            try:
                chunk_bytes = await self.asr_queue.get()
                asr_chunk_count += 1
                audio_float32 = np.frombuffer(chunk_bytes, dtype=np.int16).astype(np.float32) / 32768.0

                # --- Voice Activity Detection (Silero) ---
                # Resample 24 kHz → 16 kHz for Silero. Very cheap (numeric factor 2/3).
                # Falls back to RMS threshold if Silero unavailable.
                if vad_available:
                    audio_16k_vad = signal.resample_poly(audio_float32, 2, 3)
                    speech_prob = self._vad_probability(audio_16k_vad)
                    threshold = VAD_THR_SPEAKING if self.is_speaking else VAD_THR_IDLE
                    is_speech = speech_prob > threshold
                else:
                    # RMS fallback (approx old behaviour, no calibration).
                    speech_prob = float(np.sqrt(np.mean(audio_float32 ** 2)))
                    threshold = 0.15 if self.is_speaking else 0.035
                    is_speech = speech_prob > threshold

                # Grant a one-time echo grace on the first chunk so the AI greeting
                # has room to start without being treated as user barge-in.
                if asr_chunk_count == 1:
                    self.barge_in_lock_until = time.time() + 1.5

                # rms kept only as a secondary diagnostic (shown in traces).
                rms = float(np.sqrt(np.mean(audio_float32 ** 2)))

                # Periodic ASR trace: one line every ~6s so we can see the pipeline breathing.
                if asr_chunk_count % 20 == 0:
                    await self.emit(
                        "log",
                        f"[ASR-TRACE] p={speech_prob:.2f} thr={threshold:.2f} rms={rms:.3f} "
                        f"speech={is_speech} speaking={self.is_speaking} buf={len(buffer)}"
                    )

                # Edge-triggered transition log.
                if is_speech and not was_speech:
                    await self.emit("log", f"[ASR] ⇧ speech-start p={speech_prob:.2f} thr={threshold:.2f}")
                elif was_speech and not is_speech and is_speech_active:
                    await self.emit("log", f"[ASR] ⇩ speech-gap p={speech_prob:.2f} silence_count={silence_count}")
                was_speech = is_speech

                if self.is_speaking and is_speech:
                    # [FAST] Ignore barge-in for first 2.5s of call (was 5s)
                    # Greeting is short enough to be done by then.
                    start_time = getattr(self, 'start_time', time.time())
                    if (time.time() - start_time) < 2.5:
                        continue 
                        
                    # [FAST] Echo gate reduced from 1.2s → 0.5s
                    # 0.5s still blocks most analog echo while allowing fast barge-in
                    if (time.time() - self.speaking_start_time) < 0.5:
                        continue
                    
                    # [Phase 15 Hard-Lock]
                    # Extreme protection for the start of the sentence
                    if time.time() < self.barge_in_lock_until:
                        continue

                    barge_in_count += 1
                    if barge_in_count >= BARGE_IN_FRAMES:
                        await self.emit("log", f"[Barge-in] DETECTED p={speech_prob:.2f} frames={barge_in_count}")
                        self.interrupt_event.set()
                        # Flush queues
                        while not self.tts_queue.empty(): self.tts_queue.get_nowait()
                        while not self.llm_queue.empty(): self.llm_queue.get_nowait()
                        await self.emit("status", "interrupted")
                        self.is_speaking = False
                        barge_in_count = 0
                        # Yield so TTS worker sees the interrupt and breaks its loop
                        await asyncio.sleep(0.05)
                        # Clear interrupt so the NEXT response can be spoken
                        self.interrupt_event.clear()
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

                # End-of-turn: Silero VAD is accurate enough that plain silence-count
                # gating is sufficient (no RMS "snap-trigger" hack needed).
                if is_speech_active and silence_count >= SILENCE_CHUNKS:
                    await self.emit("log", f"[ASR] VAD trigger silence={silence_count} buf={len(buffer)}")

                    # [OBS] Open a new turn at the VAD edge. All downstream stage
                    # marks (LLM TTFT, TTS synth, first audio out) hang off this.
                    self._turn_counter += 1
                    turn = TurnLogger(self.session_id, self._turn_counter)
                    turn.mark("vad_end")
                    self._turns[turn.turn_id] = turn
                    # Cap retained turns; barge-in can leave entries without END_RESPONSE.
                    if len(self._turns) > 8:
                        for old_id in sorted(self._turns.keys())[:-8]:
                            self._turns.pop(old_id, None)

                    audio_bytes = b"".join(buffer)
                    buffer = [] # [Phase 19.2] Reset immediately to avoid double-processing
                    silence_count = 0
                    is_speech_active = False
                    self._max_rms_in_turn = 0.01
                    audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                    # Efficient Resampling: 24kHz -> 16kHz (ASR)
                    if self.input_sampling_rate == 24000:
                        # Simple and fast decimation for 24k to 16k (Factor 1.5)
                        resample_ratio = Fraction(16000, 24000)
                        audio_16k = await asyncio.to_thread(
                            signal.resample_poly, audio_np, resample_ratio.numerator, resample_ratio.denominator
                        )
                    elif self.input_sampling_rate == 8000:
                        # Upsampling 8k -> 16k is just linear interpolation or repeat
                        audio_16k = await asyncio.to_thread(np.repeat, audio_np, 2)
                    else:
                        resample_ratio = Fraction(16000, self.input_sampling_rate)
                        audio_16k = await asyncio.to_thread(
                            signal.resample_poly, audio_np, resample_ratio.numerator, resample_ratio.denominator
                        )
                    turn.mark("asr_resample_done")

                    # [AUDIO] Trust Whisper's internal mel-spec log normalization to
                    # handle volume. The previous `audio/peak` normalization amplified
                    # quiet speech to full scale (along with its noise floor), producing
                    # the gibberish hallucinations we observed. Only boost VERY quiet
                    # audio so it's not below Whisper's no-speech threshold.
                    peak = float(np.max(np.abs(audio_16k)))
                    if 0.005 < peak < 0.05:
                        audio_16k = audio_16k * (0.1 / peak)

                    padding = np.zeros(int(16000 * 0.2), dtype=np.float32)
                    audio_padded = np.concatenate([padding, audio_16k, padding])

                    # Tuning rationale:
                    #   vad_filter=True        — Silero VAD inside faster-whisper skips silence
                    #                            and suppresses hallucinations on fragmented clips.
                    #   beam_size=5            — default for accuracy. With VAD + int8_float16
                    #                            on RTX 3060 this is still <500 ms on short clips.
                    #   without_timestamps=True — we don't use word timings anyway; saves compute.
                    #   temperature=0.0        — greedy decoding; fastest path.
                    #   language="hi"          — force Hindi. Auto-detect was flipping to
                    #                            Korean/Indonesian on short utterances with low
                    #                            confidence; bad audio clipping is gone so the
                    #                            old gibberish failure mode doesn't return.
                    turn.mark("asr_decode_start")
                    segments, asr_info = await asyncio.to_thread(
                        self.asr_model.transcribe,
                        audio_padded,
                        beam_size=5,
                        language="hi",
                        condition_on_previous_text=False,
                        vad_filter=True,
                        vad_parameters=dict(min_silence_duration_ms=500),
                        without_timestamps=True,
                        temperature=0.0,
                    )
                    text = "".join([s.text for s in segments]).strip()
                    turn.mark("asr_decode_done")
                    detected_lang = getattr(asr_info, "language", None) or "?"
                    lang_prob = float(getattr(asr_info, "language_probability", 0.0) or 0.0)
                    turn.add(
                        asr_text=text[:80],
                        asr_lang=detected_lang,
                        asr_lang_prob=round(lang_prob, 3),
                        asr_beam=5,
                        asr_samples=int(len(audio_padded)),
                        asr_peak=float(peak),
                    )
                    await self.emit(
                        "log",
                        f"[ASR] decode {turn.delta_ms('asr_decode_start','asr_decode_done')}ms "
                        f"lang={detected_lang}({lang_prob:.2f}) beam=1 samples={len(audio_padded)} "
                        f"chars={len(text)} peak={peak:.3f}"
                    )
                    # Removed junk patterns - they are too aggressive for noisy telephony links

                    if text and len(text) >= 1:
                        await self.emit("log", f"[ASR] Whisper → '{text}'")
                        await self.emit("transcript", {"role": "user", "text": text})
                        await self.llm_queue.put((text, turn.turn_id))
                    else:
                        await self.emit("log", f"[ASR] Whisper → (empty, {len(audio_padded)} samples)")
                        # No downstream consumer — drop the turn so it doesn't linger.
                        self._turns.pop(turn.turn_id, None)
                    buffer = []
                    silence_count = 0
                    is_speech_active = False
            except Exception as e:
                print(f"ASR Error: {e}")

    async def llm_worker(self):
        print("[LLM] Worker started.")
        async with httpx.AsyncClient(timeout=30.0) as client:
            while self.is_running:
                item = await self.llm_queue.get()
                # Backward-compat: ASR pushes (text, turn_id); legacy callers may push str.
                if isinstance(item, tuple):
                    user_text, turn_id = item
                else:
                    user_text, turn_id = item, None
                turn = self._turns.get(turn_id) if turn_id is not None else None
                await self.emit("log", f"[LLM] → req: {user_text[:80]!r}")
                llm_start = time.time()
                payload = {
                    "model": LLM_MODEL,
                    "messages": [
                        {
                            "role": "system", 
                            "content": (
                                f"IDENTITY: Your name is {self.active_voice_metadata.get('name', 'AI')}. You are {self.active_voice_metadata.get('age', 'unknown')} years old. {self.active_voice_metadata.get('about', '')}\n\n"
                                f"PERSONA: Speak as {self.active_voice_metadata.get('gender', 'neutral')}. Style: {self.active_voice_metadata.get('style', 'conversational')}.\n"
                                f"FAMOUS DIALOGUES/CATCHPHRASES: {self.active_voice_metadata.get('dialogues', '')}\n"
                                "(Use your dialogues naturally if appropriate, do not force them every time.)\n\n"
                                "SCRIPT RULES (STRICT):\n"
                                "1. Speak ONLY in DEVANAGARI HINDI.\n"
                                "2. NEVER use English characters (A-Z) except for verified acoustic tags.\n"
                                "3. ANTI-REPETITION: Do NOT start every sentence with 'हाँ, बिलकुल', 'मुझे समझ आ गया', or 'ठीक है'. Be direct and varied.\n"
                                "4. CONTEXT ADHERENCE: Use the Conversation History to stay on topic. If the user mentions a movie title, do NOT treat it as a literal word.\n\n"
                                "ACOUSTIC TAGS (ENGLISH ONLY):\n"
                                "Use ONLY these 4 tags, always in English, never translated:\n"
                                "  [laughter], [sigh], [sniff], [dissatisfaction-hnn]\n"
                                "Even though your reply is Hindi/Devanagari, the tags MUST stay English.\n"
                                "NEVER write [हँसी], [आह], [हम्म], [Haha], [Smiling], or any other bracket.\n\n"
                                "MISSION: Be extremely helpful, witty, and concise."
                            )
                        },
                        *self.history[-20:], # [CONTEXT] Last 20 exchanges (was 10) for better memory
                        {"role": "user", "content": user_text}
                    ],
                    "stream": True,
                    # num_gpu=99 offloads all layers to the GPU (the model is already in
                    # RTX 3060 VRAM). num_ctx=1024 keeps the KV cache ~128 MB so we fit
                    # within the remaining VRAM alongside Whisper + OmniVoice.
                    # Must match main.py prewarm, otherwise Ollama reloads the model.
                    "options": {"num_predict": 200, "temperature": 0.5, "num_gpu": 99, "num_ctx": 1024}
                }
                sentence = ""
                full_response = ""
                # [STATE PURGE] Clear ASR history whenever AI starts a new sentence
                # to prevent stale noise/echo from causing immediate interruption.
                while not self.asr_queue.empty():
                    try: self.asr_queue.get_nowait()
                    except: break
                is_first_chunk = True
                await self.emit("status", "thinking")
                if turn:
                    turn.mark("llm_request_sent")
                    turn.add(llm_model=LLM_MODEL, llm_num_gpu=99, llm_num_ctx=1024)
                try:
                    async with client.stream("POST", LLM_URL, json=payload) as response:
                        async for line in response.aiter_lines():
                            if self.interrupt_event.is_set():
                                await self.emit("status", "interrupted")
                                break
                            if line.startswith("data: "):
                                if "[DONE]" in line: break
                                body = json.loads(line[6:])
                                chunk = body.get("choices", [{}])[0].get("delta", {}).get("content", "")
                                if chunk and turn and "llm_first_token" not in turn.t:
                                    turn.mark("llm_first_token")
                                    await self.emit(
                                        "log",
                                        f"[LLM] TTFT {turn.delta_ms('llm_request_sent','llm_first_token')}ms "
                                        f"(model={LLM_MODEL}, num_gpu=99, num_ctx=1024)"
                                    )
                                sentence += chunk
                                full_response += chunk
                                await self.emit("llm_chunk", chunk)

                                # Trigger logic tuned for LOW first-audio latency:
                                #   First fragment fires as early as possible so TTS can kick off
                                #   while the rest of the LLM response is still streaming.
                                #     - On end-of-sentence punctuation (. ? ! ।) after ≥3 words.
                                #     - On comma/colon/semicolon after ≥5 words.
                                #     - Or every ≥12 words if the LLM is still producing.
                                #   Subsequent fragments use looser rules to keep prosody natural.
                                word_count = len(sentence.split())
                                has_end_punct = any(p in chunk for p in ".?!।\n")
                                has_mid_punct = any(p in chunk for p in ",;:")
                                if is_first_chunk:
                                    trigger = (
                                        (has_end_punct and word_count >= 3)
                                        or (has_mid_punct and word_count >= 5)
                                        or word_count >= 12
                                    )
                                else:
                                    trigger = has_end_punct or word_count >= 30

                                if trigger:
                                    cleaned = clean_text(sentence)
                                    if cleaned and len(cleaned) >= 1:
                                        if turn and "llm_first_sentence" not in turn.t:
                                            turn.mark("llm_first_sentence")
                                        await self.tts_queue.put((cleaned, turn_id))
                                        is_first_chunk = False
                                    sentence = ""
                    if sentence.strip() and not self.interrupt_event.is_set():
                        await self.tts_queue.put((clean_text(sentence), turn_id))
                    await self.tts_queue.put(("__END_RESPONSE__", turn_id))
                    if turn:
                        turn.mark("llm_done")
                        turn.add(llm_chars=len(full_response))
                    await self.emit("log", f"[LLM] ← done in {time.time() - llm_start:.1f}s ({len(full_response)} chars)")
                    
                    display_text = full_response
                    _UI_ALLOWED_TAGS = {"laughter", "sigh", "sniff", "dissatisfaction-hnn"}
                    def ui_whitelist(match):
                        content = match.group(1).lower().strip()
                        # English-only rule: strip anything that isn't one of our canonical tags.
                        return f"[{content}]" if content in _UI_ALLOWED_TAGS else ""
                    display_text = re.sub(r'\[(.*?)\]', ui_whitelist, display_text)
                    await self.emit("transcript", {"role": "ai", "text": display_text})
                    
                    # Store in history — last 40 messages = 20 full turns of conversation memory
                    self.history.append({"role": "user", "content": user_text})
                    self.history.append({"role": "assistant", "content": full_response})
                    if len(self.history) > 40:  # [CONTEXT] Doubled from 20 → 40 for longer memory
                        self.history = self.history[-40:]
                except Exception as e:
                    import traceback
                    print(f"LLM Error: {e}")
                    traceback.print_exc()
                    await self.emit("log", f"[LLM] ERROR: {type(e).__name__}: {e}")

    async def tts_worker(self):
        """Sequential TTS Worker: simple, robust, and traceable."""
        print("[TTS] Worker started.")
        while self.is_running:
            try:
                item = await self.tts_queue.get()
                # Accept both legacy strings (auto-greet from main.py) and (text, turn_id) tuples.
                if isinstance(item, tuple):
                    text, turn_id = item
                else:
                    text, turn_id = item, None
                turn = self._turns.get(turn_id) if turn_id is not None else None

                if self.interrupt_event.is_set() or text == "": continue
                if text == "__END_RESPONSE__":
                    # [FAST] Grace period reduced 0.8s → 0.3s
                    # 0.3s is enough for echo to clear on digital lines. Phone line echo
                    # is already handled by the 0.5s echo gate in asr_worker.
                    await asyncio.sleep(0.3)
                    self.is_speaking = False
                    self._is_first_audio_in_session = False
                    await self.emit("status", "idle")
                    # [OBS] Turn complete — emit summary and retire the TurnLogger.
                    if turn is not None:
                        try:
                            payload = turn.summary_payload(_models.device_info)
                            await self.emit("log", f"[TURN] {json.dumps(payload, ensure_ascii=False)}")
                            self._turn_history.append(payload.get("stages_ms", {}))
                        except Exception as e:
                            await self.emit("log", f"[TURN-ERR] {type(e).__name__}: {e}")
                        self._turns.pop(turn_id, None)
                    continue

                cleaned_text = clean_text(text)
                if not cleaned_text: continue

                # Safer print for Windows console
                try:
                    print(f"[TTS] Synthesizing: {cleaned_text}")
                except UnicodeEncodeError:
                    print(f"[TTS] Synthesizing: [Hindi Text Content]")
                try:
                    if turn and "tts_synth_start" not in turn.t:
                        turn.mark("tts_synth_start")
                    audio_list = await asyncio.to_thread(
                        self.tts_model.generate,
                        text=cleaned_text, voice_clone_prompt=self.voice_prompt,
                        language="hi", generation_config=OmniVoiceGenerationConfig(num_step=8) # 8 = quality sweet-spot for voice-clone fidelity
                    )
                    audio_24k = audio_list[0]
                    if turn and "tts_synth_done" not in turn.t:
                        turn.mark("tts_synth_done")
                        turn.add(
                            tts_num_step=8,
                            tts_in_chars=len(cleaned_text),
                            tts_out_samples=int(len(audio_24k)),
                        )
                        await self.emit(
                            "log",
                            f"[TTS] synth {turn.delta_ms('tts_synth_start','tts_synth_done')}ms "
                            f"num_step=8 in_chars={len(cleaned_text)} out_samples={len(audio_24k)}"
                        )
                    
                    # Gain Control: 1.0 (Unity Gain) is cleanest for digital paths
                    audio_24k = audio_24k * 1.0
                    
                    vol = np.max(np.abs(audio_24k)) if len(audio_24k) > 0 else 0
                    print(f"[TTS] Success: {len(audio_24k)} samples. Peak Vol: {vol:.4f}", flush=True)
                    try:
                        await self.emit("log", f"[TTS] Synth OK: {len(audio_24k)} samples, Vol: {vol:.2f}")
                    except UnicodeEncodeError:
                        await self.emit("log", f"[TTS] Synth OK: {len(audio_24k)} samples")

                    # --- Diagnostic Beep (First time only) ---
                    if not getattr(self, '_beep_sent', False):
                        beep_samples = int(24000 * 0.15) # 150ms
                        t = np.linspace(0, 0.15, beep_samples, False)
                        beep = 0.2 * np.sin(440 * 2 * np.pi * t) # 440Hz
                        audio_24k = np.concatenate([beep, np.zeros(1200), audio_24k])
                        self._beep_sent = True

                    # Add gap and convert to PCM with safe clipping
                    silence_gap = np.zeros(int(24000 * 0.3), dtype=np.float32)
                    audio_full = np.concatenate([audio_24k, silence_gap])
                    pcm_int16 = (np.clip(audio_full, -1, 1) * 32767).astype(np.int16)
                    pcm_bytes = pcm_int16.tobytes()

                    # --- [FAST] Speaking Lock ---
                    self.is_speaking = True
                    # First audio: 2s lock (was 4s). Enough to protect greeting from echo.
                    if getattr(self, '_is_first_audio_in_session', True):
                        self.barge_in_lock_until = max(self.barge_in_lock_until, time.time() + 2.0)
                    else:
                        # Mid-conversation: 1.0s lock (was 2.0s) for snappy barge-in
                        self.barge_in_lock_until = max(self.barge_in_lock_until, time.time() + 1.0)
                    
                    await self.emit("status", "speaking")
                    
                    # Split into micro-chunks (50ms) for jitter resistance
                    chunk_size = TTS_CHUNK_SAMPLES * 2
                    for offset in range(0, len(pcm_bytes), chunk_size):
                        if self.interrupt_event.is_set(): break
                        chunk = pcm_bytes[offset:offset + chunk_size]
                        if turn and "tts_first_audio_out" not in turn.t:
                            turn.mark("tts_first_audio_out")
                            await self.emit(
                                "log",
                                f"[TTS] first audio out at "
                                f"+{turn.delta_ms('vad_end','tts_first_audio_out')}ms from VAD"
                            )
                        await self.emit("audio_chunk", chunk)
                except Exception as tts_err:
                    import traceback
                    print(f"[TTS] Gen Fail: {tts_err}")
                    await self.emit("log", f"[ERROR] TTS Failed: {tts_err}")
                    traceback.print_exc()

            except Exception as e:
                print(f"[TTS] Loop Error: {e}")

    async def _state_tracer(self):
        """Periodic snapshot of pipeline state, visible in the LogViewer."""
        while self.is_running:
            try:
                await asyncio.sleep(2.0)
                lock_remaining = max(0.0, self.barge_in_lock_until - time.time())
                await self.emit(
                    "log",
                    f"[STATE] speaking={self.is_speaking} "
                    f"asr_q={self.asr_queue.qsize()} "
                    f"llm_q={self.llm_queue.qsize()} "
                    f"tts_q={self.tts_queue.qsize()} "
                    f"barge_lock={lock_remaining:.1f}s"
                )
            except asyncio.CancelledError:
                raise
            except Exception as e:
                print(f"[STATE-TRACER] error: {e}")

    async def run(self):
        self.start_time = time.time()
        print(f"[Engine] Starting workers for session {id(self)}...", flush=True)
        self.tasks = [
            asyncio.create_task(self.asr_worker()),
            asyncio.create_task(self.llm_worker()),
            asyncio.create_task(self.tts_worker()),
            asyncio.create_task(self._state_tracer()),
        ]
        print(f"[Engine] Workers created. Awaiting gather...")
        try:
            await asyncio.gather(*self.tasks)
        except asyncio.CancelledError:
            pass
        finally:
            await self.cleanup()

    async def cleanup(self):
        print("[Engine] Session ended.")

    def session_summary(self):
        """Aggregate per-stage timings across all completed turns this session."""
        stages = ["asr_decode", "llm_ttft", "llm_total", "tts_synth_first", "first_audio_out"]
        agg = {}
        for stage in stages:
            vals = [t.get(stage) for t in self._turn_history if t.get(stage) is not None]
            if not vals:
                continue
            sorted_vals = sorted(vals)
            p95_idx = max(0, int(len(sorted_vals) * 0.95) - 1) if len(sorted_vals) >= 2 else 0
            agg[stage] = {
                "n": len(vals),
                "p50": int(statistics.median(vals)),
                "p95": int(sorted_vals[p95_idx]),
                "max": int(max(vals)),
            }
        return {
            "session": self.session_id,
            "turns": len(self._turn_history),
            "session_seconds": int(time.time() - self.start_time),
            "stages_p50_p95_ms": agg,
        }

    def stop(self):
        self.is_running = False
        for task in getattr(self, 'tasks', []):
            task.cancel()
