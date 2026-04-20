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
# Default model if not specified — must match prewarm in main.py
DEFAULT_LLM_MODEL = "gemma3:4b"  
ENERGY_THRESHOLD = 600          # Legacy (RMS fallback only)
SILENCE_CHUNKS = 6              # 6×170ms = ~1.0s silence before end-of-turn fires.
                                # Increased for higher stability in ASR detection.
MAX_BUFFER_CHUNKS = 250
BARGE_IN_FRAMES = 1              # With Silero VAD: one high-confidence speech chunk (~170 ms)
                                # is enough to trigger barge-in. Silero + VAD_THR_SPEAKING=0.7
                                # rejects coughs/clicks via probability, not frame-count.
# Hysteresis — prevents `is_speech` flipping on chunks whose probability sits on the threshold,
# which causes the UI status to thrash between "listening" and "idle".
VAD_ENTER_IDLE = 0.5             # Enter speech when prob rises above this (AI idle).
VAD_EXIT_IDLE = 0.3              # Stay in speech until prob drops below this (AI idle).
# Lowered enter/exit thresholds during AI speech: now that the frontend pipes
# TTS through a dedicated <audio> element, browser AEC subtracts the AI voice
# cleanly and the mic *actually* carries the user's voice during playback.
# A more sensitive trigger here gives Gemini-style "interrupt the moment I
# start talking" feel without false-firing on residual echo.
VAD_ENTER_SPEAKING = 0.35
VAD_EXIT_SPEAKING = 0.25
ECHO_GATE_SEC = 0.15            # Faster: AEC stabilises quickly with the new playback graph.
HARD_LOCK_FIRST_AUDIO_SEC = 0.4 # Greeting protection — reduced for Gemini responsiveness.
HARD_LOCK_MID_TURN_SEC = 0.1    # Mid-conversation lock. User can cut in after 0.1 s.
STATUS_DEDUP_MS = 200            # Skip duplicate status emissions within this window.
TTS_CHUNK_SAMPLES = 480         # [Phase 20] 20ms at 24kHz. Universal VoIP standard for crystal clarity.

# Use absolute path for reliability
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF_AUDIO_PATH = os.path.join(ROOT_DIR, "assets", "voices", "ravi_sir.mp3")

def _strip_repetition_hallucinations(text: str) -> str:
    """Drop Whisper's common Hindi repetition-loop hallucinations.

    Symptoms we've seen: 'हुआ हुआ हुआ हुआ', 'हाँ हाँ हाँ हाँ', 'क्या क्या क्या'.
    These slip past `compression_ratio_threshold` when the loop is short and the
    overall transcript is also short. Heuristic: if ≥3 of the last 4 whitespace
    tokens are the same word, strip the loop and keep at most one occurrence;
    if the *entire* utterance is just one repeated word ≥3 times, drop it as
    pure noise (a single word in a real reply is fine — only repeats are noise).
    """
    if not text:
        return text
    tokens = text.split()
    if len(tokens) < 3:
        return text
    # Whole-utterance pure repetition → drop.
    if len(set(tokens)) == 1 and len(tokens) >= 3:
        return ""
    # Trailing repetition tail → keep one copy.
    last = tokens[-1]
    tail = 0
    for tok in reversed(tokens):
        if tok == last:
            tail += 1
        else:
            break
    if tail >= 3:
        tokens = tokens[: len(tokens) - tail] + [last]
    return " ".join(tokens).strip()


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
        # ACCURACY-FIRST CONFIG (was: large-v3-turbo + int8_float16, accuracy compromise):
        #   • Model `large-v3` (not turbo) — turbo is English-optimised; for Hindi
        #     it produced repetition hallucinations like 'हुआ हुआ हुआ हुआ' and
        #     misheard short utterances. large-v3 is ~30% more accurate on Hindi
        #     and only ~1.5x slower on RTX 3060.
        #   • Compute `float16` (not int8_float16) — int8 quant degrades non-English
        #     output noticeably; float16 is full precision for the GPU path. VRAM
        #     cost ~3 GB vs ~1.5 GB; fits easily in 12 GB alongside OmniVoice.
        #   • Fallback chain: large-v3 fp16 → large-v3 int8_fp16 → CPU int8.
        # --- Balanced Multi-GPU Config (Phase 17) ---
        # GPU 0 (RTX 3060, 12GB): Reserved for Omni TTS + Ollama LLM.
        # GPU 1 (GTX 1650, 4GB): Assigned to Whisper ASR.
        stt_fallback_reason = None
        stt_model_name = "turbo"
        stt_compute = "int8" # [Phase 17] Peak stability for GTX 1650 (No Tensor Cores)
        try:
            print(f"[Engine] Loading Whisper {stt_model_name} on GPU 1 (GTX 1650, {stt_compute})...")
            self.asr_model = WhisperModel(
                stt_model_name, device="cuda", device_index=1, compute_type=stt_compute,
            )
            stt_device = "cuda:1"
        except Exception as e1:
            stt_fallback_reason = f"GPU 1 load fail: {type(e1).__name__}: {e1}"[:200]
            print(f"[Engine] {stt_fallback_reason} — trying GPU 0 as last resort…")
            try:
                self.asr_model = WhisperModel(
                    stt_model_name, device="cuda", device_index=0, compute_type="int8_float16",
                )
                stt_device = "cuda:0"
            except Exception as e2:
                print(f"[Engine] GPU load failed entirely. CPU fallback…")
                stt_compute = "int8"
                self.asr_model = WhisperModel(stt_model_name, device="cpu", compute_type=stt_compute)
                stt_device = "cpu"

        try:
            self.tts_model = OmniVoice.from_pretrained(
                "k2-fsa/OmniVoice",
                device_map="cuda:0",
                dtype=torch.float16,
                load_asr=False,
            )
            self.voice_prompt = self.tts_model.create_voice_clone_prompt(
                ref_audio=REF_AUDIO_PATH, preprocess_prompt=True
            )
            tts_device_map = "cuda:0"
            tts_dtype = "float16"
            print(f"[Engine] Loaded: STT={stt_model_name}/{stt_compute} on {stt_device}, TTS on cuda:0.")
        except Exception as e:
            stt_fallback_reason = (stt_fallback_reason or "") + f" | TTS GPU fail: {e}"[:200]
            print(f"[Engine] TTS GPU load failed: {e}. Falling back to CPU TTS (much slower)...")
            self.tts_model = OmniVoice.from_pretrained(
                "k2-fsa/OmniVoice",
                device_map="cpu",
                dtype=torch.float32,
                load_asr=False,
            )
            self.voice_prompt = self.tts_model.create_voice_clone_prompt(
                ref_audio=REF_AUDIO_PATH, preprocess_prompt=True
            )
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
            "stt_model": stt_model_name,
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
        self.llm_model = DEFAULT_LLM_MODEL # [Phase 16] Dynamic model switching support
        self.voice_prompt = _models.voice_prompt
        # --- MULTILANGUAL INDIAN ASR GUARD (Phase 17) ---
        # A dense, diverse Hinglish/Hindi prompt that anchors Whisper against
        # halls/loops. Includes Indian greetings, tech names, cities, and common 
        # conversational fillers (ji, haan, matlav, actually).
        self.initial_prompt = (
            "नमस्ते, हैलो, जी भाई, कैसे हैं आप? मैं बिल्कुल ठीक हूँ। "
            "जी, हाँ, नहीं, ठीक है, अच्छा, सही, गलत, माफ़ कीजिए, धन्यवाद, शुक्रिया। "
            "मेरा नाम, आपका नाम, घर, परिवार, दोस्त, शहर, देश, भारत, दिल्ली, Mumbai, "
            "कल, आज, अभी, बाद में, सुबह, शाम, रात, दिन, भूख, प्यास, खाना, पानी, चाय। "
            "फ्लिपकार्ट, अमेज़न, जोमैटो, स्विगी, पेटीएम, गूगल, यूट्यूब, व्हाट्सऐप, "
            "एक्चुअली, मतलब, शायद, ज़रूरी, काम, पैसा, मार्केट, दुकान, गाड़ी, फ़ोन। "
            "एक, दो, तीन, चार, पाँच, छह, सात, आठ, नौ, दस, सौ, हज़ार, लाख। "
            "क्या आप सुन रहे हैं? मुझे समझ नहीं आया। फिर से बोलिए। "
            "अच्छा ये बताइए, क्या हाल-चाल हैं? सब बढ़िया है।"
        )
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
        self.is_speaking = False
        self.history = []  # [Phase 16] Clear context on reset
        self.interrupt_event.set() # Stop any active loops
        while not self.llm_queue.empty(): self.llm_queue.get_nowait()
        while not self.tts_queue.empty(): self.tts_queue.get_nowait()
        self.interrupt_event.clear()
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
        # Dedup rapid-fire status flips (e.g. VAD briefly flickering on a chunk
        # whose probability is near the threshold). Prevents UI orb thrash.
        if msg_type == "status":
            now_ms = time.perf_counter() * 1000
            last = getattr(self, "_last_status", (None, 0.0))
            if last[0] == data and (now_ms - last[1]) < STATUS_DEDUP_MS:
                return
            self._last_status = (data, now_ms)
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
        in_speech = False  # hysteresis state: stays True until prob drops below exit threshold
        vad_available = self._vad_model is not None
        await self.emit(
            "log",
            f"[ASR] Worker ready. VAD={'silero' if vad_available else 'rms-fallback'} "
            f"enter(idle/spk)={VAD_ENTER_IDLE}/{VAD_ENTER_SPEAKING} "
            f"exit(idle/spk)={VAD_EXIT_IDLE}/{VAD_EXIT_SPEAKING} "
            f"silence_chunks={SILENCE_CHUNKS} barge_frames={BARGE_IN_FRAMES} "
            f"echo_gate={ECHO_GATE_SEC}s lock_first={HARD_LOCK_FIRST_AUDIO_SEC}s "
            f"lock_mid={HARD_LOCK_MID_TURN_SEC}s"
        )

        while self.is_running:
            try:
                chunk_bytes = await self.asr_queue.get()
                asr_chunk_count += 1
                audio_float32 = np.frombuffer(chunk_bytes, dtype=np.int16).astype(np.float32) / 32768.0

                # --- Voice Activity Detection (Silero) with hysteresis ---
                # Rising edge needs a higher probability than falling edge, so chunks
                # whose prob sits near the boundary don't oscillate is_speech.
                # Resample to 16 kHz based on the session's native input rate. The old
                # hardcoded (2,3) ratio was wrong for 8 kHz Asterisk calls, making Silero
                # see a 5 kHz signal labelled as 16 kHz → useless probabilities on phone.
                if vad_available:
                    src_rate = self.input_sampling_rate
                    if src_rate == 16000:
                        audio_16k_vad = audio_float32
                    elif src_rate == 24000:
                        audio_16k_vad = signal.resample_poly(audio_float32, 2, 3)
                    elif src_rate == 8000:
                        audio_16k_vad = signal.resample_poly(audio_float32, 2, 1)
                    else:
                        ratio = Fraction(16000, src_rate)
                        audio_16k_vad = signal.resample_poly(
                            audio_float32, ratio.numerator, ratio.denominator
                        )
                    speech_prob = self._vad_probability(audio_16k_vad)
                    if self.is_speaking:
                        enter_thr, exit_thr = VAD_ENTER_SPEAKING, VAD_EXIT_SPEAKING
                    else:
                        enter_thr, exit_thr = VAD_ENTER_IDLE, VAD_EXIT_IDLE
                    threshold = enter_thr  # reported in the trace line below
                    if in_speech:
                        in_speech = speech_prob > exit_thr
                    else:
                        in_speech = speech_prob > enter_thr
                    is_speech = in_speech
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

                # Edge-triggered transition log + UI status.
                if is_speech and not was_speech:
                    await self.emit("log", f"[ASR] ⇧ speech-start p={speech_prob:.2f} thr={threshold:.2f}")
                    # Drive the UI "listening" orb from backend VAD (authoritative).
                    # Only when the AI isn't currently speaking — during AI speech this edge
                    # is the barge-in candidate and gets its own "interrupted" status path.
                    if not self.is_speaking:
                        await self.emit("status", "listening")
                elif was_speech and not is_speech and is_speech_active:
                    await self.emit("log", f"[ASR] ⇩ speech-gap p={speech_prob:.2f} silence_count={silence_count}")
                was_speech = is_speech

                if self.is_speaking and is_speech:
                    # Greeting protection — ignore barge-in for the first 2.5 s of the call
                    # so the assistant's initial reply isn't cut off by mic noise on connect.
                    start_time = getattr(self, 'start_time', time.time())
                    if (time.time() - start_time) < 2.5:
                        continue

                    # Echo gate: give AEC time to stabilise after TTS starts.
                    if (time.time() - self.speaking_start_time) < ECHO_GATE_SEC:
                        continue

                    # Per-utterance hard-lock (longer on greeting, shorter mid-conv).
                    if time.time() < self.barge_in_lock_until:
                        continue

                    barge_in_count += 1
                    if barge_in_count >= BARGE_IN_FRAMES:
                        await self.emit("log", f"[Barge-in] DETECTED p={speech_prob:.2f} frames={barge_in_count}")
                        # 1. Signal every worker to stop producing output for this turn.
                        self.interrupt_event.set()
                        # 2. Drop everything queued so workers don't flush more audio
                        #    the moment we clear the interrupt flag.
                        while not self.tts_queue.empty():
                            try: self.tts_queue.get_nowait()
                            except Exception: break
                        while not self.llm_queue.empty():
                            try: self.llm_queue.get_nowait()
                            except Exception: break
                        # 3. Tell the UI so it stops the scheduled AudioBufferSourceNodes.
                        await self.emit("status", "interrupted")
                        self.is_speaking = False
                        barge_in_count = 0
                        # 4. Yield so tts_worker sees the flag and breaks its chunk loop
                        #    before we clear it for the next turn.
                        await asyncio.sleep(0.05)
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
                        # Proper 2x upsample with anti-aliasing. np.repeat creates
                        # step-function artifacts that Whisper struggles with, worsening
                        # phone-call ASR accuracy.
                        audio_16k = await asyncio.to_thread(
                            signal.resample_poly, audio_np, 2, 1
                        )
                    else:
                        resample_ratio = Fraction(16000, self.input_sampling_rate)
                        audio_16k = await asyncio.to_thread(
                            signal.resample_poly, audio_np, resample_ratio.numerator, resample_ratio.denominator
                        )
                    turn.mark("asr_resample_done")

                    # [AUDIO] DSP cleanup. Empirically the previous "trust Whisper's
                    # log-mel" approach broke down on this user's mic — peaks routinely
                    # came in at 0.1-0.15 (very quiet) and Whisper hallucinated repeats
                    # ('हुआ हुआ हुआ हुआ'). Light, deliberate preprocessing now:
                    #   1) DC offset removal (some mics have a non-zero bias).
                    #   2) Pre-emphasis filter — boosts high frequencies, makes Hindi
                    #      consonants (क/ख/त/थ/च/छ) much more discriminable.
                    #   3) Smart normalization: lift any peak below 0.4 up to ~0.5
                    #      so Whisper's mel-spec gets full dynamic range, but cap the
                    #      gain so we don't blow up the noise floor on near-silence.
                    #   4) 0.2 s silence padding both ends (Whisper trained on 30 s
                    #      chunks; padding stabilises the edges).
                    # --- Advanced ASR DSP (Phase 17) ---
                    # 1. DC Offset & Drift Removal
                    audio_16k = audio_16k - float(np.mean(audio_16k))
                    # 2. Pre-emphasis (Sharpen High Freqs for better Consonants)
                    audio_16k = np.append(audio_16k[0], audio_16k[1:] - 0.97 * audio_16k[:-1]).astype(np.float32)
                    
                    # 3. Gaussian Dither (Loop Breaker)
                    # Adding a microscopic layer of noise breaks Whisper's internal 
                    # infinite repeat loops on hum/silence.
                    noise = np.random.normal(0, 0.0001, audio_16k.shape).astype(np.float32)
                    audio_16k = audio_16k + noise

                    # 4. Smart Gain Normalization
                    p2 = float(np.max(np.abs(audio_16k)))
                    if 0.005 < p2 < 0.5: # Lower floor to catch faint speech
                        gain = min(0.6 / p2, 10.0)
                        audio_16k = audio_16k * gain
                    
                    # 5. Stabilizing Padding
                    padding = np.zeros(int(16000 * 0.2), dtype=np.float32)
                    audio_padded = np.concatenate([padding, audio_16k, padding])

                    # Tuning rationale (accuracy-first now that device/clipping bugs are fixed):
                    #   vad_filter=True                  — Silero VAD in faster-whisper skips silence
                    #                                      and suppresses hallucinations on fragments.
                    #   beam_size=5                      — default-ish, good accuracy/speed tradeoff on GPU.
                    #   temperature=(0.0, 0.2, 0.4, ...)  — fallback schedule: if greedy decode produces
                    #                                      low-confidence output (fails log_prob/compression
                    #                                      thresholds), Whisper retries at higher T. This
                    #                                      is the single biggest accuracy knob — fixes
                    #                                      garbled transcripts on tough audio. Cost: ~30ms
                    #                                      only on the specific hard chunks that need it.
                    #   compression_ratio_threshold=2.4  — reject repetition/loop hallucinations.
                    #   log_prob_threshold=-1.0          — retry when mean token logprob is very low.
                    #   no_speech_threshold=0.6          — drop chunks Whisper thinks are non-speech.
                    #   without_timestamps=True          — we don't use word timings; small speed win.
                    #   language="hi"                    — force Hindi (auto-detect flipped to Korean/Indo
                    #                                      on short clips).
                    turn.mark("asr_decode_start")
                    segments, asr_info = await asyncio.to_thread(
                        self.asr_model.transcribe,
                        audio_padded,
                        beam_size=1,                       # [SPEED] Greedy decoding is 5x faster on GPU 1
                        language="hi",                     # STRICT Hindi mode for Indian context
                        initial_prompt=self.initial_prompt,
                        condition_on_previous_text=False,
                        vad_filter=False,                  # Managed by our authoritative external VAD
                        without_timestamps=True,
                        temperature=0,                     # [SPEED] Single-pass greedy
                        compression_ratio_threshold=2.2,   # Catch loops earlier
                        log_prob_threshold=-1.0,
                        no_speech_threshold=0.5,
                        suppress_blank=True,
                    )
                    text = "".join([s.text for s in segments]).strip()
                    text = _strip_repetition_hallucinations(text)
                    turn.mark("asr_decode_done")
                    detected_lang = getattr(asr_info, "language", None) or "?"
                    lang_prob = float(getattr(asr_info, "language_probability", 0.0) or 0.0)
                    turn.add(
                        asr_text=text[:80],
                        asr_lang=detected_lang,
                        asr_lang_prob=round(lang_prob, 3),
                        asr_beam=5,
                        asr_samples=int(len(audio_padded)),
                        asr_peak=float(p2),
                    )
                    await self.emit(
                        "log",
                        f"[ASR] decode {turn.delta_ms('asr_decode_start','asr_decode_done')}ms "
                        f"lang={detected_lang}({lang_prob:.2f}) beam=5 samples={len(audio_padded)} "
                        f"chars={len(text)} peak={p2:.3f}"
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
                import traceback
                tb = traceback.format_exc()
                print(f"ASR Error: {e}\n{tb}")
                try:
                    await self.emit("log", f"[ASR] ERROR: {type(e).__name__}: {e}")
                except Exception:
                    pass

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
                    "model": self.llm_model, # Replaced constant with dynamic model
                    "messages": [
                        {
                            "role": "system", 
                            "content": (
                                f"IDENTITY: Your name is {self.active_voice_metadata.get('name', 'AI')}. You are {self.active_voice_metadata.get('gender', 'neutral')} speaker.\n"
                                f"PERSONA: Style: {self.active_voice_metadata.get('style', 'conversational')}.\n"
                                f"FAMOUS DIALOGUES/CATCHPHRASES: {self.active_voice_metadata.get('dialogues', '')}\n\n"
                                "STRICT GRAMMAR RULES (GENDER BINDING):\n"
                                f"1. You identify as {self.active_voice_metadata.get('gender', 'male')}. "
                                f"If female, use feminine verb endings (रही हूँ, सकती हूँ). "
                                f"If male, use masculine verb endings (रहा हूँ, सकता हूँ).\n"
                                "2. NEVER mix genders. Be consistent throughout the conversation.\n"
                                "3. LANGUAGE: Speak ONLY in DEVANAGARI HINDI script.\n\n"
                                "STRICT LANGUAGE RULES (PURE HINDI ONLY):\n"
                                "1. You must speak ONLY in DEVANAGARI HINDI script.\n"
                                "2. NEVER reply in English. NEVER translate your thoughts to English.\n"
                                "3. DO NOT use English characters (A-Z). Use Devanagari script ONLY.\n"
                                "4. Tone: Respectful, warm Indian friend — not formal/robotic.\n"
                                "5. CONVERSATIONAL FLOW (very important — sound human):\n"
                                "   • If the user just interrupted you, OPEN your reply with a short natural acknowledgement\n"
                                "     (vary it — don't repeat the same one): 'हाँ बोलिए?', 'जी बताएं?', 'अच्छा,', \n"
                                "     'अरे हाँ,', 'हम्म,', 'ओह अच्छा,', 'जी, मैं सुन रहा हूँ' — THEN answer.\n"
                                "   • If the user's message is short/casual, also start with a small filler like\n"
                                "     'अच्छा', 'हम्म', 'ठीक है', 'हाँ' so it sounds like a real human listening.\n"
                                "   • Keep replies CONCISE (1–3 sentences). Long monologues feel robotic.\n"
                                "   • If the user's question is unclear, ask back: 'क्या मतलब?', 'फिर से बोलिए?',\n"
                                "     'मैं समझा नहीं, थोड़ा और बताइए।'\n\n"
                                "ACOUSTIC TAGS (ENGLISH ONLY):\n"
                                "Use ONLY: [laughter], [sigh], [sniff], [dissatisfaction-hnn]\n"
                                "MISSION: Be extremely helpful, witty, and concise in Hindi."
                            )
                        },
                        *self.history[-20:], # [CONTEXT] Last 20 exchanges (was 10) for better memory
                        {"role": "user", "content": user_text}
                    ],
                    "stream": True,
                    # num_gpu=99 offloads all layers to the GPU (the model is already in
                    # RTX 3060 VRAM). num_ctx=1024 keeps the KV cache ~128 MB so we fit
                    # Within the remaining VRAM alongside Whisper + OmniVoice.
                    "options": {"num_predict": 300, "temperature": 0.5, "num_gpu": 99, "num_ctx": 4096}
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
                    turn.add(llm_model=self.llm_model, llm_num_gpu=99, llm_num_ctx=4096)
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
                                        f"(model={self.llm_model})"
                                    )
                                sentence += chunk
                                full_response += chunk
                                await self.emit("llm_chunk", chunk)

                                # Trigger logic tuned for DEEP BUFFER (Zero-Stutter Update):
                                word_count = len(sentence.split())
                                has_end_punct = any(p in chunk for p in ".?!।\n")
                                if is_first_chunk:
                                    # Increased to 18 words to ensure playback never catches up to synthesis.
                                    trigger = (has_end_punct and word_count >= 8) or word_count >= 15
                                else:
                                    # Subsequent chunks: wait for solid sentence or 35 words.
                                    trigger = (has_end_punct and word_count >= 6) or word_count >= 25

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
                        language="hi", 
                        generation_config=OmniVoiceGenerationConfig(
                            num_step=12,      # [Phase 17] Stabilized quality steps
                        )
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
                    
                    # --- PureVoice Normalization (Phase 17) ---
                    # Prevents the "heavy" sound caused by digital clipping.
                    peak = np.max(np.abs(audio_24k)) if len(audio_24k) > 0 else 0
                    if peak > 1.0 or peak < 0.7:
                        # Softly normalize to 0.95 to keep headroom for the browser's gain.
                        scale = 0.95 / max(peak, 1e-4)
                        audio_24k = audio_24k * scale
                    
                    vol = np.max(np.abs(audio_24k)) if len(audio_24k) > 0 else 0
                    print(f"[TTS] Success: {len(audio_24k)} samples. Peak Vol: {vol:.4f} (Applied PureVoice Norm)", flush=True)
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

                    # --- Speaking lock ---
                    # Per-utterance window where barge-in is suppressed. Protects the very
                    # first audio packet of the session (greeting) from being interrupted
                    # by the echo leaking back through the mic; subsequent turns use a
                    # tighter lock so the user can cut in quickly.
                    self.is_speaking = True
                    if getattr(self, '_is_first_audio_in_session', True):
                        self.barge_in_lock_until = max(
                            self.barge_in_lock_until, time.time() + HARD_LOCK_FIRST_AUDIO_SEC
                        )
                    else:
                        self.barge_in_lock_until = max(
                            self.barge_in_lock_until, time.time() + HARD_LOCK_MID_TURN_SEC
                        )
                    
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
