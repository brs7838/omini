import asyncio
import collections
import os
import httpx
import json
import re
import numpy as np
import torch
import time
from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from scipy import signal

try:
    from web_backend.campaigns import load_campaign, build_system_prompt
    from web_backend.stt_providers import (
        STTChoice, resolve_stt, build_stt, LocalWhisperSTT, SarvamSTT,
    )
except ImportError:
    from campaigns import load_campaign, build_system_prompt
    from stt_providers import (
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
SILENCE_CHUNKS = 3              # Increased to 3 (750ms) for conversational robustness
MAX_BUFFER_CHUNKS = 250
BARGE_IN_FRAMES = 24            # Increased to 24 (~600ms) to ensure deliberate interruption
TTS_CHUNK_SAMPLES = 6000        # 250ms at 24kHz — stream small chunks for low-latency playback

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
    
    # 4. Final Cleanup: Remove any stray brackets that didn't form a valid tag (before restoring placeholders)
    # This prevents the AI from saying "k" or click-sounds for single brackets
    text = re.sub(r'[\[\]]', '', text)

    # Restore original tags
    for placeholder, original in tag_map.items():
        text = text.replace(placeholder, original)
    
    return text.strip()

class _ModelStore:
    """Singleton: loads heavy GPU models ONCE, reused across all WebSocket
    connections. Whisper is now lazy — only loaded when the active STT
    provider is 'whisper'. When STT is 'sarvam' (default), the GPU holds
    only OmniVoice, which keeps the CUDA allocator clean and stops cuDNN
    from rechoosing convolution kernels between TTS turns. That allocator
    churn was the root of the inter-turn pitch drift."""
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loaded = False
            cls._instance.asr_model = None
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
        print(f"[Engine] Loading Whisper ({model_name}) on cuda:0 ...", flush=True)
        self.asr_model = WhisperModel(model_name, device="cuda", device_index=0, compute_type="int8_float16")
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
        self._whisper_name = None
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
        self.initial_prompt = "Natural Hindi/Hinglish: नमस्ते, क्या हाल है? मैं ठीक हूँ। Hello, how are you?"
        # self.llm_model = LLM_MODEL # REMOVED: Now set from __init__
        self.is_running = True
        self.is_speaking = False
        self.speaking_start_time = 0    # Track when AI starts speaking for barge-in grace period
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

                    padding = np.zeros(int(16000 * 0.2), dtype=np.float32)
                    audio_padded = np.concatenate([padding, audio_16k, padding])

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
                        try:
                            text = await stt.transcribe(audio_padded)
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
                        await self.llm_queue.put(text)
                        
                        # Barge-in Interruption
                        self.interrupt_event.set()
                        self.is_speaking = False # Reset state immediately so next turn is sensitive
                        # Drain remaining TTS to stop immediately
                        while not self.tts_queue.empty(): self.tts_queue.get_nowait()
                    buffer = []
                    silence_count = 0
                    is_speech_active = False
            except Exception as e:
                print(f"ASR Error: {e}")

    async def llm_worker(self):
        print("[LLM] Worker started.")
        async with httpx.AsyncClient(timeout=30.0) as client:
            while self.is_running:
                user_text = await self.llm_queue.get()
                # Fresh turn: clear any stale interrupt flag set by ASR's
                # barge-in signal, otherwise the stream loop below exits on
                # the very first chunk and the user gets silence.
                self.interrupt_event.clear()
                # Per-turn provider refresh: picks up a /provider/switch that
                # happened mid-session without forcing a WebSocket reconnect.
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
                # Per-turn campaign refresh so a /campaign PUT takes effect on
                # the next user turn without reopening the WebSocket.
                campaign = None
                if self.get_campaign is not None:
                    try:
                        campaign = self.get_campaign()
                    except Exception as _e:
                        print(f"[LLM] get_campaign error: {_e}", flush=True)
                if campaign is None:
                    campaign = load_campaign()
                system_prompt = build_system_prompt(
                    campaign,
                    voice_name=self.active_voice_metadata.get("name", "AI"),
                    voice_gender=self.active_voice_metadata.get("gender", "male"),
                )
                # Diag: prove the worker is alive and seeing new turns.
                print(f"[LLM] dequeued turn: {user_text!r} (url={self.llm_url}, model={self.llm_model}) campaign={campaign.get('label') if campaign else None!r}", flush=True)
                payload = {
                    "model": self.llm_model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        *self.history[-10:],  # last 10 exchanges for context
                        {"role": "user", "content": user_text},
                    ],
                    "stream": True,
                }
                # Per-provider request shape (Ollama's `options` dict vs
                # Sarvam's top-level OpenAI fields). Merged last so an override
                # wins over the legacy defaults.
                legacy_options = {"num_predict": 128, "temperature": 0.5}
                if self.llm_extras:
                    payload.update(self.llm_extras)
                else:
                    payload["options"] = legacy_options
                sentence = ""
                full_response = ""
                is_first_chunk = True
                # Stream-safe <think>...</think> filter state. Sarvam-M/30B can
                # still emit reasoning wrapped in <think> tags in some turns
                # even with reasoning_effort=null — this buffer makes sure no
                # such content ever reaches TTS. Keyed per turn.
                think_state = {"in_think": False, "tail": ""}
                print(f"[LLM] pre-emit status=thinking", flush=True)
                await self.emit("status", "thinking")
                print(f"[LLM] post-emit status=thinking; opening stream to {self.llm_url}", flush=True)
                try:
                    async with client.stream("POST", self.llm_url, json=payload, headers=self.llm_headers or None) as response:
                        async for line in response.aiter_lines():
                            if self.interrupt_event.is_set():
                                await self.emit("status", "interrupted")
                                break
                            if line.startswith("data: "):
                                if "[DONE]" in line: break
                                try:
                                    body = json.loads(line[6:])
                                except json.JSONDecodeError:
                                    continue
                                choices = body.get("choices") or []
                                if not choices:
                                    # Sarvam occasionally sends keep-alive frames
                                    # with empty choices; skip without crashing.
                                    continue
                                delta = choices[0].get("delta") or {}
                                raw_chunk = delta.get("content") or ""
                                # Also skip any `reasoning_content` delta some
                                # reasoning-capable backends emit on the side —
                                # we never want it spoken.
                                chunk = _strip_think_streaming(raw_chunk, think_state)
                                if not chunk:
                                    continue
                                sentence += chunk
                                full_response += chunk
                                await self.emit("llm_chunk", chunk)
                                # NOTE: per-sentence TTS flush removed. The
                                # lab server synthesizes the whole phrase in a
                                # single generate() call and has no pitch
                                # drift; the engine used to split on .?!।
                                # which made each sentence an independent
                                # generate() pass, and OmniVoice's prosody
                                # planner picks a fresh pitch center every
                                # call. Seeding didn't help because seeding
                                # only pins diffusion noise, not the planner.
                                # We now buffer the full response and synth
                                # once at end-of-stream (see flush below).
                    # End-of-stream: flush the think-filter's hold buffer. It
                    # carries up to 7 chars of lookahead we use to catch tags
                    # split across chunks; at stream end those chars are safe
                    # to emit (we're guaranteed no more tag can begin there).
                    if not think_state["in_think"] and think_state["tail"]:
                        trailing = think_state["tail"]
                        think_state["tail"] = ""
                        sentence += trailing
                        full_response += trailing
                        await self.emit("llm_chunk", trailing)
                    # Single-shot synth: send the entire response as one TTS
                    # job. Matches tts_lab/server.py's behaviour and keeps
                    # pitch consistent across the whole turn. Trade-off: first
                    # audio lands ~200-400ms later than the old chunked path,
                    # but prompts are capped to 2 sentences (campaigns.json
                    # max_sentences_per_turn) so a single call stays well
                    # under OmniVoice's 15s audio_chunk_duration budget.
                    if not self.interrupt_event.is_set():
                        whole = clean_text(full_response)
                        if whole:
                            await self.tts_queue.put(whole)
                            is_first_chunk = False
                    await self.tts_queue.put("__END_RESPONSE__")
                    
                    display_text = full_response
                    def ui_whitelist(match):
                        content = match.group(1).lower().strip()
                        mapping = {"हँसी": "laughter", "हंसी": "laughter", "haha": "laughter", "आह": "sigh", "स्निफ़": "sniff", "हम्म": "dissatisfaction-hnn"}
                        val = mapping.get(content, content)
                        return f"[{val}]" if val in ["laughter", "sigh", "sniff", "dissatisfaction-hnn"] else ""
                    display_text = re.sub(r'\[(.*?)\]', ui_whitelist, display_text)
                    await self.emit("transcript", {"role": "ai", "text": display_text})
                    
                    # Store in history for better intelligence in next turn
                    self.history.append({"role": "user", "content": user_text})
                    self.history.append({"role": "assistant", "content": full_response})
                    if len(self.history) > 20: # Keep memory lean
                        self.history = self.history[-20:]
                except Exception as e:
                    import traceback
                    print(f"LLM Error: {type(e).__name__}: {e}")
                    traceback.print_exc()

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
                print(f"[TTS] Synthesizing (text_sha={text_sha}): {cleaned_text}")
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
                                guidance_scale=2.0,
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
