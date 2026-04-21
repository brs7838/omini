import os
import torch
import numpy as np
import pyaudio
import time
import requests
import json
import threading
import queue
import re
from faster_whisper import WhisperModel
from omnivoice import OmniVoice, OmniVoiceGenerationConfig
# Resolve imports from web_backend
import sys
import os
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from web_backend.audio_utils import to_8k



# --- Configuration ---
RATE = 16000
CHUNK = 1024
SILENCE_THRESHOLD = 700     # Lowered: catches softer speech onset
SILENCE_DURATION = 0.8      # Slightly longer to avoid cutting mid-sentence pauses
MIN_SPEECH_DURATION = 0.3
MAX_RECORD_DURATION = 15.0
MIC_INDEX = 1
PRE_ROLL_CHUNKS = 8         # ~0.5s of pre-speech audio kept (so first syllable isn't lost)

# Hindi context prompt — biases Whisper toward natural Hindi vocabulary.
# A long, varied Hindi prompt is one of the strongest accuracy levers for
# faster-whisper on Indian languages. Keep it diverse (greetings, verbs, nouns,
# question words) so the language model "warms up" to Hindi script + grammar.
ASR_INITIAL_PROMPT = (
    "नमस्ते, आप कैसे हैं? मैं बिल्कुल ठीक हूँ, बहुत-बहुत धन्यवाद। "
    "कृपया मुझे बताइए, आज का मौसम कैसा है? "
    "हाँ, नहीं, ठीक है, अच्छा, सही, गलत, माफ़ कीजिए, शुक्रिया। "
    "क्या आप मेरी मदद कर सकते हैं? मुझे यह समझ नहीं आया, फिर से बोलिए। "
    "मेरा नाम, आपका नाम, घर, परिवार, दोस्त, स्कूल, काम, समय, पैसा, "
    "खाना, पानी, चाय, किताब, फ़ोन, गाड़ी, शहर, गाँव, देश, भारत। "
    "एक, दो, तीन, चार, पाँच, छह, सात, आठ, नौ, दस। "
    "आज, कल, अभी, बाद में, सुबह, शाम, रात, दिन।"
)

LLM_URL = "http://127.0.0.1:11434/v1/chat/completions"
LLM_MODEL = "gemma3:4b"

# Voice cloning setup
REF_AUDIO_PATH = os.path.join(os.path.dirname(__file__), "..", "assets", "voices", "ravi_sir_8s.wav")

def preprocess_for_asr(audio_np, sr=16000):
    """Light DSP cleanup that consistently improves Whisper accuracy on noisy mics:
       1) DC-offset removal (removes mic bias),
       2) Pre-emphasis (boosts high freqs → sharper Hindi consonants like क/ख/त/थ),
       3) Peak normalisation (gives quiet speech full dynamic range),
       4) 0.4s silence padding at both ends (Whisper is trained on padded chunks).
    """
    if audio_np.size == 0:
        return audio_np
    # 1) DC offset
    audio_np = audio_np - float(np.mean(audio_np))
    # 2) Pre-emphasis
    audio_np = np.append(audio_np[0], audio_np[1:] - 0.97 * audio_np[:-1]).astype(np.float32)
    # 3) Peak normalise
    peak = float(np.max(np.abs(audio_np)))
    if peak > 1e-4:
        audio_np = audio_np * (0.97 / peak)
    # 4) Silence padding
    pad = np.zeros(int(0.4 * sr), dtype=np.float32)
    return np.concatenate([pad, audio_np.astype(np.float32), pad])


def clean_text(text):
    """Removes metadata, markdown, and objects in brackets from AI response."""
    text = re.sub(r'#+\s*', '', text)
    text = text.replace('**', '').replace('*', '')
    text = re.sub(r'[\U00010000-\U0010ffff]', '', text)
    text = re.sub(r'[\(\[].*?[\)\]]', '', text)
    text = re.sub(r'^(उत्तर|AI|USER|आई\.आई\.)[:ः]\s*', '', text, flags=re.IGNORECASE)
    return text.strip()

class FastVoiceAssistant:
    def __init__(self):
        print("--- initializing Gemini-Style Assistant ---")
        
        # GPU Setup
        # Whisper: GPU 1 (GTX 1650), OmniVoice: GPU 0 (RTX 3060)
        print("Loading Whisper (large-v3) on GPU 1 for accurate Hindi ASR...")
        # large-v3 is significantly more accurate for Hindi than turbo.
        # If VRAM is tight, fall back to "turbo" — but turbo drops Hindi accuracy noticeably.
        try:
            self.asr_model = WhisperModel(
                "large-v3",
                device="cuda",
                device_index=1,
                compute_type="float16"
            )
            print("Loaded large-v3 (best Hindi accuracy).")
        except Exception as e:
            print(f"large-v3 load failed ({e}); falling back to turbo.")
            self.asr_model = WhisperModel(
                "turbo",
                device="cuda",
                device_index=1,
                compute_type="float16"
            )
        
        print("Loading OmniVoice TTS on GPU 0...")
        self.tts_model = OmniVoice.from_pretrained(
            "k2-fsa/OmniVoice",
            device_map="cuda:0",
            dtype=torch.float16,
            load_asr=False,
            local_files_only=True
        )
        
        print(f"Creating voice clone prompt (Ravi Sir)...")
        try:
            self.voice_prompt = self.tts_model.create_voice_clone_prompt(
                ref_audio=REF_AUDIO_PATH,
                ref_text="नमस्ते, मैं OmniVoice हूँ।",
                preprocess_prompt=True # [Phase 17] Filters noise for clearer cloning
            )
            print("Voice prompt created successfully!")
        except Exception as e:
            print(f"Error creating voice prompt: {e}")
            self.voice_prompt = None
            
        self.p = pyaudio.PyAudio()
        self.is_running = True
        
        # Queues for the pipeline
        self.audio_queue = queue.Queue()    # From Recorder to ASR
        self.text_queue = queue.Queue()     # From ASR to LLM
        self.playback_queue = queue.Queue() # From TTS to Speaker
        
        self.is_speaking = False
        self.is_processing = False
        self.abort_playback = False
        self.live_audio_buffer = []  # For real-time text feedback
        
        # Audio stream
        self.out_stream = self.p.open(
            format=pyaudio.paInt16, channels=1, rate=8000, output=True
        )

    def listen_loop(self):
        """Thread: Captures audio and puts into queue on voice trigger."""
        in_stream = self.p.open(
            format=pyaudio.paInt16, channels=1, rate=RATE, input=True,
            frames_per_buffer=CHUNK, input_device_index=MIC_INDEX
        )
        
        print("\n[System] Ready. Talk to me!")
        
        frames = []
        pre_roll = []   # rolling buffer of recent silent chunks (kept so we don't clip the first phoneme)
        silent_chunks = 0
        speech_chunks = 0
        has_spoken = False

        while self.is_running:
            try:
                data = in_stream.read(CHUNK, exception_on_overflow=False)
                audio_data = np.frombuffer(data, dtype=np.int16)
                # RMS is more robust than mean(abs) — better matches actual loudness
                energy = float(np.sqrt(np.mean(audio_data.astype(np.float32) ** 2)))

                if energy > SILENCE_THRESHOLD:
                    if self.is_speaking:
                        print("\n[Interrupted!]")
                        self.abort_playback = True
                        self.clear_queues()

                    if not has_spoken:
                        # First voiced chunk — prepend pre-roll so the onset isn't lost
                        frames.extend(pre_roll)
                        self.live_audio_buffer.extend(pre_roll)
                        pre_roll = []

                    silent_chunks = 0
                    speech_chunks += 1
                    has_spoken = True
                    frames.append(data)
                    self.live_audio_buffer.append(data)
                else:
                    if has_spoken:
                        silent_chunks += 1
                        frames.append(data)
                    else:
                        # Maintain rolling pre-roll buffer of recent silent chunks
                        pre_roll.append(data)
                        if len(pre_roll) > PRE_ROLL_CHUNKS:
                            pre_roll.pop(0)
                
                # Check for processing trigger
                if has_spoken and silent_chunks > (SILENCE_DURATION * RATE / CHUNK):
                    speech_seconds = speech_chunks * CHUNK / RATE
                    if speech_seconds >= MIN_SPEECH_DURATION:
                        print(f"\n[You] ... (heard {speech_seconds:.1f}s)")
                        self.audio_queue.put(b"".join(frames))
                    
                    frames = []
                    self.live_audio_buffer = []
                    silent_chunks = 0
                    speech_chunks = 0
                    has_spoken = False
                    
            except Exception as e:
                print(f"Mic error: {e}")

    def asr_worker(self):
        """Thread: Transcribes audio as soon as it's ready."""
        while self.is_running:
            try:
                audio_bytes = self.audio_queue.get(timeout=0.1)
                audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

                # DSP preprocessing — biggest accuracy win on noisy / low-gain mics
                audio_np = preprocess_for_asr(audio_np, sr=RATE)

                t0 = time.time()
                # Accuracy-tuned transcription.
                # NOTE: Silero `vad_filter` is intentionally OFF — for Hindi it sometimes
                # trims real speech (especially soft vowels), and we already gate audio
                # on the mic side. Turning it off improved word-recovery noticeably.
                segments, info = self.asr_model.transcribe(
                    audio_np,
                    language="hi",
                    task="transcribe",
                    beam_size=8,                      # bigger beam → better Hindi hypotheses
                    best_of=8,
                    patience=1.5,
                    temperature=[0.0, 0.2, 0.4, 0.6], # fallback temps reduce hallucination loops
                    condition_on_previous_text=False, # avoids context bleed across utterances
                    initial_prompt=ASR_INITIAL_PROMPT,
                    vad_filter=False,
                    no_speech_threshold=0.45,
                    compression_ratio_threshold=2.4,
                    log_prob_threshold=-1.0,
                    word_timestamps=False,
                    suppress_blank=True,
                )
                text = " ".join(seg.text for seg in segments).strip()
                # Drop common Whisper Hindi-mode hallucinations on near-silence
                if text in {"शुक्रिया।", "धन्यवाद।", "नमस्ते।", "सुनो।", "हाँ।"} and len(audio_bytes) < RATE * 2 * 1:
                    text = ""
                dt = time.time() - t0

                if text:
                    print(f"\rYou: {text}   (ASR {dt:.2f}s)                ")
                    self.text_queue.put(text)
                else:
                    print("\r[System] (No speech detected)          ")
            except queue.Empty:
                continue
            except Exception as e:
                print(f"\nASR error: {e}")

    def live_asr_worker(self):
        """Disabled: live partial transcription competes with the final ASR pass on the
        same GPU and noticeably degrades final accuracy. Re-enable only on a second
        Whisper instance / GPU if needed."""
        return

    def llm_tts_worker(self):
        """Thread: Calls LLM and streams sentences to TTS."""
        while self.is_running:
            try:
                user_text = self.text_queue.get(timeout=0.1)
                self.is_processing = True
                
                payload = {
                    "model": LLM_MODEL,
                    "messages": [
                        {"role": "system", "content": "आप एक सहायक भारतीय मित्र हैं जिनका नाम 'OmniVoice AI' है। केवल और केवल शुद्ध हिंदी (Hindi) में ही बात करें। संक्षिप्त और मधुर उत्तर दें। अंग्रेजी या रोमन हिंदी का प्रयोग न करें।"},
                        {"role": "user", "content": user_text}
                    ],
                    "stream": True
                }
                
                sentence = ""
                print("AI: ", end="", flush=True)
                with requests.post(LLM_URL, json=payload, stream=True) as r:
                    for line in r.iter_lines():
                        if not self.is_running: break
                        if line:
                            line_str = line.decode('utf-8')
                            if line_str.startswith("data: "):
                                if "[DONE]" in line_str: break
                                try:
                                    body = json.loads(line_str[6:])
                                    chunk = body.get("choices", [{}])[0].get("delta", {}).get("content", "")
                                    sentence += chunk
                                    print(chunk, end="", flush=True)
                                    
                                    if any(p in chunk for p in ".!?।\n"):
                                        cleaned = clean_text(sentence)
                                        if cleaned and len(cleaned) > 1:
                                            self.generate_and_stream_tts(cleaned)
                                        sentence = ""
                                except Exception:
                                    continue
                
                if sentence.strip():
                    cleaned = clean_text(sentence)
                    if cleaned and len(cleaned) > 1:
                        self.generate_and_stream_tts(cleaned)
                
                print()
                self.is_processing = False
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"LLM Error: {e}")
                self.is_processing = False

    def generate_and_stream_tts(self, text):
        """Generates audio for a single sentence and adds to playback queue."""
        if not text or len(text) < 2: return
        
        try:
            # 12 steps = Premium Human Quality (fixes "metallic" Hindi sound)
            audio_list = self.tts_model.generate(
                text=text, 
                language="hi", 
                voice_clone_prompt=self.voice_prompt,
                generation_config=OmniVoiceGenerationConfig(
                    num_step=12
                )
            )
            audio_24k = audio_list[0]
            
            # --- PureVoice Normalization (Phase 17) ---
            # Keeps the audio crisp and prevents "heavy" distortion
            peak = np.max(np.abs(audio_24k)) if len(audio_24k) > 0 else 0
            if peak > 1.0 or peak < 0.7:
                scale = 0.95 / max(peak, 1e-4)
                audio_24k = audio_24k * scale

            # Resample on GPU for near-zero delay
            audio_8k = to_8k(audio_24k, orig_sr=24000, device="cuda:0")
            pcm_data = (audio_8k * 32767).astype(np.int16).tobytes()
            self.playback_queue.put(pcm_data)
        except Exception as e:
            print(f"\nTTS Error: {e}")

    def playback_worker(self):
        """Thread: Plays audio chunks smoothly with instant interruption support."""
        while self.is_running:
            try:
                pcm_data = self.playback_queue.get(timeout=0.1)
                self.is_speaking = True
                self.abort_playback = False
                
                # Write in small chunks (1024 bytes) to allow instant interruption
                chunk_len = 2048 # 1024 samples * 2 bytes
                for i in range(0, len(pcm_data), chunk_len):
                    if self.abort_playback or not self.is_running:
                        break
                    self.out_stream.write(pcm_data[i:i+chunk_len])
                
                if self.playback_queue.empty() and not self.abort_playback:
                    self.is_speaking = False
            except queue.Empty:
                self.is_speaking = False
                continue

    def clear_queues(self):
        """Utility to wipe all pending work on interruption."""
        while not self.audio_queue.empty(): self.audio_queue.get()
        while not self.text_queue.empty(): self.text_queue.get()
        while not self.playback_queue.empty(): self.playback_queue.get()
        self.is_speaking = False
        self.is_processing = False

    def run(self):
        # Start all worker threads
        threads = [
            threading.Thread(target=self.listen_loop, daemon=True),
            threading.Thread(target=self.asr_worker, daemon=True),
            threading.Thread(target=self.live_asr_worker, daemon=True),
            threading.Thread(target=self.llm_tts_worker, daemon=True),
            threading.Thread(target=self.playback_worker, daemon=True),
        ]
        for t in threads: t.start()
        
        try:
            while True: time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down...")
            self.is_running = False
            self.p.terminate()

if __name__ == "__main__":
    assistant = FastVoiceAssistant()
    assistant.run()
