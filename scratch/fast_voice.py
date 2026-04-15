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
from audio_utils import to_8k

# --- Configuration ---
RATE = 16000
CHUNK = 1024
SILENCE_THRESHOLD = 1200    # Increased to prevent self-triggering
SILENCE_DURATION = 0.7     # Lowered for Gemini-style speed
MIN_SPEECH_DURATION = 0.4
MAX_RECORD_DURATION = 10.0
MIC_INDEX = 1

LLM_URL = "http://127.0.0.1:11434/v1/chat/completions"
LLM_MODEL = "gemma3:4b"

# Voice cloning setup
REF_AUDIO_PATH = os.path.join(os.path.dirname(__file__), "..", "assets", "voices", "ravi_sir_8s.wav")

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
        print("Loading Whisper (Turbo) on GPU 1...")
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
                preprocess_prompt=False
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
        silent_chunks = 0
        speech_chunks = 0
        has_spoken = False
        
        while self.is_running:
            # If the user starts talking while we are playing, STOP playback.
            # (Crude interruption support)
            try:
                data = in_stream.read(CHUNK, exception_on_overflow=False)
                audio_data = np.frombuffer(data, dtype=np.int16)
                energy = np.abs(audio_data).mean()
                
                if energy > SILENCE_THRESHOLD:
                    if self.is_speaking:
                        print("\n[Interrupted!]")
                        self.abort_playback = True
                        self.clear_queues()
                    
                    silent_chunks = 0
                    speech_chunks += 1
                    has_spoken = True
                    frames.append(data)
                    self.live_audio_buffer.append(data)
                else:
                    if has_spoken:
                        silent_chunks += 1
                        frames.append(data)
                
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
                
                t0 = time.time()
                # Final transcription for the LLM
                segments, _ = self.asr_model.transcribe(audio_np, language="hi")
                text = " ".join([seg.text for seg in segments]).strip()
                
                if text:
                    print(f"\rYou: {text} (Final)                ")
                    self.text_queue.put(text)
                else:
                    print("\r[System] (No speech detected)          ")
            except queue.Empty:
                continue

    def live_asr_worker(self):
        """Thread: Provides partial transcription while the user is still speaking."""
        while self.is_running:
            if len(self.live_audio_buffer) > 10 and not self.is_processing:
                # Take current buffer snapshot
                snapshot = b"".join(list(self.live_audio_buffer))
                audio_np = np.frombuffer(snapshot, dtype=np.int16).astype(np.float32) / 32768.0
                
                try:
                    # Run a very fast partial transcription
                    segments, _ = self.asr_model.transcribe(
                        audio_np, 
                        language="hi",
                        beam_size=1  # Lowest beam for max speed during live preview
                    )
                    partial_text = " ".join([seg.text for seg in segments]).strip()
                    if partial_text:
                        print(f"\rYou: {partial_text}...", end="", flush=True)
                except:
                    pass
            time.sleep(0.4) # Update every 400ms

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
                generation_config=OmniVoiceGenerationConfig(num_step=12)
            )
            audio_24k = audio_list[0]
            
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
