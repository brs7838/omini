import os
import torch
import numpy as np
import pyaudio
import wave
import time
import requests
import json
import threading
import queue
import re
import sys
import io
from faster_whisper import WhisperModel
from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice.utils.text import chunk_text_punctuation
from audio_utils import to_8k  # Local utility

# --- Configuration ---
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000 
SILENCE_THRESHOLD = 800  # More sensitive
SILENCE_DURATION = 0.8   # Snappier turn-taking
MAX_BUFFER_SECONDS = 15  # Prevent 30s crash

# Ollama Server Config (OpenAI Compatible)
LLM_URL = "http://127.0.0.1:11434/v1/chat/completions"
LLM_MODEL = "gemma3:4b" 

# Voice clone prompt path (included in repo)
REF_AUDIO_PATH = os.path.join(os.path.dirname(__file__), "..", "assets", "voices", "ravi_sir.mp3")

def clean_text(text):
    """Removes metadata, markdown, and artifacts from AI response."""
    text = re.sub(r'#+\s*', '', text)
    text = text.replace('**', '').replace('*', '')
    text = re.sub(r'[\U00010000-\U0010ffff]', '', text)
    text = re.sub(r'[\(\[].*?[\)\]]', '', text)
    text = re.sub(r'^(उत्तर|सांख्यक|संख्या|अनुवाद|सवाल|AI|USER|आई\.आई\.)[:ः]\s*', '', text, flags=re.IGNORECASE)
    return text.strip()

class VoiceAssistant:
    def __init__(self):
        print("Loading OmniVoice TTS on GPU 0...")
        self.model = OmniVoice.from_pretrained(
            "k2-fsa/OmniVoice",
            device_map="cuda:0", # Primary GPU
            dtype=torch.float16,
            load_asr=False # We will load ASR manually on GPU 1
        )
        
        print("Loading Whisper ASR (Faster) on GPU 1 (GTX 1650)...")
        # Using faster-whisper for near-instant transcription
        self.asr_model = WhisperModel(
            "large-v3-turbo", 
            device="cuda", 
            device_index=1, 
            compute_type="int8_float16" # Fast and memory efficient for GTX 1650
        )
        
        print(f"Creating voice clone prompt from: {REF_AUDIO_PATH}")
        try:
            self.voice_prompt = self.model.create_voice_clone_prompt(
                ref_audio=REF_AUDIO_PATH,
                preprocess_prompt=True
            )
            print("Voice prompt created successfully!")
        except Exception as e:
            print(f"Error creating voice prompt: {e}")
            self.voice_prompt = None

        self.p = pyaudio.PyAudio()
        self.is_running = True
        self.is_speaking = False
        self.is_processing = False 
        self.playback_queue = queue.Queue()
        
        # Audio output stream state
        self.out_stream = None
        self.init_output_stream()

        # Start background playback thread
        self.playback_thread = threading.Thread(target=self._playback_worker, daemon=True)
        self.playback_thread.start()

    def init_output_stream(self):
        """Initializes the output stream at 8000Hz (Telephony standard)."""
        try:
            if self.out_stream:
                try: self.out_stream.close()
                except: pass
            
            print("Audio Output: Enabled (8000 Hz, Mono)")
            self.out_stream = self.p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=8000, # Telephony Standard
                output=True,
                frames_per_buffer=1024
            )
            return True
        except Exception as e:
            print(f"Failed to init audio output: {e}")
            return False

    def _playback_worker(self):
        """Background thread to play audio chunks from the queue."""
        while self.is_running:
            try:
                pcm_data = self.playback_queue.get(timeout=0.1)
                self.is_speaking = True
                
                try:
                    if self.out_stream:
                        self.out_stream.write(pcm_data)
                except OSError as e:
                    print(f"\n[Audio Error] {e}. Recovering...")
                    self.init_output_stream()
                    if self.out_stream:
                        self.out_stream.write(pcm_data)
                
                if self.playback_queue.empty():
                    time.sleep(0.4) 
                    self.is_speaking = False
                    
            except queue.Empty:
                self.is_speaking = False
                continue
            except Exception as e:
                print(f"Playback error: {e}")
                self.is_speaking = False

    def listen(self):
        in_stream = self.p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
            input_device_index=None
        )
        
        print("\nReady! Speak in Hindi... (Telephony 8k Playback active)")
        
        frames = []
        silent_chunks = 0
        has_spoken = False
        
        while self.is_running:
            if self.is_speaking or self.is_processing:
                try:
                    available = in_stream.get_read_available()
                    if available > 0:
                        in_stream.read(available, exception_on_overflow=False)
                except:
                    pass
                time.sleep(0.1)
                continue

            try:
                data = in_stream.read(CHUNK, exception_on_overflow=False)
                frames.append(data)
                
                audio_data = np.frombuffer(data, dtype=np.int16)
                energy = np.abs(audio_data).mean()
                
                if energy > SILENCE_THRESHOLD:
                    if not has_spoken:
                        print("Listening...", end="\r")
                    silent_chunks = 0
                    has_spoken = True
                else:
                    if has_spoken:
                        silent_chunks += 1
                
                # Prevent indefinitely growing buffer (30s error fix)
                if len(frames) > (MAX_BUFFER_SECONDS * RATE / CHUNK):
                    if not has_spoken:
                        frames = frames[-(CHUNK * 2):] # Keep only last 2 chunks
                    else:
                        # Force process if it's too long
                        print("Auto-processing long segment...")
                        silent_chunks = 100 
                
                if has_spoken and silent_chunks > (SILENCE_DURATION * RATE / CHUNK):
                    self.is_processing = True 
                    print("Processing...      ")
                    full_audio = b"".join(frames)
                    threading.Thread(target=self.process_interaction, args=(full_audio,), daemon=True).start()
                    
                    # Reset
                    frames = []
                    silent_chunks = 0
                    has_spoken = False
            except Exception as e:
                print(f"\nMic error: {e}")
                time.sleep(0.1)

    def process_interaction(self, audio_bytes):
        # Convert audio bytes to float32 numpy array for faster-whisper
        audio_fp32 = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        
        try:
            # Transcribe with faster-whisper
            segments, info = self.asr_model.transcribe(
                audio_fp32, 
                beam_size=1, # Fast!
                language=None, # Auto-detect for Hinglish
                vad_filter=True
            )
            text = "".join([s.text for s in segments]).strip()
            
            if text:
                print(f"Whisper heard: '{text}' (lang: {info.language})")
        except Exception as e:
            print(f"ASR Error: {e}")
            self.is_processing = False
            return
        
        # Filter out junk and noise artifacts like "???", "...", etc.
        if not text or len(text) < 2 or re.match(r'^[\?\.\s!-]+$', text):
            self.is_processing = False
            return
        
        print(f"You: {text}")
        
        try:
            payload = {
                "model": LLM_MODEL,
                "messages": [
                    {"role": "system", "content": "आप एक सहायक भारतीय मित्र हैं जिसका नाम 'OmniVoice AI' है। केवल और केवल हिंदी (Hindi/Hinglish) में ही बात करें। संक्षिप्त और मधुर उत्तर दें।"},
                    {"role": "user", "content": text}
                ],
                "stream": True
            }
            
            sentence = ""
            # llama.cpp /v1/chat/completions returns 'data: {...}'
            with requests.post(LLM_URL, json=payload, stream=True) as r:
                for line in r.iter_lines():
                    if line:
                        line_str = line.decode('utf-8')
                        if line_str.startswith("data: "):
                            if "[DONE]" in line_str:
                                break
                            body = json.loads(line_str[6:])
                            chunk = body.get("choices", [{}])[0].get("delta", {}).get("content", "")
                            sentence += chunk
                            
                            # Stream by sentence for low latency
                            if any(p in chunk for p in ".!?।\n"):
                                cleaned = clean_text(sentence)
                                if cleaned and len(cleaned) > 1:
                                    self.enqueue_tts(cleaned)
                                sentence = ""
                
        except Exception as e:
            print(f"LLM Error: {e}")
        finally:
            self.is_processing = False
            print("\nReady for your next message...")

    def enqueue_tts(self, text):
        """Generates audio and resamples to 8kHz for telephony simulation."""
        print(f"AI: {text}", end="\r")
        kw = {
            "text": text,
            "voice_clone_prompt": self.voice_prompt,
            "generation_config": OmniVoiceGenerationConfig(num_step=16)
        }
        
        try:
            audio_list = self.model.generate(**kw)
            audio_24k = audio_list[0]
            
            # Resample to 8k
            audio_8k = to_8k(audio_24k, orig_sr=24000)
            
            pcm_data = (audio_8k * 32767).astype(np.int16).tobytes()
            self.playback_queue.put(pcm_data)
        except Exception as e:
            print(f"\nTTS Error: {e}")

    def run(self):
        try:
            self.listen()
        except KeyboardInterrupt:
            print("\nShutting down Assistant...")
        finally:
            self.is_running = False
            if self.out_stream:
                self.out_stream.stop_stream()
                self.out_stream.close()
            self.p.terminate()

if __name__ == "__main__":
    assistant = VoiceAssistant()
    assistant.run()
