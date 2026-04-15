import os
import torch
import numpy as np
import pyaudio
import asyncio
import httpx
import json
import time
import queue
import re
import threading
from rich.console import Console
from rich.theme import Theme
from faster_whisper import WhisperModel
from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from audio_utils import to_8k

# --- Configuration ---
RATE = 16000
CHUNK = 1024
CHANNELS = 1
FORMAT = pyaudio.paInt16

# VAD / Barge-in Config
ENERGY_THRESHOLD = 1000 # Lowered for better sensitivity
SILENCE_CHUNKS = 6 # ~0.4s window
INTERRUPT_COOLDOWN = 1.2 
MAX_BUFFER_CHUNKS = 250 # ~15 seconds cap to prevent ASR overflow

# LLM Config
LLM_URL = "http://127.0.0.1:11434/v1/chat/completions"
LLM_MODEL = "gemma3:4b"

# Voice Config
REF_AUDIO_PATH = os.path.join(os.path.dirname(__file__), "..", "assets", "voices", "ravi_sir.mp3")

def clean_text(text):
    """Handles markdown, metadata, and STOPS translations while PROTECTING emotional tags."""
    # 1. Protect special OmniVoice tags by masking them
    tags = ["[laughter]", "[sigh]", "[sniff]", "[dissatisfaction-hnn]"]
    for i, tag in enumerate(tags):
        text = text.replace(tag, f"__TAG{i}__")
    
    # 2. Strip brackets, parentheses, and other garbage
    text = re.sub(r'[\(\[].*?[\)\]]', '', text)
    text = re.sub(r'#+\s*', '', text)
    text = text.replace('**', '').replace('*', '')
    text = re.sub(r'[\U00010000-\U0010ffff]', '', text)
    
    # 3. Restore protected tags
    for i, tag in enumerate(tags):
        text = text.replace(f"__TAG{i}__", tag)
    
    return text.strip()

class DuplexAssistant:
    def __init__(self):
        print("--- [Duplex Assistant] Initializing High-Performance Core ---")
        
        # 1. Models Initialization (Multi-GPU)
        # Reverted to standard 'large-v3-turbo' to leverage existing cache
        print("Loading Whisper ASR (Faster) on GPU 1... (This may take a minute if downloading)")
        self.asr_model = WhisperModel(
            "large-v3-turbo", 
            device="cuda", 
            device_index=1, 
            compute_type="int8_float16"
        )
        
        self.tts_model = OmniVoice.from_pretrained(
            "k2-fsa/OmniVoice",
            device_map="cuda:0",
            dtype=torch.float16,
            load_asr=False
        )
        
        # ASR Prompt Management (To prevent hallucinations)
        self.initial_prompt = "Hinglish conversation."
        
        print("Loading voice prompt...")
        self.voice_prompt = self.tts_model.create_voice_clone_prompt(
            ref_audio=REF_AUDIO_PATH,
            preprocess_prompt=True
        )

        # 2. State Management
        self.loop = None
        self.is_running = True
        self.is_speaking = False
        self.is_user_talking = False
        self.interrupt_event = asyncio.Event()
        
        # 3. Pipelines (Queues)
        self.asr_queue = asyncio.Queue()
        self.llm_queue = asyncio.Queue()
        self.tts_queue = asyncio.Queue()
        self.playback_queue = queue.Queue() # Thread-safe queue for PyAudio callback
        
        # 4. Audio Setup
        self.p = pyaudio.PyAudio()
        self.in_stream = None
        self.out_stream = None
        self.stream_lock = threading.Lock() # For thread-safe stream access
        
        # 5. UI Setup (Rich)
        self.console = Console(theme=Theme({"ai": "bold cyan", "user": "bold green", "sys": "dim italic"}))
        
        # For barge-in detection
        self.last_interrupt_time = 0
        self.stream_error = False

    def _init_streams(self):
        """Standardized way to safely open/reopen audio devices with locking."""
        with self.stream_lock:
            try:
                if self.in_stream:
                    try: self.in_stream.stop_stream(); self.in_stream.close()
                    except: pass
                if self.out_stream:
                    try: self.out_stream.stop_stream(); self.out_stream.close()
                    except: pass

                self.in_stream = self.p.open(
                    format=FORMAT, channels=CHANNELS, rate=RATE,
                    input=True, stream_callback=self.audio_callback,
                    frames_per_buffer=CHUNK
                )
                self.out_stream = self.p.open(
                    format=FORMAT, channels=CHANNELS, rate=8000,
                    output=True
                )
                self.stream_error = False
                return True
            except Exception as e:
                # Silently fail here, watchdog will retry
                return False

    def audio_callback(self, in_data, frame_count, time_info, status):
        """High-performance callback for mic capture and barge-in detection."""
        audio_data = np.frombuffer(in_data, dtype=np.int16)
        energy = np.abs(audio_data).mean()
        
        # Dynamic Echo Suppression: Quadruple threshold while AI is speaking
        # (Crucial to prevent 'self-chatter' where AI hears itself)
        effective_threshold = ENERGY_THRESHOLD * 4.0 if self.is_speaking else ENERGY_THRESHOLD
        
        # Barge-in Detection
        if energy > effective_threshold:
            if self.is_speaking and (time.time() - self.last_interrupt_time > INTERRUPT_COOLDOWN):
                print("\n[Barge-in Detected!]")
                self.loop.call_soon_threadsafe(self.interrupt_event.set)
                self.last_interrupt_time = time.time()
            self.is_user_talking = True
        else:
            self.is_user_talking = False

        # Forward audio to ASR via the loop
        if self.loop:
            self.loop.call_soon_threadsafe(self.asr_queue.put_nowait, in_data)
            
        return (None, pyaudio.paContinue)

    async def asr_worker(self):
        """Processes rolling audio chunks for low-latency transcription."""
        print("[ASR] Worker started.")
        buffer = []
        silence_count = 0
        
        while self.is_running:
            try:
                chunk = await self.asr_queue.get()
                buffer.append(chunk)
                
                audio_data = np.frombuffer(chunk, dtype=np.int16)
                if np.abs(audio_data).mean() < ENERGY_THRESHOLD:
                    silence_count += 1
                else:
                    silence_count = 0
                
                # Safety: Prevent buffer overflow if no silence is detected
                if len(buffer) > MAX_BUFFER_CHUNKS:
                    buffer = buffer[-MAX_BUFFER_CHUNKS:]

                # Immediate Response Logic: If user interrupted, force transcription ASAP
                if self.interrupt_event.is_set() and len(buffer) > 3:
                    silence_count = SILENCE_CHUNKS # Trigger immediately
                
                # If we have a decent chunk and it's followed by some silence, transcribe
                if len(buffer) > 5 and silence_count >= SILENCE_CHUNKS:
                    audio_bytes = b"".join(buffer)
                    audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                    
                    # Run ASR in thread (Higher beam_size for accurate Hindi)
                    segments, info = await asyncio.to_thread(
                        self.asr_model.transcribe, 
                        audio_np, 
                        beam_size=5, 
                        best_of=5,
                        temperature=0.0,
                        vad_filter=True,
                        language="hi",
                        initial_prompt=self.initial_prompt
                    )
                    text = "".join([s.text for s in segments]).strip()
                    
                    # --- Hallucination & Noise Filtering ---
                    # 1. Mirror Prompt Filter (Discard if it matches/contains our initial prompt)
                    if self.initial_prompt.lower() in text.lower() or text.lower() in self.initial_prompt.lower():
                        text = ""
                    
                    # 2. Repetition Filter (Detect "hong hong hong" or "Aap ek AI...")
                    words = text.split()
                    if len(words) > 3:
                        # If a single word makes up more than 50% of a long sentence, it's likely noise
                        counts = {w: words.count(w) for w in set(words)}
                        max_repeat = max(counts.values())
                        if max_repeat > len(words) * 0.5:
                            text = ""
                            
                    # 3. Standard Garbage Filter
                    hallucinations = ["[BLANK_AUDIO]", "Please subscribe", "you", "Thank you for watching"]
                    if text and not any(h in text.lower() for h in hallucinations) and len(text) > 1:
                        self.console.print(f"[user]You: {text}[/user]")
                        # If we had a transcription, trigger LLM
                        await self.llm_queue.put(text)
                    elif text:
                        # Log filtered text in dim colors for debugging
                        self.console.print(f"[sys](Whisper filtered: '{text}')[/sys]")
                    
                    buffer = []
                    silence_count = 0
                    
            except Exception as e:
                print(f"ASR Worker Error: {e}")

    async def llm_worker(self):
        """Streams tokens from Ollama and chunks them for TTS."""
        print("[LLM] Worker started.")
        async with httpx.AsyncClient(timeout=30.0) as client:
            while self.is_running:
                user_text = await self.llm_queue.get()
                
                payload = {
                    "model": LLM_MODEL,
                    "messages": [
                        {
                            "role": "system", 
                            "content": "You are an expressive Indian friend. ALWAYS use Devanagari script for everything. \nYou must express emotions using special tags (integrated into your speech):\n- Use [laughter] when you find something funny or are being lighthearted.\n- Use [sigh] for sadness, relief, or deep thinking.\n- Use [sniff] for sensitivity.\n- Use [dissatisfaction-hnn] for confusion/thinking.\n\nExample:\n- [laughter] अरे भाई! क्या मजेदार बात कही आपने!\n- [sigh] सच में, ये तो बहुत दुख की बात है।\n\nNEVER use English characters. One script, deep emotions."
                        },
                        {"role": "user", "content": user_text}
                    ],
                    "stream": True,
                    "options": {
                        "num_predict": 128,
                        "temperature": 0.5 # Lowered for more consistent Hinglish
                    }
                }
                
                self.console.print("[ai]AI: [/ai]", end="")
                sentence = ""
                try:
                    async with client.stream("POST", LLM_URL, json=payload) as response:
                        async for line in response.aiter_lines():
                            if self.interrupt_event.is_set():
                                self.console.print(" [sys][Interrupted][/sys]")
                                break
                            
                            if line.startswith("data: "):
                                if "[DONE]" in line: break
                                body = json.loads(line[6:])
                                chunk = body.get("choices", [{}])[0].get("delta", {}).get("content", "")
                                sentence += chunk
                                self.console.print(f"[ai]{chunk}[/ai]", end="")
                                
                                # Human-like Flow: Trigger TTS on sentence ends or larger word blocks
                                words = sentence.split()
                                if len(words) >= 18 or any(p in chunk for p in ".!?।\n"):
                                    cleaned = clean_text(sentence)
                                    if cleaned and len(cleaned) > 2:
                                        await self.tts_queue.put(cleaned)
                                    sentence = ""
                                    
                    if sentence.strip() and not self.interrupt_event.is_set():
                        await self.tts_queue.put(clean_text(sentence))
                    print()
                except Exception as e:
                    print(f"\nLLM Stream Error: {e}")

    async def tts_worker(self):
        """Generates audio chunks via OmniVoice in a background thread."""
        print("[TTS] Worker started.")
        while self.is_running:
            text = await self.tts_queue.get()
            
            if self.interrupt_event.is_set():
                continue # Skip if interrupted
            
            try:
                # Heavy generation in thread
                audio_list = await asyncio.to_thread(
                    self.tts_model.generate,
                    text=text,
                    voice_clone_prompt=self.voice_prompt,
                    language="hi", # Explicitly trigger Hindi phonetic engine
                    generation_config=OmniVoiceGenerationConfig(num_step=15) # Optimized Speed/Quality
                )
                
                audio_24k = audio_list[0]
                # Efficient GPU resampling
                audio_8k = await asyncio.to_thread(to_8k, audio_24k, orig_sr=24000, device="cuda:0")
                pcm_data = (audio_8k * 32767).astype(np.int16).tobytes()
                
                if not self.interrupt_event.is_set():
                    self.playback_queue.put(pcm_data)
            except Exception as e:
                print(f"TTS Error: {e}")

    async def playback_worker(self):
        """Asynchronously monitors the playback queue and manages output stream."""
        print("[Playback] Worker started.")
        while self.is_running:
            # Check for interrupt
            if self.interrupt_event.is_set():
                # Flush everything
                while not self.playback_queue.empty(): self.playback_queue.get()
                while not self.tts_queue.empty(): await self.tts_queue.get()
                self.is_speaking = False
                self.interrupt_event.clear()
                print("[Playback] Queues Flushed.")
                continue

            try:
                # Use a non-blocking check on the playback queue
                if not self.playback_queue.empty():
                    pcm_data = self.playback_queue.get_nowait()
                    self.is_speaking = True
                    
                    # Play in chunks with thread-safe lock protection
                    chunk_size = 2048 # samples * bytes
                    for i in range(0, len(pcm_data), chunk_size):
                        if self.interrupt_event.is_set(): break
                        chunk = pcm_data[i:i+chunk_size]
                        with self.stream_lock:
                            if self.out_stream and self.out_stream.is_active():
                                await asyncio.to_thread(self.out_stream.write, chunk)
                            else:
                                self.stream_error = True
                                break
                    
                    if self.playback_queue.empty():
                        self.is_speaking = False
                else:
                    await asyncio.sleep(0.01)
            except Exception as e:
                print(f"Playback Error: {e}")
                self.stream_error = True # Signal watchdog
                await asyncio.sleep(1.0) # Back off

    async def monitor_health(self):
        """Watchdog to ensure streams and workers are healthy."""
        while self.is_running:
            # Check if streams are dead or signaling error
            in_dead = self.in_stream and not self.in_stream.is_active()
            out_dead = self.out_stream and not self.out_stream.is_active()
            
            if in_dead or out_dead or self.stream_error:
                reason = "Mic dead" if in_dead else "Speaker dead/error"
                print(f"[Watchdog] Audio issue detected ({reason}). Restarting streams...")
                await asyncio.to_thread(self._init_streams)
            await asyncio.sleep(3)

    async def run(self):
        self.loop = asyncio.get_running_loop()
        
        # Setup streams
        if not await asyncio.to_thread(self._init_streams):
            print("Failed to initialize audio. Exiting.")
            return
        
        print("\n--- VOICE ASSISTANT READY (DUPLEX MODE) ---")
        print("Barge-in enabled: speak at any time to interrupt the AI.")
        
        # Start core workers
        tasks = [
            asyncio.create_task(self.asr_worker()),
            asyncio.create_task(self.llm_worker()),
            asyncio.create_task(self.tts_worker()),
            asyncio.create_task(self.playback_worker()),
            asyncio.create_task(self.monitor_health()),
        ]
        
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            pass
        finally:
            self.stop()

    def stop(self):
        self.is_running = False
        if self.in_stream:
            self.in_stream.stop_stream()
            self.in_stream.close()
        if self.out_stream:
            self.out_stream.stop_stream()
            self.out_stream.close()
        self.p.terminate()

if __name__ == "__main__":
    assistant = DuplexAssistant()
    try:
        asyncio.run(assistant.run())
    except KeyboardInterrupt:
        print("\nExiting...")
