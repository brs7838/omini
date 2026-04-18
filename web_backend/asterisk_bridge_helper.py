import asyncio
import json
import socket
import struct
import time
import base64
import random
import audioop
import numpy as np
import httpx
from scipy import signal
import websockets
import os
import sys

# Ensure imports work when run from within web_backend
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from engine import WebAssistant

# --- Voice Loading ---
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VOICES_FILE = os.path.join(ROOT_DIR, "assets", "voices", "voices.json")

def load_voices():
    if not os.path.exists(VOICES_FILE):
        return []
    with open(VOICES_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

# --- Configuration ---
ARI_URL = "http://192.168.8.59:8088/ari"
ARI_WS_URL = "ws://192.168.8.59:8088/ari/events?app=ai-call-app&api_key=ari_user:ari_pass"
ARI_AUTH = ("ari_user", "ari_pass")

class RTPProtocol(asyncio.DatagramProtocol):
    def __init__(self, on_data_received):
        self.on_data_received = on_data_received
        self.transport = None

    def connection_made(self, transport):
        self.transport = transport

    def datagram_received(self, data, addr):
        self.on_data_received(data, addr)

class AsteriskCallSession:
    def __init__(self, channel_id, bridge_app, log_handler=None, voice_id=None):
        self.channel_id = channel_id
        self.bridge_app = bridge_app
        self.rtp_peer_pt = 11 # [Phase 18] Aligned with standard SLIN/L16 Payload Type
        self.rtp_ssrc = os.urandom(4) # Unique SSRC for every session
        self.log_handler = log_handler
        self.voice_id = voice_id
        self.external_channel_id = None
        self.bridge_id = None
        self.assistant = None
        self.is_active = True
        self.peer_addr = None
        self._pt_learned = False
        self.rtp_seq = random.randint(1000, 5000)
        self.rtp_ts = random.randint(10000, 50000)
        self.rtp_buffer = bytearray()
        self.rtp_buffer_count = 0
        self.transport = None
        self.protocol = None
        
        # [Phase 17 Port Prediction]
        # Fallback to standard Asterisk RTP port until actual port is learned
        self.peer_addr = ('192.168.8.59', 10000)
        
        # [Phase 13/17] Heartbeat task
        self._heartbeat_task = None
        self._last_ai_audio_time = 0

        # RTP inbound rate tracer counters
        self._rtp_in_count = 0
        self._rtp_in_last_ts = time.time()
        self._rtp_tracer_task = None
        # RMS tracking for the inbound tracer — sum of per-packet RMS, divided
        # by count to get the average signal level during each 5s window.
        self._rtp_rms_sum = 0.0
        self._rtp_rms_peak = 0.0
        
    async def _run_rtp_tracer(self):
        """Every 5s, emit inbound RTP packet rate to the log stream."""
        while self.is_active:
            try:
                await asyncio.sleep(5.0)
                now = time.time()
                elapsed = now - self._rtp_in_last_ts
                if elapsed <= 0:
                    continue
                pps = self._rtp_in_count / elapsed
                avg_rms = (self._rtp_rms_sum / self._rtp_in_count) if self._rtp_in_count else 0.0
                msg = (f"[Call {self.channel_id}] RTP-IN {self._rtp_in_count} pkt / {elapsed:.1f}s = {pps:.1f} pps "
                       f"| avg_rms={avg_rms:.4f} peak={self._rtp_rms_peak:.4f}")
                print(msg, flush=True)
                if self.log_handler:
                    await self.log_handler("asterisk", msg)
                self._rtp_in_count = 0
                self._rtp_rms_sum = 0.0
                self._rtp_rms_peak = 0.0
                self._rtp_in_last_ts = now
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[Call {self.channel_id}] RTP tracer error: {e}", flush=True)

    async def _run_heartbeat(self):
        """Send silent RTP frames to keep connection alive during thinking/pauses."""
        print(f"[Call {self.channel_id}] Heartbeat (Comfort Noise) task started.", flush=True)
        # 20ms of 16-bit PCM silence (matches voice ptime for 8k slin)
        silence_frame = b'\x00' * 320
        while self.is_active:
            try:
                # Only send if AI hasn't spoken in the last 100ms
                if (time.time() - self._last_ai_audio_time) > 0.1:
                    if self.peer_addr:
                        self.send_rtp_frame(silence_frame) # [Phase 17.2] Correct method
                await asyncio.sleep(0.02)
            except Exception as e:
                # Suppress flood of errors if transport is not ready
                await asyncio.sleep(1)

    def send_rtp_frame(self, frame):
        """Build and send a single RTP packet (320 bytes / 20ms L16)."""
        if not self.transport or not self.peer_addr:
            return
        
        rtp_packet = self._build_rtp_packet(frame)
        try:
            self.transport.sendto(rtp_packet, self.peer_addr)
        except Exception:
            pass

    async def start(self):
        try:
            msg = f"[Call {self.channel_id}] Starting session..."
            print(msg, flush=True)
            if self.log_handler: await self.log_handler("asterisk", msg)

            # 1. Answer the call
            async with httpx.AsyncClient(auth=ARI_AUTH, timeout=10.0) as client:
                resp = await client.post(f"{ARI_URL}/channels/{self.channel_id}/answer")
                print(f"[Call {self.channel_id}] Answer response: {resp.status_code}", flush=True)

            # 2. Setup UDP listener for RTP using asyncio for better stability
            loop = asyncio.get_running_loop()
            listen_addr = ("0.0.0.0", 0)
            self.transport, self.protocol = await loop.create_datagram_endpoint(
                lambda: RTPProtocol(self.on_rtp_received),
                local_addr=listen_addr
            )
            self.udp_port = self.transport.get_extra_info('sockname')[1]
            print(f"[Call {self.channel_id}] Listening for RTP on port {self.udp_port}", flush=True)

            # 3. Create External Media channel in Asterisk (Forcing ulaw)
            external_media_data = {
                "app": "ai-call-app",
                "external_host": f"{self.bridge_app.local_ip}:{self.udp_port}",
                "format": "slin"
            }

            async with httpx.AsyncClient(auth=ARI_AUTH, timeout=10.0) as client:
                # Use params= (query string) instead of json= body — ARI parses
                # query params reliably; JSON body support varies by version.
                resp = await client.post(f"{ARI_URL}/channels/externalMedia", params=external_media_data)
                if resp.status_code != 200:
                    print(f"[Call {self.channel_id}] Failed to create ExternalMedia: {resp.text}", flush=True)
                    return
                ext_resp = resp.json()
                self.external_channel_id = ext_resp["id"]
                
                # [STABILITY] Proactively learn peer_addr from ARI response
                # This allows us to start sending audio IMMEDIATELY without waiting 
                # for the first packet from Asterisk (eliminates handshake delay).
                vars = ext_resp.get('channelvars', {})
                ast_host = vars.get('UNICASTRTP_LOCAL_ADDRESS')
                ast_port = vars.get('UNICASTRTP_LOCAL_PORT')
                if ast_host and ast_port:
                    self.peer_addr = (ast_host, int(ast_port))
                    print(f"[Call {self.channel_id}] Proactively learned Asterisk RTP path: {self.peer_addr}", flush=True)

                print(f"[Call {self.channel_id}] ExternalMedia created: {self.external_channel_id}", flush=True)

            # 4. Bridge them
            async with httpx.AsyncClient(auth=ARI_AUTH, timeout=10.0) as client:
                resp = await client.post(f"{ARI_URL}/bridges", params={"type": "mixing"})
                self.bridge_id = resp.json()["id"]
                print(f"[Call {self.channel_id}] Bridge created: {self.bridge_id}", flush=True)
                # Add both channels to bridge
                resp2 = await client.post(f"{ARI_URL}/bridges/{self.bridge_id}/addChannel", params={"channel": f"{self.channel_id},{self.external_channel_id}"})
                print(f"[Call {self.channel_id}] Channels bridged: {resp2.status_code}", flush=True)

                # [Phase 17] START HEARTBEAT IMMEDIATELY
                # Send silence immediately to "warm up" the gateway and prevent auto-reject.
                self._heartbeat_task = asyncio.create_task(self._run_heartbeat())
                self._rtp_tracer_task = asyncio.create_task(self._run_rtp_tracer())

            # 5. Initialize Vaani Assistant (shared models)
            print(f"[Call {self.channel_id}] Initializing AI assistant...", flush=True)
            self.assistant = WebAssistant(event_handler=self.on_ai_event, input_sampling_rate=8000)
            self.assistant.reset_session() # Reset session clock and state
            print(f"[Call {self.channel_id}] AI assistant ready.", flush=True)

            # 6. Load selected voice (persona + voice clone)
            if self.voice_id and self.voice_id != "ravi":
                try:
                    voices = load_voices()
                    voice = next((v for v in voices if v["id"] == self.voice_id), None)
                    if voice:
                        self.assistant.active_voice_metadata = {
                            "name": voice.get("name", "AI"),
                            "gender": voice.get("gender", "male"),
                            "age": voice.get("age", "30"),
                            "about": voice.get("about", ""),
                            "dialogues": voice.get("catchphrases", ""),
                            "style": voice.get("style", "conversational"),
                        }
                        full_audio_path = os.path.join(ROOT_DIR, voice["file_path"])
                        if os.path.exists(full_audio_path):
                            self.assistant.voice_prompt = await asyncio.to_thread(
                                self.assistant.tts_model.create_voice_clone_prompt,
                                ref_audio=full_audio_path,
                                preprocess_prompt=False  # [FAST] False = no extra processing, matches fast_voice.py
                            )
                            print(f"[Call {self.channel_id}] Voice loaded: {voice['name']}", flush=True)
                        else:
                            print(f"[Call {self.channel_id}] Voice file missing: {full_audio_path}, using default", flush=True)
                except Exception as e:
                    print(f"[Call {self.channel_id}] Voice load failed: {e}, using default", flush=True)

            self.assistant_task = asyncio.create_task(self.assistant.run())

            # Wait for Asterisk RTP path to be established
            for _ in range(600):  # Wait up to 3 seconds (5ms intervals)
                if self.peer_addr:
                    break
                await asyncio.sleep(0.005)
            if self.peer_addr:
                print(f"[Call {self.channel_id}] RTP path ready: {self.peer_addr}", flush=True)
            else:
                msg = f"[Call {self.channel_id}] WARNING: Still no RTP from Asterisk. Please ensure Windows Firewall allows UDP on all ports for Python!"
                print(msg, flush=True)
                if self.log_handler: await self.log_handler("asterisk", msg)

            # Auto-greet in selected voice (single path — direct TTS, then END_RESPONSE
            # so is_speaking resets cleanly and VAD opens up for the user's first turn).
            name = self.assistant.active_voice_metadata.get("name", "Vaani")
            gender_verb = "raha" if self.assistant.active_voice_metadata.get("gender") == "male" else "rahi"
            greet = f"नमस्ते! मैं {name} बोल {gender_verb} हूँ। आज मैं आपकी क्या सहायता कर सकता हूँ?"
            await self.assistant.tts_queue.put(greet)
            await self.assistant.tts_queue.put("__END_RESPONSE__")

            print(f"[Call {self.channel_id}] Fully connected and bridged to shared AI core.", flush=True)
        except Exception as e:
            import traceback
            print(f"[Call {self.channel_id}] FATAL ERROR in session.start(): {e}", flush=True)
            traceback.print_exc()
            import sys; sys.stdout.flush(); sys.stderr.flush()

    def on_rtp_received(self, data, addr):
        """Callback from RTPProtocol when data arrives."""
        if not self.is_active: return
        self.peer_addr = addr
        if not self._pt_learned and len(data) >= 2:
            self.rtp_peer_pt = data[1] & 0x7F
            self._pt_learned = True
            print(f"[Call {self.channel_id}] Learned Asterisk RTP PT={self.rtp_peer_pt} from {addr}", flush=True)
        # Strip 12-byte RTP header to get raw PCM audio
        if len(data) <= 12:
            return
        
        # Diagnostic: Handle both Big and Little Endian slin
        raw_payload = data[12:]
        
        # [AUTODETECT] Determine endianness by analyzing RMS of both interpretations
        # (This is the most robust way to handle RPi vs Windows conflicts)
        try:
            # Try Little-Endian (Native)
            pcm_le = np.frombuffer(raw_payload, dtype=np.int16).astype(np.float32) / 32768.0
            rms_le = np.sqrt(np.mean(pcm_le**2))
            
            # Try Big-Endian (Standard L16)
            pcm_be = np.frombuffer(raw_payload, dtype='>i2').astype(np.float32) / 32768.0
            rms_be = np.sqrt(np.mean(pcm_be**2))
            
            # The one with REALLY high RMS is usually the one with 'byte-swapped' noise
            if rms_be > 0.7 and rms_le < 0.4:
                pcm_data = pcm_le # Gateway is sending Little Endian
            elif rms_le > 0.7 and rms_be < 0.4:
                pcm_data = pcm_be # Gateway is sending Big Endian (Standard)
            else:
                pcm_data = pcm_be # Fallback to standard
                
        except Exception as e:
            print(f"[Call {self.channel_id}] Decoder Error: {e}")
            return
        
        # Track float RMS of the CHOSEN interpretation (pre-int16 conversion) so
        # the tracer can report the real inbound signal level.
        pkt_rms = float(np.sqrt(np.mean(pcm_data ** 2)))
        self._rtp_rms_sum += pkt_rms
        if pkt_rms > self._rtp_rms_peak:
            self._rtp_rms_peak = pkt_rms

        # [FIX] Scale float32 [-1, 1] back up to int16 range BEFORE casting.
        # The previous `pcm_data.astype(np.int16)` on a float array in [-1, 1]
        # truncated every sample to 0, silencing all inbound audio to the ASR.
        pcm_bytes = (np.clip(pcm_data, -1, 1) * 32767).astype(np.int16).tobytes()
        self.rtp_buffer.extend(pcm_bytes)
        self.rtp_buffer_count += 1
        self._rtp_in_count += 1
        
        # [FAST] Buffer ~100ms of audio (5 packets at 20ms each) before pushing
        # to ASR queue. Reduced from 15 (300ms) to 5 (100ms) for faster response.
        if self.rtp_buffer_count >= 5 and self.assistant:
            self.assistant.asr_queue.put_nowait(bytes(self.rtp_buffer))
            self.rtp_buffer = bytearray()
            self.rtp_buffer_count = 0

    def _build_rtp_packet(self, pcm_payload: bytes) -> bytes:
        """Wrap a PCM payload in a 12-byte RTP header using PT 10."""
        header = struct.pack("!BBH", 0x80, 10, self.rtp_seq & 0xFFFF)
        header += struct.pack("!I", self.rtp_ts & 0xFFFFFFFF)
        header += self.rtp_ssrc
        self.rtp_seq += 1
        # 160 samples per 20ms, 320 samples per 40ms
        self.rtp_ts += len(pcm_payload) // 2
        return header + pcm_payload

    async def on_ai_event(self, msg_type, data):
        """Handle events from WebAssistant."""
        if msg_type == "audio_chunk":
            if not self.is_active:
                return
            # Offload heavy audio processing (decimation, conversion) to a thread
            def process_audio(raw_data):
                # 1. From 24kHz Native to Float
                audio_24k = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
                
                # 2. Improved Resampling: Avoid aliasing noise by using a 3-tap mean filter
                # (Simple but effective for 24k -> 8k conversion)
                if len(audio_24k) >= 3:
                    # Pad to make it divisible by 3
                    n = len(audio_24k)
                    trimmed = audio_24k[:n - (n % 3)]
                    audio_8k = trimmed.reshape(-1, 3).mean(axis=1)
                else:
                    audio_8k = audio_24k[::3]
                
                # 3. Convert 16-bit PCM Linear to Big-Endian (Standard L16)
                # Apply 2.0x POWER GAIN for telephony clarity (Standard for noisy analog links)
                return (np.clip(audio_8k * 2.0, -1, 1) * 32767).astype('>i2').tobytes()

            pcm_8k = await asyncio.to_thread(process_audio, data)

            # [STABILITY] Skip wait if peer_addr was proactively learned from ARI
            if self.peer_addr:
                pass 
            else:
                print(f"[Call {self.channel_id}] No peer_addr yet. Waiting for Asterisk to wake up...")
                for _ in range(250):  # Wait up to 5 seconds for the first packet
                    if self.peer_addr or not self.is_active:
                        break
                    await asyncio.sleep(0.02)

            if self.transport and self.peer_addr:
                # [Phase 20] Send as 20ms RTP frames for Crystal Clarity
                # 20ms @ 8kHz = 160 samples = 320 bytes
                FRAME_BYTES = 320 
                frames_sent = 0
                
                session_start_time = time.perf_counter()
                
                for i in range(0, len(pcm_8k), FRAME_BYTES):
                    if not self.is_active: break
                    frame = pcm_8k[i:i + FRAME_BYTES]
                    if len(frame) < FRAME_BYTES:
                        frame += b'\x00' * (FRAME_BYTES - len(frame))
                    
                    rtp_packet = self._build_rtp_packet(frame)
                    
                    try:
                        self.transport.sendto(rtp_packet, self.peer_addr)
                        frames_sent += 1
                    except Exception: break
                    
                    # Precise Pacing: 20ms per frame (matches FRAME_BYTES=320 @ 8kHz)
                    expected_elapsed = (frames_sent * 0.020)
                    
                    while True:
                        actual_elapsed = time.perf_counter() - session_start_time
                        remaining = expected_elapsed - actual_elapsed
                        if remaining > 0.005: 
                            await asyncio.sleep(remaining - 0.002)
                        else: break
                    
                    while (time.perf_counter() - session_start_time) < expected_elapsed:
                        pass
                
                # Update last audio time to suppress heartbeat
                self._last_ai_audio_time = time.time()
                
                if frames_sent > 0:
                    print(f"[Call {self.channel_id}] Sent {frames_sent} 40ms RTP frames (Phase 8 Robust).", flush=True)
            elif not self.peer_addr:
                print(f"[Call {self.channel_id}] WARNING: No peer_addr after wait, audio DROPPED ({len(pcm_8k)} bytes)")
            else:
                print(f"[Call {self.channel_id}] WARNING: No transport, audio dropped")
        elif msg_type == "transcript":
            role = data['role'].upper()
            text = data['text']
            print(f"[Bridge-Transcript] {role}: {text}")
            if self.log_handler:
                await self.log_handler("transcript", {"role": role, "text": text, "source": "asterisk"})
        else:
            # Forward every other msg_type (log, status, llm_chunk, ...) to /ws/logs
            # so the browser LogViewer sees engine pipeline events during Asterisk calls.
            if self.log_handler:
                await self.log_handler(msg_type, data)

    async def stop(self):
        self.is_active = False
        # Flush remaining RTP buffer
        if self.rtp_buffer and self.assistant:
            try:
                self.assistant.asr_queue.put_nowait(bytes(self.rtp_buffer))
            except Exception:
                pass
        print(f"[Call {self.channel_id}] Hanging up...")
        if self.assistant:
            self.assistant.stop()

        if self._rtp_tracer_task:
            self._rtp_tracer_task.cancel()
        if self._heartbeat_task:
            self._heartbeat_task.cancel()

        if self.transport:
            self.transport.close()

        async with httpx.AsyncClient(auth=ARI_AUTH) as client:
            try:
                await client.delete(f"{ARI_URL}/channels/{self.channel_id}")
                if self.external_channel_id:
                    await client.delete(f"{ARI_URL}/channels/{self.external_channel_id}")
                if hasattr(self, 'bridge_id'):
                    await client.delete(f"{ARI_URL}/bridges/{self.bridge_id}")
            except:
                pass
        
        print(f"[Call {self.channel_id}] Terminated.")

class VaaniAsteriskBridge:
    def __init__(self, log_handler=None):
        self.sessions = {}
        self.log_handler = log_handler
        self.local_ip = self.detect_local_ip()
        self.is_running = True
        print(f"[Bridge] Initialized on local IP: {self.local_ip}")

    def detect_local_ip(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            # Current IP of Asterisk Server (RPi)
            s.connect(('192.168.8.59', 1))
            ip = s.getsockname()[0]
        except Exception:
            ip = '127.0.0.1'
        finally:
            s.close()
        return ip

    async def run(self):
        print(f"[Bridge] Starting... Connecting to ARI at {ARI_WS_URL}", flush=True)
        while self.is_running:
            try:
                # Add connection timeout to avoid hanging indefinitely if IP is wrong
                async with websockets.connect(ARI_WS_URL, open_timeout=10) as ws:
                    print("[Bridge] ARI Connected. Ready to handle calls.", flush=True)
                    async for message in ws:
                        if not self.is_running: break
                        event = json.loads(message)
                        event_type = event.get("type")
                        
                        if event_type == "StasisStart":
                            channel = event["channel"]
                            channel_id = channel["id"]
                            channel_name = channel.get("name", "")

                            # Skip ExternalMedia (UnicastRTP) channels — they are
                            # created by us inside session.start().  Without this
                            # guard each ExternalMedia triggers another StasisStart
                            # causing an infinite loop of sessions.
                            if channel_name.startswith("UnicastRTP"):
                                print(f"[Bridge] Skipping ExternalMedia channel: {channel_name}", flush=True)
                                continue

                            # Also skip if this channel is already tracked as an
                            # external channel of an existing session.
                            is_external = any(
                                s.external_channel_id == channel_id
                                for s in self.sessions.values()
                            )
                            if is_external:
                                continue

                            caller = channel.get("caller", {}).get("number", "Unknown")
                            args = event.get("args", [])
                            voice_id = args[2] if len(args) >= 3 else None
                            msg = f"INCOMING CALL: {channel_id} (From: {caller}, Voice: {voice_id or 'default'})"
                            print(f"\n[Bridge] {msg}", flush=True)
                            if self.log_handler: await self.log_handler("asterisk", msg)

                            session = AsteriskCallSession(channel_id, self, log_handler=self.log_handler, voice_id=voice_id)
                            self.sessions[channel_id] = session
                            asyncio.create_task(session.start())
                            
                        elif event_type == "StasisEnd":
                            channel_id = event["channel"]["id"]
                            if channel_id in self.sessions:
                                print(f"[Bridge] CALL ENDED: {channel_id}", flush=True)
                                await self.sessions[channel_id].stop()
                                del self.sessions[channel_id]

            except Exception as e:
                if self.is_running:
                    print(f"[Bridge] ARI Connection lost or failed ({e}). Retrying in 5s...", flush=True)
                    await asyncio.sleep(5)
                else:
                    break

    async def hangup_session(self, channel_id):
        """Remotely terminate a specific session."""
        if channel_id in self.sessions:
            print(f"[Bridge] Manual hangup requested for {channel_id}")
            await self.sessions[channel_id].stop()
            del self.sessions[channel_id]
            return True
        return False

    async def hangup_all(self):
        """Terminate all active calls."""
        channels = list(self.sessions.keys())
        for cid in channels:
            await self.hangup_session(cid)

    def stop(self):
        self.is_running = False
        print("[Bridge] Stopping...")
