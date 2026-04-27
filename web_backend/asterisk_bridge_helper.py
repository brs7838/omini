import asyncio
from typing import Optional
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
from engine import WebAssistant  # type: ignore

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
    def __init__(self, channel_id, bridge_app, log_handler=None, voice_id=None, transcript_handler=None):
        self.channel_id = channel_id
        self.bridge_app = bridge_app
        self.rtp_peer_pt = 11 # [Phase 18] Aligned with standard SLIN/L16 Payload Type
        self.rtp_ssrc = os.urandom(4) # Unique SSRC for every session
        self.log_handler = log_handler
        self.transcript_handler = transcript_handler
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
        self._playback_queue: asyncio.Queue = asyncio.Queue()
        self._playback_task: Optional[asyncio.Task] = None
        self._last_ai_audio_time = 0.0
        self.ai_audio_queue = asyncio.Queue()
        self.transport = None
        self.protocol = None
        
        # [Phase 17 Port Prediction]
        # Fallback to standard Asterisk RTP port until actual port is learned
        self.peer_addr = ('192.168.8.59', 10000)
        
        # [Phase 13/17] Heartbeat task
        self._heartbeat_task: Optional[asyncio.Task] = None

        # RTP inbound rate tracer counters
        self._rtp_in_count = 0
        self._rtp_in_last_ts = time.time()
        self._rtp_tracer_task = None
        self._playback_task = None
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
        # 20ms of 16-bit PCM silence (matches voice ptime for 16k slin16)
        silence_frame = b'\x00' * 640
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
                "format": "slin16" # [HD] Standard for 16kHz audio
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
                self._heartbeat_task = asyncio.create_task(self._run_heartbeat())
                self._rtp_tracer_task = asyncio.create_task(self._run_rtp_tracer())
                self._playback_task = asyncio.create_task(self._run_rtp_playback())

            # 5. Initialize Vaani Assistant (shared models)
            print(f"[Call {self.channel_id}] Initializing AI assistant...", flush=True)
            provider = getattr(self.bridge_app.state, 'llm_provider', None)
            # Per-turn callbacks so a mid-call /provider/switch or /campaign
            # edit applies to the next user turn without dropping the call.
            def _cur_provider():
                return getattr(self.bridge_app.state, 'llm_provider', None)
            def _cur_campaign():
                return getattr(self.bridge_app.state, 'campaign', None)
            self.assistant = WebAssistant(
                event_handler=self.on_ai_event,
                llm_url=provider.url if provider else None,
                llm_model=provider.model if provider else None,
                llm_headers=provider.headers() if provider else None,
                llm_extras=provider.payload_extras() if provider else None,
                get_provider=_cur_provider,
                get_campaign=_cur_campaign,
                input_sampling_rate=16000,  # [HD] Correct for 16kHz Telephony
            )
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

            # Auto-greet removed. We rely on deterministic hand-crafted
            # TTS greeting pushed correctly by the backend logic, not duplicating it here.

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
            
            # Correct interpretation has markedly LOWER RMS than the byte-swapped one.
            # Use a ratio so detection works at normal speech levels, not just extremes.
            if rms_le <= rms_be * 0.66:
                pcm_data = pcm_le  # native LE (Asterisk slin default on RPi/x86)
            elif rms_be <= rms_le * 0.66:
                pcm_data = pcm_be  # gateway sending standard BE L16
            else:
                pcm_data = pcm_le  # ambiguous (silence) — safe LE default
                
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

    async def _run_rtp_playback(self):
        """Dedicated task to drain ai_audio_queue and send RTP frames with precise pacing."""
        print(f"[Call {self.channel_id}] Playback task started.", flush=True)
        while self.is_active:
            try:
                data = await self.ai_audio_queue.get()
                
                # Offload heavy audio processing (resampling, conversion) to a thread
                def process_audio(raw_data):
                    # 1. From 24kHz Native to Float
                    audio_24k = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
                    
                    # 2. Proper polyphase downsample 24k -> 16k (HD).
                    # Ratio 2/3: 24000 * (2/3) = 16000
                    if len(audio_24k) >= 3:
                        audio_16k = signal.resample_poly(audio_24k, 2, 3).astype(np.float32)
                    else:
                        # Fallback simple decimation if too short
                        audio_16k = audio_24k[::1.5] # Approximate
                    
                    # 3. Convert 16-bit PCM Linear to Big-Endian (Standard L16)
                    # Apply 1.5x gain for telephony clarity
                    return (np.clip(audio_16k * 1.5, -1.0, 1.0) * 32767).astype('>i2').tobytes()

                pcm_16k = await asyncio.to_thread(process_audio, data)

                if self.transport and self.peer_addr:
                    # [HD] 16kHz 20ms RTP frame is 640 bytes (320 samples * 2 bytes)
                    FRAME_BYTES = 640 
                    frames_sent = 0
                    session_start_time = time.perf_counter()
                    
                    for i in range(0, len(pcm_16k), FRAME_BYTES):
                        if not self.is_active: break
                        frame = pcm_16k[i:i + FRAME_BYTES]
                        if len(frame) < FRAME_BYTES:
                            frame += b'\x00' * (FRAME_BYTES - len(frame))
                        
                        self.send_rtp_frame(frame)
                        frames_sent += 1
                        
                        expected_elapsed = (frames_sent * 0.020)
                        sleep_for = expected_elapsed - (time.perf_counter() - session_start_time)
                        
                        # Windows asyncio.sleep has ~15ms minimum resolution. 
                        # We sleep for the bulk, then spin-yield for precision.
                        if sleep_for > 0.015:
                            await asyncio.sleep(sleep_for - 0.015)
                        while (time.perf_counter() - session_start_time) < expected_elapsed:
                            await asyncio.sleep(0)
                    
                    self._last_ai_audio_time = time.time()
                self.ai_audio_queue.task_done()
            except Exception as e:
                print(f"[Call {self.channel_id}] Playback error: {e}", flush=True)
                await asyncio.sleep(0.1)

    async def on_ai_event(self, msg_type, data):
        """Handle events from WebAssistant."""
        if msg_type == "audio_chunk":
            if not self.is_active:
                return
            # [NON-BLOCKING] Push to queue to unblock engine's tts_worker
            self.ai_audio_queue.put_nowait(data)
        elif msg_type == "transcript":
            role = data.get('role', 'ai')
            text = data.get('text', '')
            print(f"[Bridge-Transcript] {role.upper()}: {text}")
            # Push to conversation UI (/ws voice_clients) so live transcript
            # appears during Asterisk calls — same channel browser mode uses.
            if self.transcript_handler:
                await self.transcript_handler(role, text)
            if self.log_handler:
                await self.log_handler("transcript", {"role": role.upper(), "text": text, "source": "asterisk"})
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
        if hasattr(self, '_playback_task') and self._playback_task:
            self._playback_task.cancel()

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
    def __init__(self, log_handler=None, transcript_handler=None):
        self.sessions = {}
        self.log_handler = log_handler
        self.transcript_handler = transcript_handler
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

                            # Skip channels originated by rpi_audio_bridge.py —
                            # those are handled by /ws/phone on this server.
                            caller_info = channel.get("caller", {})
                            if (caller_info.get("name") == "AI-Call"
                                    or caller_info.get("number") == "AI-Call"):
                                print(f"[Bridge] Skipping RPi bridge channel: {channel_id}", flush=True)
                                continue

                            caller = channel.get("caller", {}).get("number", "Unknown")
                            args = event.get("args", [])
                            voice_id = args[2] if len(args) >= 3 else None
                            msg = f"INCOMING CALL: {channel_id} (From: {caller}, Voice: {voice_id or 'default'})"
                            print(f"\n[Bridge] {msg}", flush=True)
                            if self.log_handler: await self.log_handler("asterisk", msg)

                            session = AsteriskCallSession(channel_id, self, log_handler=self.log_handler, voice_id=voice_id, transcript_handler=self.transcript_handler)
                            self.sessions[channel_id] = session
                            asyncio.create_task(session.start())
                            
                        elif event_type == "StasisEnd":
                            channel_id = event["channel"]["id"]
                            if channel_id in self.sessions:
                                print(f"[Bridge] CALL ENDED: {channel_id}", flush=True)
                                await self.sessions[channel_id].stop()
                                del self.sessions[channel_id]

                        elif event_type == "ChannelDestroyed":
                            # Log hangup cause for every outbound originate that
                            # dies without ever hitting Stasis. Without this, a
                            # failed call (unreachable gateway, 403 from SIP
                            # trunk, GSM no-answer, etc.) silently shows "200 OK"
                            # in the backend log because the only thing recorded
                            # is the originate HTTP response — the real failure
                            # reason is buried in ARI events we weren't capturing.
                            channel = event.get("channel", {})
                            cname = channel.get("name", "?")
                            cid = channel.get("id", "?")
                            cstate = channel.get("state", "?")
                            cause = event.get("cause")
                            cause_txt = event.get("cause_txt") or "(no text)"
                            # Skip noisy UnicastRTP destroys — those are our
                            # ExternalMedia cleanup, not real call failures.
                            if not cname.startswith("UnicastRTP"):
                                msg = (f"CHANNEL DESTROYED {cid} name={cname} "
                                       f"final_state={cstate} cause={cause} "
                                       f"reason='{cause_txt}'")
                                print(f"[Bridge] {msg}", flush=True)
                                if self.log_handler:
                                    await self.log_handler("asterisk", msg)

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
