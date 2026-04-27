#!/usr/bin/env python3
"""
RPi Audio Bridge — HD (slin24) with robust ARI handling.
Fixes vs prior deployed copy:
  * Filter out UnicastRTP StasisStart so we do not try to wrap our own ExternalMedia.
  * Log real HTTP status+body on any ARI failure (no more opaque 'id' KeyError).
  * Use UNICASTRTP_LOCAL_ADDRESS/PORT from response so we know Asterisk RTP target.
Endianness: Asterisk slin24 RTP payload is BIG-ENDIAN (L16) — we byteswap both ways.
"""

import argparse
import json
import os
import socket
import struct
import sys
import threading
import time
import wave
import audioop
import requests
import websocket as ws_client

import silero_vad


# -------------------- Diagnostic capture ------------------------------------
# Lightweight audio recorder. One instance per call, owned by main(). Keeps two
# rolling PCM files (mic-in from Asterisk, tts-in from the AI server) plus a
# per-utterance WAV for the mic path so we can match them against PC-side
# captures and quickly confirm where a bad/empty turn originated.

class Capture:
    def __init__(self, root="/tmp/vaani_captures"):
        ts = time.strftime("%Y%m%d_%H%M%S")
        self.dir = os.path.join(root, ts)
        try:
            os.makedirs(self.dir, exist_ok=True)
            self.rtp_in = open(os.path.join(self.dir, "rtp_incoming_24k_le.pcm"), "ab")
            self.ai_in = open(os.path.join(self.dir, "ai_tts_received_24k_le.pcm"), "ab")
            self.ok = True
            print(f"[CAPTURE] session dir: {self.dir}", flush=True)
        except Exception as e:
            self.ok = False
            print(f"[CAPTURE] disabled (setup failed: {e})", flush=True)
        self.t0 = time.monotonic()
        self._lock = threading.Lock()
        self._utt_idx = 0
        self._tts_idx = 0
        self._current_utt = bytearray()
        self.in_speech = False

    def _safe(self, fn):
        if not self.ok:
            return
        try:
            fn()
        except Exception as e:
            print(f"[CAPTURE] write err: {e}", flush=True)

    def write_rtp_in(self, pcm_le: bytes):
        def _w():
            with self._lock:
                self.rtp_in.write(pcm_le)
                if self.in_speech:
                    self._current_utt.extend(pcm_le)
        self._safe(_w)

    def write_ai_in(self, pcm_le: bytes):
        self._safe(lambda: self.ai_in.write(pcm_le))

    def save_tts_segment(self, pcm_le: bytes):
        def _w():
            self._tts_idx += 1
            path = os.path.join(
                self.dir,
                f"tts_received_{self._tts_idx:03d}_"
                f"{int(time.monotonic() - self.t0)}s_{len(pcm_le)}B.wav",
            )
            _write_wav(path, pcm_le, 16000)
            print(f"[CAPTURE] tts segment → {os.path.basename(path)}", flush=True)
        self._safe(_w)

    def speech_start(self):
        def _w():
            with self._lock:
                self.in_speech = True
                self._current_utt.clear()
        self._safe(_w)

    def speech_end(self):
        if not self.ok:
            return
        try:
            with self._lock:
                self.in_speech = False
                data = bytes(self._current_utt)
                self._current_utt.clear()
            if not data:
                return
            self._utt_idx += 1
            path = os.path.join(
                self.dir,
                f"utterance_{self._utt_idx:03d}_"
                f"{int(time.monotonic() - self.t0)}s_{len(data)}B.wav",
            )
            _write_wav(path, data, 24000)
            print(f"[CAPTURE] utterance → {os.path.basename(path)}", flush=True)
        except Exception as e:
            print(f"[CAPTURE] speech_end save err: {e}", flush=True)

    def close(self):
        for attr in ("rtp_in", "ai_in"):
            try:
                getattr(self, attr).close()
            except Exception:
                pass
        print(f"[CAPTURE] closed (dir={self.dir})", flush=True)


def _write_wav(path: str, pcm_le: bytes, rate: int):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm_le)


class RTPBridge:
    def __init__(self, host="127.0.0.1", port=20000, capture=None):
        self.host, self.port = host, port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.settimeout(0.5)
        self.remote_addr = None
        self.running = False
        self.packets_in = self.packets_out = 0
        self._seq = 0
        self._timestamp = 0
        self._ssrc = 0xABCD1234
        self.on_audio = None
        self._buf = bytearray()
        self._lock = threading.Lock()
        self._buffering = True
        self.capture = capture
        self._first_in_logged = False
        # ulaw mode: static PT=0, 8 kHz, 160 bytes = 20 ms frame.
        self._pt = 0
        self._pt_learned = False
        # ratecv state for 16k LE PCM16 (from PC) -> 8k LE PCM16 (before ulaw).
        self._tx_rate_state = None

    def set_remote(self, host, port):
        self.remote_addr = (host, int(port))
        print(f"[RTP] Remote learned proactively: {self.remote_addr}", flush=True)

    def start(self):
        self.sock.bind((self.host, self.port))
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()
        threading.Thread(target=self._sender, daemon=True).start()
        print(f"[RTP] Listening on {self.host}:{self.port}", flush=True)

    def stop(self):
        self.running = False
        try:
            self.sock.close()
        except Exception:
            pass

    def flush(self):
        with self._lock:
            self._buf.clear()
            self._buffering = True

    def _loop(self):
        while self.running:
            try:
                data, addr = self.sock.recvfrom(4096)
                if self.remote_addr is None:
                    self.remote_addr = addr
                    print(f"[RTP] Remote learned from first packet: {addr}", flush=True)
                self.packets_in += 1
                if len(data) < 12:
                    continue
                pt = data[1] & 0x7F
                payload = data[12:]

                if pt == 0:  # ulaw — our current format
                    p8 = audioop.ulaw2lin(payload, 2)
                    pcm16k, _ = audioop.ratecv(p8, 2, 1, 8000, 16000, None)
                elif pt == 8:  # alaw
                    p8 = audioop.alaw2lin(payload, 2)
                    pcm16k, _ = audioop.ratecv(p8, 2, 1, 8000, 16000, None)
                else:
                    # Dynamic-PT slin16 fallback (BE on wire).
                    pcm16k = audioop.byteswap(payload, 2)

                if not self._first_in_logged and pcm16k:
                    self._first_in_logged = True
                    print(f"[RTP] First inbound PCM: {len(pcm16k)}B pt={pt}", flush=True)

                if self.capture and pcm16k:
                    self.capture.write_rtp_in(pcm16k)

                if self.on_audio and pcm16k:
                    self.on_audio(pcm16k, 16000)
            except socket.timeout:
                continue
            except OSError:
                break
            except Exception as e:
                print(f"[RTP] recv err: {e}", flush=True)

    def send(self, data: bytes):
        # AI sends native LE int16 @ 16 kHz. Downsample to 8 kHz and encode
        # to ulaw here so the packetizer just copies out 160-byte frames.
        pcm8k, self._tx_rate_state = audioop.ratecv(
            data, 2, 1, 16000, 8000, self._tx_rate_state
        )
        ulaw = audioop.lin2ulaw(pcm8k, 2)
        with self._lock:
            self._buf.extend(ulaw)

    def _sender(self):
        while self.running and not self.remote_addr:
            time.sleep(0.05)
        print(f"[RTP] Streaming to {self.remote_addr}", flush=True)
        start_t = time.monotonic()
        pkts = 0
        while self.running:
            chunk = None
            with self._lock:
                # ulaw: 160 bytes = 160 samples × 1 byte per 20ms @ 8 kHz.
                if self._buffering and len(self._buf) >= 800:
                    self._buffering = False
                if not self._buffering:
                    if len(self._buf) >= 160:
                        chunk = bytes(self._buf[:160])
                        del self._buf[:160]
                    else:
                        chunk = b"\xff" * 160  # ulaw silence
                        self._buffering = True
                else:
                    chunk = b"\xff" * 160  # ulaw silence

            if chunk:
                h = struct.pack("!BBHII", 0x80, self._pt, self._seq, self._timestamp, self._ssrc)
                try:
                    self.sock.sendto(h + chunk, self.remote_addr)
                    self.packets_out += 1
                except Exception:
                    pass
                self._seq = (self._seq + 1) & 0xFFFF
                self._timestamp = (self._timestamp + 160) & 0xFFFFFFFF

            pkts += 1
            target = start_t + (pkts * 0.020)
            while time.monotonic() < target:
                time.sleep(0.001)


class ARIBridge:
    def __init__(self, host, port, user, password, app):
        self.base = f"http://{host}:{port}/ari"
        self.auth = (user, password)
        self.ws_url = f"ws://{host}:{port}/ari/events?api_key={user}:{password}&app={app}"
        self.ch_id = self.br_id = self.ex_id = None
        self.app = app
        self._hd_done = False

    def _api(self, m, p, **k):
        try:
            r = requests.request(m, f"{self.base}{p}", auth=self.auth, timeout=5, **k)
            if r.status_code not in (200, 201, 204):
                print(f"[ARI] {m} {p} → {r.status_code}: {r.text[:250]}", flush=True)
            return r
        except Exception as e:
            print(f"[ARI] {m} {p} EXC: {e}", flush=True)
            return None

    def run(self, mode, rtp_host, rtp_port, endpoint, on_start, on_end, rtp):
        def msg(ws, m):
            try:
                ev = json.loads(m)
            except Exception:
                return
            t = ev.get("type")
            if t == "StasisStart":
                ch = ev.get("channel", {})
                name = ch.get("name", "")
                cid = ch.get("id")
                clr = ch.get("caller", {}).get("number", "?")
                # Ignore our own ExternalMedia StasisStart — it triggers a second
                # event for the UnicastRTP channel which must not be re-wrapped.
                if name.startswith("UnicastRTP") or self._hd_done:
                    print(f"[ARI] Skip StasisStart for {name or cid}", flush=True)
                    return
                self.ch_id = cid
                print(f"[ARI] Call: {clr} channel={name} id={cid}", flush=True)

                # 1. Mixing bridge
                r = self._api("POST", "/bridges", json={"type": "mixing"})
                if not r or r.status_code not in (200, 201):
                    print(f"[ARI] Bridge create failed", flush=True)
                    return
                self.br_id = r.json().get("id")

                # 2. Add phone channel to bridge
                self._api("POST", f"/bridges/{self.br_id}/addChannel",
                          params={"channel": self.ch_id})

                # 3. ExternalMedia (ulaw both — the format that *actually*
                # reaches the GSM callee through the Dinstar trunk. slin16 setup
                # succeeded but audio never got to the callee; ulaw is the
                # pre-HD-attempt working baseline.
                r = self._api("POST", "/channels/externalMedia", json={
                    "app": self.app,
                    "external_host": f"{rtp_host}:{rtp_port}",
                    "format": "ulaw",
                    "direction": "both",
                })
                if not r or r.status_code not in (200, 201):
                    print(f"[ARI] HD Setup FAILED (see HTTP log above)", flush=True)
                    return
                body = r.json()
                self.ex_id = body.get("id")
                if not self.ex_id:
                    print(f"[ARI] HD Setup: no id in body {body}", flush=True)
                    return

                # 3b. Proactively set RTP remote from channelvars so we can start
                # sending audio without waiting for inbound RTP handshake.
                cv = body.get("channelvars", {}) or {}
                addr = cv.get("UNICASTRTP_LOCAL_ADDRESS")
                port = cv.get("UNICASTRTP_LOCAL_PORT")
                if addr and port:
                    rtp.set_remote(addr, port)

                # 4. Add ExternalMedia channel to bridge
                self._api("POST", f"/bridges/{self.br_id}/addChannel",
                          params={"channel": self.ex_id})

                self._hd_done = True
                print(f"[ARI] Linked HD Channels: {self.ch_id} <-> {self.ex_id}", flush=True)
                on_start(clr)

            elif t == "StasisEnd":
                ch = ev.get("channel", {})
                cid = ch.get("id")
                # Only tear down when the primary call ends (not our ExternalMedia).
                if cid == self.ch_id:
                    print("[ARI] Primary channel ended", flush=True)
                    on_end()
                    self._cleanup()

        self._ws = ws_client.WebSocketApp(self.ws_url, on_message=msg)

        # Start WS first, then originate so Asterisk sees a live subscriber.
        ws_thread = threading.Thread(target=self._ws.run_forever, daemon=True)
        ws_thread.start()
        time.sleep(0.7)  # let WS connect + register Stasis app
        if mode == "outbound":
            r = self._api("POST", "/channels", json={
                "endpoint": endpoint, "app": self.app, "callerId": "AI-Call"})
            if r and r.status_code in (200, 201):
                print(f"[ARI] Originated: {endpoint}", flush=True)
            else:
                print(f"[ARI] Originate FAILED", flush=True)
        ws_thread.join()

    def _cleanup(self):
        if self.ex_id:
            self._api("DELETE", f"/channels/{self.ex_id}")
        if self.br_id:
            self._api("DELETE", f"/bridges/{self.br_id}")
        self.ex_id = self.br_id = self.ch_id = None
        self._hd_done = False

    def hangup(self):
        if self.ch_id:
            self._api("DELETE", f"/channels/{self.ch_id}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="outbound")
    ap.add_argument("--endpoint")
    ap.add_argument("--server", required=True)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", default=8088, type=int)
    ap.add_argument("--user", default="ari_user")
    ap.add_argument("--password", default="ari_pass")
    ap.add_argument("--rtp-host", default="127.0.0.1")
    ap.add_argument("--rtp-port", default=25000, type=int)
    ap.add_argument("--capture-dir", default="/tmp/vaani_captures",
                    help="Where per-call audio captures are written (disable with empty string)")
    args = ap.parse_args()

    capture = Capture(args.capture_dir) if args.capture_dir else None

    rtp = RTPBridge(args.rtp_host, args.rtp_port, capture=capture)
    ari = ARIBridge(args.host, args.port, args.user, args.password, "vaani")

    ai_ws = None
    ai_connected = threading.Event()
    ai_bytes_received = [0]  # closure-writable counter

    def connect():
        nonlocal ai_ws
        def on_open(ws):
            ai_connected.set()
            print(f"[AI-WS] Connected to {args.server}", flush=True)
        def on_msg(ws, m):
            if isinstance(m, bytes):
                ai_bytes_received[0] += len(m)
                if capture:
                    capture.write_ai_in(m)
                    capture.save_tts_segment(m)
                rtp.send(m)
            else:
                try:
                    ev = json.loads(m)
                    etype = ev.get("type")
                    if etype == "hangup":
                        print("[AI-WS] Hangup request", flush=True)
                        ari.hangup()
                    elif etype == "barge_in":
                        # User interrupted — drop everything queued so the line
                        # goes silent immediately and the user can speak.
                        print("[AI-WS] BARGE-IN -> RTP flush", flush=True)
                        rtp.flush()
                    else:
                        print(f"[AI-WS] event: {etype}", flush=True)
                except Exception:
                    pass
        def on_err(ws, e):
            print(f"[AI-WS] err: {e}", flush=True)
        def on_close(ws, c, r):
            # When the PC disconnects, tear the call down and exit so the next
            # dial doesn't leave a stale bridge holding UDP port 25000.
            print(f"[AI-WS] closed {c} {r} — hanging up and exiting", flush=True)
            ai_connected.clear()
            try:
                ari.hangup()
            except Exception:
                pass
            rtp.stop()
            if capture:
                capture.close()
            os._exit(0)
        ai_ws = ws_client.WebSocketApp(args.server, on_open=on_open,
                                       on_message=on_msg, on_error=on_err,
                                       on_close=on_close)
        ai_ws.run_forever()

    threading.Thread(target=connect, daemon=True).start()
    if not ai_connected.wait(timeout=10):
        print("[AI-WS] FAILED to connect to AI server", flush=True)
        sys.exit(1)

    def _send_json(o):
        try:
            if ai_ws and ai_connected.is_set():
                ai_ws.send(json.dumps(o))
        except Exception:
            pass

    def _send_bytes(b):
        try:
            if ai_ws and ai_connected.is_set():
                ai_ws.send(b, opcode=0x2)
        except Exception:
            pass

    utt_bytes_sent = [0]  # bytes forwarded to AI-WS for the current utterance

    def _on_speech_start():
        utt_bytes_sent[0] = 0
        if capture:
            capture.speech_start()
        print("[VAD] speech_start", flush=True)
        _send_json({"type": "speech_start"})

    def _on_speech_audio(b: bytes):
        utt_bytes_sent[0] += len(b)
        _send_bytes(b)

    def _on_speech_end():
        if capture:
            capture.speech_end()
        n = utt_bytes_sent[0]
        secs = n / (16000 * 2) if n else 0.0
        print(f"[VAD] speech_end — sent {n}B to AI (~{secs:.2f}s @16k)", flush=True)
        _send_json({"type": "speech_end"})

    vad = silero_vad.VADGate(
        on_speech_start=_on_speech_start,
        on_speech_audio=_on_speech_audio,
        on_speech_end=_on_speech_end,
    )

    def on_audio(pcm, rate):
        try:
            vad.feed(pcm, rate)
        except TypeError:
            # Older VADGate API — fall back to single-arg.
            vad.feed(pcm)

    rtp.on_audio = on_audio
    rtp.start()

    # Heartbeat: prints every 5 s so silent gaps are distinguishable from a
    # hung process in /tmp/vaani_bridge.log. Daemon thread dies with process.
    def _heartbeat():
        while True:
            time.sleep(5.0)
            print(f"[HB] rtp_in={rtp.packets_in} rtp_out={rtp.packets_out} "
                  f"ai_conn={ai_connected.is_set()} ai_rx={ai_bytes_received[0]}B "
                  f"remote={rtp.remote_addr}", flush=True)
    threading.Thread(target=_heartbeat, daemon=True).start()

    def on_call_start(caller):
        _send_json({"type": "call_start", "caller": caller, "mode": args.mode})

    def on_call_end():
        _send_json({"type": "call_end"})
        rtp.flush()

    try:
        ari.run(args.mode, args.rtp_host, args.rtp_port, args.endpoint,
                on_call_start, on_call_end, rtp)
    except KeyboardInterrupt:
        ari.hangup()

    rtp.stop()
    if capture:
        capture.close()
    print(f"[DONE] in={rtp.packets_in} out={rtp.packets_out} ai_rx={ai_bytes_received[0]}B", flush=True)


if __name__ == "__main__":
    main()
