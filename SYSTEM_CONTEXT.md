# Omini / Vaani Web — Live System Context

> **Purpose**: Single source of truth for continuing bug-fix work. Written from direct inspection of the code on disk and live state on the Asterisk Pi (2026-04-18).
> **Scope**: What is *actually* true right now, not what the original spec doc claimed. Where the spec and reality disagree, reality wins and the gap is noted.

---

## 1. Verified Infrastructure

### Windows PC — 192.168.8.2
- Runs FastAPI (`web_backend/main.py`, port 8000) and Next.js 16 frontend (port 3000)
- GPU split: RTX 3060 (cuda:0) → OmniVoice TTS; GTX 1650 (cuda:1) → Whisper ASR
- Ollama runs `gemma3:4b` on **RTX 3060** (`num_gpu=99`) pinned via `CUDA_VISIBLE_DEVICES=0` by the launcher (`start_vaani_web.py` → `restart_ollama_pinned()`). Shares the 3060 with OmniVoice.
- uv-managed Python 3.12.9 venv at `.venv\`

### Raspberry Pi — 192.168.8.59 (verified via SSH)
- **OS**: `Ubuntu aarch64 6.8.0-1051-raspi` (NOT Raspbian — spec doc is wrong on this)
- **Asterisk**: `20.6.0~dfsg+~cs6.13.40431414-2build5` ✅ matches spec
- **Hostname**: `host2`
- **ARI HTTP**: http://192.168.8.59:8088/ari, creds `ari_user` / `ari_pass` (plain)
- **ARI WS app**: `ai-call-app` — currently registered, backend was connected
- **RTP range**: 10000–20000 (from `rtp.conf`)
- **Logger**: only console + `messages.log`; `/var/log/asterisk/full` does NOT exist (spec assumed it did). Use `journalctl -u asterisk` for boot events.

### Dinstar FXO — 192.168.8.60
- SIP trunk; Asterisk registers endpoints 1001–1032 against it
- Contact format: `sip:192.168.8.60` for each endpoint
- Dinstar password convention: endpoint num = password (1001/1001, 1002/1002, …)

### Asterisk config summary (verified)
- `pjsip.conf`: 32 endpoints 1001–1032. **All endpoints allow `ulaw` only, not slin.** Asterisk transcodes ulaw↔slin for the ExternalMedia channel internally.
- **Endpoint 1017 is the only one routed to `context=ai-incoming`** (Stasis app). All others (including **1001 used for outbound dials**) are routed to `basic-context`, which uses a 32-digit extension pattern and calls `/home/pi/Documents/32GSMgatewayServer/dtmf.sh`. This is a coexisting "32-port GSM gateway server" dialplan that pre-dates the AI integration.
- `extensions.conf`: has `[stasis-test]` and `[ai-incoming]` contexts, both calling `Stasis(ai-call-app)`. An inbound-AI call path exists but is only wired to endpoint 1017.
- **Outbound from the Python backend works by ARI-originating with `app: ai-call-app`**, which bypasses the dialplan context entirely — so the `basic-context` routing doesn't interfere. This is why dialing still works despite 1001 being in the "wrong" context.

---

## 2. File Layout (verified, line counts current as of this doc)

```
E:\Ai\Omini with Astrisk\Omini\
├── start_vaani_web.py                 (149 lines)  super-launcher
├── calls_history.json                 call log (316 lines, last entry 2026-04-18 12:44)
├── backend.log                        last run ended on a KeyboardInterrupt
├── README.md, LICENSE, pyproject.toml, requirements.txt, uv.lock
├── assets/voices/
│   ├── voices.json                    11 voice personas (ravi + 10 clones)
│   └── *.mp3 / *.wav                  ref audio (includes ravi_sir.mp3, ravi_sir_8s.wav)
├── web_backend/
│   ├── main.py                        (659)  FastAPI + WS
│   ├── engine.py                      (500)  ASR+LLM+TTS pipeline
│   ├── asterisk_bridge_helper.py      (504)  ARI bridge + RTP
│   ├── audio_utils.py                 (26)   pydub trim
│   ├── bulk_import.py, test_ws.py
│   └── scratch/cleanup_asterisk.py    manual zombie channel reaper
├── web_frontend/src/
│   ├── app/{layout,page}.tsx, globals.css
│   ├── components/ AssistantOrb, VocalisOrb, Starfield, Sidebar, Dialer, CallHistory, VoiceLibrary, LogViewer
│   └── hooks/useVoice.ts
├── omnivoice/                         TTS library source
└── scratch/                           experimental code (duplex_assistant.py, fast_voice.py, etc.)
```

---

## 3. Confirmed Bugs (from reading the code, not speculation)

### BUG-1 — `LLM_MODEL` is undefined in `engine.py` ⚠️ BLOCKER
- `engine.py:315` references `"model": LLM_MODEL` inside the LLM payload, but the symbol is **not defined or imported** in `engine.py` anywhere.
- It is defined in `scratch/voice_assistant.py:30`, `scratch/fast_voice.py:25`, `scratch/duplex_assistant.py:32` only.
- The first LLM call in any session will raise `NameError: name 'LLM_MODEL' is not defined` inside `llm_worker`, killing that task.
- Suspicion: the error is swallowed by the broad `except Exception as e: print(f"LLM Error: {e}")` at `engine.py:393`, which is why the AI produces no spoken response even when ASR fires. The user sees "no response" rather than a crash.
- Fix direction: add `LLM_MODEL = "gemma3:4b"` alongside the other constants at the top of `engine.py`.

### BUG-2 — RTP pacing is 2× too slow (AI voice plays at half speed)
- `asterisk_bridge_helper.py:339-363`: frames are 320 bytes = 160 samples = **20 ms** @ 8 kHz
- But the pacing loop uses `expected_elapsed = (frames_sent * 0.040)` — paces one 20 ms frame every 40 ms
- Effect: AI audio is streamed to Asterisk at 50% real-time. User hears voice stretched / with big gaps.
- Spec doc already calls this out as known ("Ongoing Improvement Areas" item 2) but it hasn't been fixed.
- Fix: `expected_elapsed = (frames_sent * 0.020)`.

### BUG-3 — Double-greeting at start of every outbound call
- `asterisk_bridge_helper.py:203`: `await self.assistant.llm_queue.put("नमस्ते")` → LLM generates a full greeting response, then pushes to TTS.
- `asterisk_bridge_helper.py:222`: `await self.assistant.tts_queue.put(greet)` → a hand-crafted greeting is also pushed directly to TTS.
- Two greetings run in parallel/sequence. Whichever loses the race arrives mid-call.
- Fix direction: pick one path. The direct-to-TTS greeting (line 222) is deterministic and works without the LLM, so it is the safer primary. The "नमस्ते" LLM trigger should be removed or made contingent on the user speaking first.

### BUG-4 — Heartbeat frame size ≠ RTP frame size → jitter
- `asterisk_bridge_helper.py:79`: heartbeat silence frame is 640 bytes (40 ms @ 8 kHz SLIN)
- Actual voice RTP frames are 320 bytes (20 ms) — see `FRAME_BYTES = 320` at line 334
- Asterisk/Dinstar expects a consistent ptime. Switching between 40 ms and 20 ms frames mid-call can cause audible jitter and potential PT confusion.
- Comment on line 92 claims "640 bytes / 40ms L16" which documents the mismatch but doesn't justify it.
- Fix direction: make heartbeat 320-byte (20 ms) frames at 50 Hz cadence.

### BUG-5 — Session cleanup does not reliably kill the Asterisk bridge (leaking bridges)
- Live state on the Pi right now: `ari show app ai-call-app` lists **62 leaked stasis bridges** (mostly 0-channel, some oldest at 18+ hours uptime). `bridge show all` confirms.
- Root cause candidates in `asterisk_bridge_helper.py:381-406` `AsteriskCallSession.stop()`:
  - Delete calls are wrapped in bare `except:` which silently swallows failures.
  - `self.bridge_id` is checked with `hasattr` rather than truthiness — if `start()` crashed before creating the bridge, this attribute would not exist; but if it was created and then deletion failed, we never retry.
  - `StasisEnd` for the ExternalMedia UnicastRTP channel is *skipped* at lines 440-460 of the bridge dispatcher (only inbound "real" channels are tracked). So an ExternalMedia channel that dies unexpectedly won't trigger cleanup of the containing bridge.
  - If the Python process is killed (SIGINT — see `backend.log`, KeyboardInterrupt crashed the process), `stop()` never runs and bridges leak outright.
- Immediate reaper: `python web_backend/scratch/cleanup_asterisk.py` deletes channels but NOT bridges. To clear bridges: `for id in $(asterisk -rx 'bridge show all' | awk 'NR>2{print $1}'); do asterisk -rx "bridge destroy $id"; done` on the Pi.
- Fix direction: (a) in `stop()`, log delete failures instead of swallowing them; (b) add a startup reaper in `VaaniAsteriskBridge.__init__` that destroys any pre-existing bridge owned by `ai-call-app` when the backend boots; (c) consider using ARI's `bridge destroy` with channel list first then fall back to DELETE.

### BUG-6 — `/calls/active` returns phone number == channel_id
- `main.py:175`: `return [{"channel_id": cid, "phone": s.channel_id} ...]`. The `phone` field is copied from `channel_id`, not from the actual dialed number. The dialed number is not stored on the session object at all.
- Fix direction: store `self.phone_number` on `AsteriskCallSession.__init__` and plumb it through from the dial endpoint.

### BUG-7 — `/calls/dial` does not associate the dialed number with the created channel
- `main.py:191-243` originates via ARI but does not capture the returned channel ID, nor does it update the history entry with the channel ID. As a result the `/calls/hangup` endpoint can't target "the most recent call" without the user manually fetching `/calls/active`.
- The appArgs `dialed_from_web,{phone},{voice_id}` pass the phone number through to StasisStart, but `VaaniAsteriskBridge.run()` (lines 440-471) only consumes `args[2]` (voice_id) and ignores `args[1]` (phone). So the bridge *receives* the phone number but throws it away.
- Fix direction: in the StasisStart handler, parse `args[1]` and set `session.phone_number = args[1]`.

### BUG-8 — `ENERGY_THRESHOLD = 600` constant is unused
- `engine.py:17` declares `ENERGY_THRESHOLD = 600` but the ASR worker uses `calibrated_threshold` (noise-floor adaptive) everywhere. Spec doc extensively references `ENERGY_THRESHOLD` as if it were authoritative — but it is dead code.
- Not a functional bug, but a source of confusion when tuning VAD.

### BUG-9 — `backend.log` is open-for-write-truncate on every restart
- `start_vaani_web.py:89`: `open("backend.log", "w")` truncates the log each launch. Previous crash context is lost. Recommend `"a"` (append) with rotation or timestamped filenames.

### BUG-10 — `frames_sent * 0.040` AND double busy-wait coupling (perf)
- Even after fixing BUG-2, the pacing loop at `asterisk_bridge_helper.py:355-363` does an `await asyncio.sleep(remaining - 0.002)` *and* a tight `while … < expected_elapsed: pass` busy-wait. The busy-wait pegs one CPU core during TTS playback on Windows. Given that high-priority mode is enabled (`main.py:105`) this can starve other async tasks on the process.
- Fix direction: keep the sleep; drop the busy-wait. Sub-2ms jitter won't be audible over a PSTN line.

---

## 4. Verified Facts About the Pipeline

- Ref audio for default voice Ravi: `assets/voices/ravi_sir.mp3` (NOT `ravi_sir_8s.wav`, despite an 8s version existing on disk). Path is resolved in `engine.py:25`.
- `voices.json` contains 11 entries: `ravi` (protected) + 10 cloned celebrity personas (Amitabh, Alia, Amrish, Kader, Nana, Pankaj, Shraddha, Sunil Shetty, Sunny Deol, Deepika) — note VIP_PERSONAS in `main.py:331-356` only hard-codes 4 (Amitabh, SRK, Salman, Modi). The 4 hardcoded personas do NOT auto-populate the voices list; they only intercept `/voices/generate-persona` calls.
- WebAssistant and AsteriskCallSession share `_models` singleton (`engine.py:111`), so the Whisper + OmniVoice models are loaded once globally.
- Per-call voice clone prompt generation (`asterisk_bridge_helper.py:187-192`) regenerates the prompt on every call — potentially slow for first-time clones, but cached per-process after that.
- `input_sampling_rate=8000` for Asterisk sessions (`asterisk_bridge_helper.py:167`), `24000` for browser WebSocket (`engine.py:114` default).
- Endianness auto-detect (`asterisk_bridge_helper.py:243-263`) compares RMS of LE vs BE int16. Threshold logic: if either interpretation gives `RMS > 0.7` *and* the other gives `RMS < 0.4`, pick the lower one. Default fallback is BE. In practice the Pi's Asterisk sends network byte order (BE), and LE-interpretation RMS tends to be high-garbage ~ 0.8+.

---

## 5. Call-Flow (as-actually-wired)

```
[Dashboard]           POST /calls/dial {phone, voice_id}
   │
   ▼
[FastAPI main.py:191]
   │   checks if bridge.sessions is empty (parallel-call lock)
   │   POSTs to ARI /channels  with:
   │     endpoint = PJSIP/1001/sip:<phone>@192.168.8.60
   │     app      = ai-call-app
   │     appArgs  = "dialed_from_web,<phone>,<voice_id>"
   ▼
[Asterisk] dials out via 1001 → Dinstar → PSTN
   │
   │   when remote answers → StasisStart fires on ARI WebSocket
   ▼
[VaaniAsteriskBridge.run()]  (asterisk_bridge_helper.py:428-485)
   │   skips channels whose name starts with "UnicastRTP" (our own ExternalMedia)
   │   skips channels already tracked as an external_channel_id of another session
   │   pulls args[2] as voice_id  (BUG: args[1]=phone is ignored)
   │   spawns AsteriskCallSession.start()
   ▼
[AsteriskCallSession.start()]  (bridge_helper:102-229)
   1. ARI POST /channels/{id}/answer
   2. asyncio datagram_endpoint → self.udp_port (ephemeral)
   3. ARI POST /channels/externalMedia ?app=ai-call-app&external_host=<winIP>:<port>&format=slin
         proactive peer_addr learn from channelvars.UNICASTRTP_LOCAL_ADDRESS/PORT
   4. ARI POST /bridges (mixing) + /bridges/{id}/addChannel?channel=chan1,ext_chan
   5. asyncio.create_task(_run_heartbeat)          ← 640-byte frames every 40ms (BUG-4)
   6. WebAssistant(event_handler=on_ai_event, input_sampling_rate=8000)
      assistant.reset_session()
   7. if voice_id && voice_id != "ravi": regenerate voice_clone_prompt
   8. asyncio.create_task(assistant.run())
   9. assistant.llm_queue.put("नमस्ते")           ← BUG-3 double greeting path A
  10. (wait up to 3s for peer_addr)
  11. assistant.tts_queue.put(hand-crafted greet)  ← BUG-3 double greeting path B
   ▼
[Ongoing]  Asterisk RTP in  → on_rtp_received → asr_queue (batched 15 frames ≈ 300ms)
           asr_queue → ASR worker → llm_queue → LLM worker (Ollama stream) → tts_queue
           tts_queue → TTS worker → emit("audio_chunk", pcm_24k_bytes)
           on_ai_event("audio_chunk") → resample 24k→8k, 2× gain, pack BE int16, send RTP frames
```

---

## 6. Environment Variables / Runtime Knobs (as set in code)

Set in both `start_vaani_web.py` and `web_backend/main.py` BEFORE any heavy imports:
- `OPENBLAS_MAIN_FREE=1`, `OPENBLAS_NUM_THREADS=1`
- `NUMEXPR_MAX_THREADS=1`, `OMP_NUM_THREADS=1`
- `PYTHONFPEMASK=1`
- `NPY_DISABLE_CPU_FEATURES=AVX512F`
- `KMP_DUPLICATE_LIB_OK=TRUE`
Additionally in launcher only:
- `CUDA_DEVICE_ORDER=PCI_BUS_ID`
- `OLLAMA_MAX_LOADED_MODELS=0`
Runtime process priority:
- Launcher starts backend with `BELOW_NORMAL_PRIORITY_CLASS` (`start_vaani_web.py:103`)
- Backend then self-elevates to `HIGH_PRIORITY_CLASS` via `psutil` (`main.py:104`). These two settings partially conflict — the self-elevation wins after it runs, but between launcher-start and main.py running the process is BELOW_NORMAL.

---

## 7. Key Configuration Constants (from source, verified)

| Name | Value | Where | Notes |
|---|---|---|---|
| `LLM_URL` | `http://127.0.0.1:11434/v1/chat/completions` | engine.py:16 | OpenAI-compat endpoint |
| `LLM_MODEL` | **undefined** | engine.py | ⚠ BUG-1 |
| `ENERGY_THRESHOLD` | 600 | engine.py:17 | ⚠ BUG-8 unused |
| `SILENCE_CHUNKS` | 2 | engine.py:18 | ~600ms |
| `BARGE_IN_FRAMES` | 8 | engine.py:20 | ~160ms |
| `TTS_CHUNK_SAMPLES` | 480 | engine.py:21 | 20ms @ 24kHz |
| `FRAME_BYTES` | 320 | bridge_helper:334 | 20ms @ 8kHz |
| heartbeat frame | 640 bytes | bridge_helper:79 | ⚠ BUG-4 (40ms) |
| RTP outbound PT | 10 | bridge_helper:287 | fixed |
| RTP peer PT default | 11 | bridge_helper:50 | overridden on first inbound packet |
| pacing per frame | 0.040 s | bridge_helper:353 | ⚠ BUG-2 (should be 0.020) |
| ASR model | `large-v3-turbo` | engine.py:91 | cuda:1 float16 |
| TTS model | `k2-fsa/OmniVoice` | engine.py:94 | cuda:0 float16 |
| TTS num_step | 12 | engine.py:425 | |
| LLM num_predict | 128 | engine.py:339 | |
| LLM temperature | 0.5 | engine.py:339 | |
| LLM history kept | last 10 turns | engine.py:335,391 | |
| Barge-in hard-lock (greeting) | 4.0 s | engine.py:192, 457 | |
| Barge-in hard-lock (mid-call) | 2.0 s | engine.py:459 | |
| Echo gate per sentence | 1.2 s | engine.py:209 | |
| Call-start barge-in immunity | 5.0 s | engine.py:204 | |
| End-of-response silence | 0.8 s | engine.py:407 | |

---

## 8. Live System State Snapshot (2026-04-18)

- Asterisk on Pi: up, ARI app `ai-call-app` registered with **62 active subscription bridges** (leaks from crashed sessions), 0 live channels, 0 active calls.
- Backend log tail shows a normal startup sequence ending in a user-initiated KeyboardInterrupt — the last graceful shutdown did not clean up the bridges.
- calls_history.json: last entry 2026-04-18 12:44:20, phone 8757839258, status "dialing" (never updated to "completed"/"failed" — there is no code path that writes a terminal status back). Many historical entries have this same stuck status.
- Frontend: 8 components present (AssistantOrb, CallHistory, Dialer, LogViewer, Sidebar, Starfield, VocalisOrb, VoiceLibrary) + one hook (useVoice).

---

## 9. Open Questions Worth Asking Before Fixing

1. **What is the *symptom* the user currently cares most about?** There are many bugs; the immediate experience ("no response from AI", "voice sounds slow/garbled", "call drops") will point to which one to fix first. BUG-1 (no LLM response) and BUG-2 (50% speed audio) are both severe and both plausible.
2. **Is endpoint 1001 actually registered at Dinstar right now?** `pjsip show endpoint 1001` reports "Not in use" / contact "NonQual" — not a clear "registered/OK". If Dinstar isn't honoring 1001, outbound calls fail before they even reach the PSTN and the backend path is irrelevant.
3. **Can we clear the 62 leaked bridges now or should we wait for a maintenance window?** They don't block new calls (stasis bridges aren't unique per channel) but they eat RAM and look alarming.
4. **Windows Firewall rule for UDP** — spec mentions RTP could be dropped. Should verify that Python.exe (or better, the udp port range the backend grabs) has an inbound-allow rule.
5. **Is the "ongoing call lock"** in `/calls/dial` (main.py:198) too aggressive? When a previous call fails to StasisEnd (e.g., Python was killed), the sessions dict might be empty so the lock releases — but the Asterisk bridge is still sitting there. That's actually OK because it's per-process state.

---

## 10. Suggested Fix Order (for when we start fixing)

1. **BUG-1** — trivial one-line fix unlocks the whole LLM path. Without this, nothing else matters.
2. **BUG-2** — another one-line fix, massive UX impact.
3. **BUG-3** — cleanup: remove the double-greet so the first impression isn't chaotic.
4. **BUG-4** — make heartbeat match RTP frame size.
5. **BUG-5** — bridge cleanup: startup reaper + better error reporting in `stop()`.
6. **BUG-7** — plumb phone number through args[1] so history/active calls make sense.
7. **BUG-10** — drop the CPU-pegging busy-wait.
8. Everything else (BUG-6, BUG-8, BUG-9) is cosmetic / operational hygiene.

---

*Last updated: 2026-04-18. Keep this file current — next agent touching the codebase should verify findings and record new ones here.*
