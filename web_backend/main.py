from __future__ import annotations
import os

# --- Environment Fixes (CRITICAL for Windows/NumPy Stability) ---
os.environ["OPENBLAS_MAIN_FREE"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_MAX_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["PYTHONFPEMASK"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["NPY_DISABLE_CPU_FEATURES"] = "AVX512F"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# Strict cuBLAS determinism — MUST be set before torch is imported,
# otherwise torch.use_deterministic_algorithms(True) raises at call time.
# Required so repeated diffusion sampling of the same text produces the
# same audio (no pitch drift across turns).
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
# -------------------------------------------------------------

import asyncio
import sys
import base64
import time
import psutil  # type: ignore # For High-Priority scheduling
from typing import Optional

# Windows console is cp1252 by default; Devanagari (Hindi) prints would crash.
try:
    # Use getattr for safe access to 'reconfigure' without triggering type-checker
    stdout_reconfig = getattr(sys.stdout, "reconfigure", None)
    if stdout_reconfig:
        stdout_reconfig(encoding="utf-8", errors="replace")
        
    stderr_reconfig = getattr(sys.stderr, "reconfigure", None)
    if stderr_reconfig:
        stderr_reconfig(encoding="utf-8", errors="replace")
except Exception:
    pass

# Add project root and backend dir to sys.path for robust imports
# This ensures that even if run from root, modules like 'engine' are found.
ROOT_DIR = r"e:\Ai\Omini with Astrisk\Omini"
BACKEND_DIR = os.path.join(ROOT_DIR, "web_backend")
for p in [BACKEND_DIR, ROOT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

import json
import re
import websockets
import uuid
import shutil
import audioop
import httpx
import numpy as np
from omnivoice import OmniVoiceGenerationConfig
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
try:
    from web_backend.engine import WebAssistant, _models, _strip_think_streaming
except ImportError:
    from engine import WebAssistant, _models, _strip_think_streaming  # type: ignore

try:
    from web_backend.llm_providers import (
        resolve_provider, ollama_unload_all, ollama_warmup, save_provider_config,
    )
except ImportError:
    from llm_providers import (  # type: ignore
        resolve_provider, ollama_unload_all, ollama_warmup, save_provider_config,
    )

try:
    from web_backend.stt_providers import (
        resolve_stt, build_stt, save_stt_config,
    )
except ImportError:
    from stt_providers import (  # type: ignore
        resolve_stt, build_stt, save_stt_config,
    )

try:
    from web_backend.campaigns import load_campaign, save_campaign, build_system_prompt
except ImportError:
    from campaigns import load_campaign, save_campaign, build_system_prompt  # type: ignore

try:
    from web_backend.audio_utils import trim_audio_file
except ImportError:
    from audio_utils import trim_audio_file  # type: ignore

from datetime import datetime

try:
    from web_backend.asterisk_bridge_helper import VaaniAsteriskBridge
except ImportError:
    from asterisk_bridge_helper import VaaniAsteriskBridge  # type: ignore


ARI_BASE_URL = "http://192.168.8.59:8088/ari"
ARI_AUTH = ("ari_user", "ari_pass")
HISTORY_FILE = os.path.join(ROOT_DIR, "calls_history.json")

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_URL = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01"

from contextlib import asynccontextmanager

# --- Log Management ---
log_clients: set = set()
voice_clients: set = set()  # browser /ws connections — receive phone transcripts

# Pre-synthesized phone audio — set during lifespan startup to avoid cold TTS
# on the first call. None means fall back to live synthesis.
_cached_greeting_ulaw: bytes | None = None
SESSION_LOG = os.path.join(ROOT_DIR, "session_trace.log")
# Non-blocking file sink: broadcast_log pushes lines to this queue and returns
# immediately. A background task (started in lifespan) batches flushes to disk
# so the LLM/TTS hot path never waits on I/O. The queue is bounded so a stalled
# disk doesn't cause unbounded memory growth — it just drops oldest lines.
_log_file_queue: asyncio.Queue | None = None
_LOG_QUEUE_MAX = 5000


async def push_phone_transcript(role: str, text: str, is_partial: bool = False):
    """Push a phone-call transcript line to all connected browser /ws clients."""
    clean = re.sub(r'\[.*?\]', '', text).strip()
    payload = json.dumps({"type": "transcript", "data": {"role": role, "text": clean, "is_partial": is_partial}})
    dead = set()
    for client in list(voice_clients):
        try:
            await client.send_text(payload)
        except Exception:
            dead.add(client)
    for c in dead:
        voice_clients.discard(c)

async def broadcast_log(msg_type, data):
    """Broadcast internal events to any connected /ws/logs clients.
    File persistence is offloaded to _log_file_writer via an unbounded-put queue."""
    # 1. Enqueue for the background file writer (fire-and-forget).
    q = _log_file_queue
    if q is not None:
        try:
            payload = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
            ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            line = f"{ts} [{msg_type}] {payload}\n"
            try:
                q.put_nowait(line)
            except asyncio.QueueFull:
                # Drop the OLDEST line and push the new one — preserves recency.
                try:
                    q.get_nowait()
                    q.put_nowait(line)
                except Exception:
                    pass
        except Exception:
            pass  # never let logging break the pipeline

    # 2. Fan out to any connected /ws/logs browser clients.
    if not log_clients: return
    msg = json.dumps({"type": msg_type, "data": data, "timestamp": time.time()})
    disconnected = set()
    for client in log_clients:
        try:
            await asyncio.wait_for(client.send_text(msg), timeout=0.1)
        except:
            disconnected.add(client)
    for d in disconnected: log_clients.remove(d)


def _append_lines_sync(lines):
    """Synchronous batch append. Called only from the background writer thread."""
    with open(SESSION_LOG, "a", encoding="utf-8") as f:
        f.writelines(lines)


async def _log_file_writer():
    """Background task: drains the log queue and batch-writes to session_trace.log.
    Batches up to 200 lines per flush; flushes whenever a line arrives and the queue empties."""
    global _log_file_queue
    assert _log_file_queue is not None
    batch = []
    while True:
        try:
            # Wait for at least one line.
            line = await _log_file_queue.get()
            batch.append(line)
            # Drain whatever else is already queued (up to 200) without blocking.
            while batch and len(batch) < 200:
                try:
                    batch.append(_log_file_queue.get_nowait())
                except asyncio.QueueEmpty:
                    break
            # Flush the batch off the event loop.
            await asyncio.to_thread(_append_lines_sync, batch)
            batch.clear()
        except asyncio.CancelledError:
            # Final flush on shutdown.
            if batch:
                try:
                    await asyncio.to_thread(_append_lines_sync, batch)
                except Exception:
                    pass
            raise
        except Exception as e:
            print(f"[LogWriter] error: {e}", flush=True)
            batch.clear()
            await asyncio.sleep(0.5)


# ulaw silence byte (encodes ~0 linear).
_ULAW_SILENCE = b"\xff"

def _with_silence_prefix(ulaw: bytes, prefix_ms: int) -> bytes:
    """Prepend N ms of ulaw silence to an audio blob.

    On outbound calls the RPi bridge only unlocks its RTP sender after it
    receives the first inbound packet from Asterisk (that's when remote_addr
    gets learned). Sending the greeting with a silent prefix gives Asterisk
    time to establish the bidirectional RTP path so the real audio isn't
    dropped by the bridge's remote_addr guard.
    """
    if prefix_ms <= 0:
        return ulaw
    # 8kHz × prefix_ms/1000 × 1 byte/sample = 8 * prefix_ms
    return _ULAW_SILENCE * (8 * prefix_ms) + ulaw


def _synth_to_ulaw(text: str) -> bytes:
    """Fast TTS to u-law 8kHz via decimation."""
    
    audio_list = _models.tts_model.generate(
        text=text,
        voice_clone_prompt=_models.voice_prompt,
        language="hi",
        generation_config=OmniVoiceGenerationConfig(
            num_step=12, guidance_scale=2.0, t_shift=0.1,
            audio_chunk_threshold=600.0, postprocess_output=False,
        ),
    )
    if not audio_list:
        return b""
        
    audio_data = audio_list[0]
    if hasattr(audio_data, "cpu"):
        audio_data = audio_data.cpu().numpy()
    audio_data = np.asarray(audio_data, dtype=np.float32)
    
    # Peak-Normalize and Scale to int16
    pcm_24k = (np.clip(audio_data * 1.0, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
    
    # Resample 24kHz -> 8kHz
    pcm_8k, _ = audioop.ratecv(pcm_24k, 2, 1, 24000, 8000, None)
    
    # Boost for telephony
    pcm_8k = audioop.mul(pcm_8k, 2, 1.4)
    
    return audioop.lin2ulaw(pcm_8k, 2)


@asynccontextmanager
async def lifespan(app):
    # 0. Start the non-blocking log file writer BEFORE anything else emits.
    global _log_file_queue
    _log_file_queue = asyncio.Queue(maxsize=_LOG_QUEUE_MAX)
    log_writer_task = asyncio.create_task(_log_file_writer())
    print("[Startup] Log writer task started.")

    # 1. Pre-load AI models (shared between Web and Bridge)
    print("[Startup] Pre-loading AI models...")
    _models.load()

    # Pre-loading models only. Greeting is now live for better dynamic personality.
    print("[Startup] Pre-loading AI models...")
    _models.load()

    # Resolve the active LLM provider once at startup. Local Ollama gets a GPU
    # prewarm; remote APIs (Sarvam) are HTTP-only so we skip all local warmup.
    active_provider = resolve_provider()
    print(f"[Startup] LLM provider: {active_provider.name} model={active_provider.model}")

    if active_provider.uses_local_gpu:
        # Prewarm Ollama with FULL GPU offload (num_gpu=99). Cuts TTFT from
        # ~400 ms (CPU) to ~100 ms (GPU). num_ctx=1024 keeps KV cache small.
        print("[Startup] Pre-warming Ollama on CPU (num_gpu=0, num_ctx=4096)...")
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                await client.post("http://127.0.0.1:11434/api/generate", json={
                    "model": active_provider.model,
                    "prompt": "test",
                    "stream": False,
                    "options": {"num_predict": 1, "num_gpu": 0, "num_ctx": 4096}
                })
            print("[Startup] Ollama pre-warmed on CPU.")
        except Exception as e:
            print(f"[Startup] Ollama pre-warm skipped: {e}")
    else:
        # Active provider is remote (Sarvam). If ollama.exe is still running
        # from a previous session (launcher skipped the taskkill, manual
        # backend restart, etc.), it may be holding models in VRAM that
        # OmniVoice now needs. Evict them proactively — no-op if Ollama isn't
        # running, since the HTTP call just fails silently.
        print(f"[Startup] Active provider is {active_provider.name}; evicting any resident Ollama models...")
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                unloaded = await ollama_unload_all(client)
            if unloaded:
                print(f"[Startup] Freed Ollama VRAM: {unloaded}")
            else:
                print("[Startup] No resident Ollama models to evict (or Ollama not running).")
        except Exception as e:
            print(f"[Startup] Ollama eviction skipped: {e}")

    # [OBS] Emit one-shot device manifest.
    boot_manifest = {
        "llm_provider": active_provider.name,
        "llm_model": active_provider.model,
        "llm_url": active_provider.url,
        "pid": os.getpid(),
        "windows_priority": _priority_class_set
    }
    if active_provider.uses_local_gpu:
        boot_manifest["llm_configured_num_gpu"] = 99
        boot_manifest["llm_configured_num_ctx"] = 4096
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                ps_resp = await client.get("http://127.0.0.1:11434/api/ps")
                ps_data = ps_resp.json() if ps_resp.status_code == 200 else {}
            for m in ps_data.get("models", []) or []:
                if m.get("name", "").startswith(active_provider.model):
                    size_total = int(m.get("size", 0) or 0)
                    size_vram = int(m.get("size_vram", 0) or 0)
                    if size_total > 0 and size_vram >= int(size_total * 0.9):
                        actual = "cuda"
                    elif size_vram == 0:
                        actual = "cpu"
                    else:
                        actual = "mixed"
                    boot_manifest["llm_size_total_bytes"] = size_total
                    boot_manifest["llm_size_vram_bytes"] = size_vram
                    boot_manifest["llm_actual_device"] = actual
                    break
        except Exception as e:
            boot_manifest["llm_probe_error"] = f"{type(e).__name__}: {e}"[:200]
    else:
        boot_manifest["llm_actual_device"] = "remote-api"

    try:
        await broadcast_log("BOOT", json.dumps(boot_manifest, ensure_ascii=False))
    except Exception as e:
        print(f"[Startup] BOOT log emit failed: {e}")

    # Resolve STT provider. Default is sarvam (cloud) — Whisper only loads
    # when explicitly selected. Running a single big GPU model (OmniVoice)
    # instead of two is what keeps the CUDA allocator clean enough for
    # cuDNN to pick consistent conv kernels turn-to-turn, which is what
    # was causing the inter-turn pitch drift before.
    stt_choice = resolve_stt()
    print(f"[Startup] STT provider: {stt_choice.name} model={stt_choice.model}")
    if stt_choice.uses_local_gpu:
        _models.load_whisper(stt_choice.model)
    app.state.stt_choice = stt_choice
    app.state.stt_client = build_stt(stt_choice, whisper_model=_models.asr_model)

    # 2. Start Asterisk Bridge as a background task
    app.state.llm_provider = active_provider
    # Prime the campaign cache once at boot so the very first turn of the day
    # doesn't pay the disk read. Missing/invalid campaigns.json returns {}
    # and `build_system_prompt` falls back to a safe generic assistant prompt.
    app.state.campaign = load_campaign()
    print(f"[Startup] Campaign loaded: label={app.state.campaign.get('label')!r} candidate={app.state.campaign.get('candidate_name')!r}")
    app.state.bridge = VaaniAsteriskBridge(log_handler=broadcast_log, transcript_handler=push_phone_transcript)
    bridge_task = asyncio.create_task(app.state.bridge.run())
    print("[Startup] Asterisk Bridge initialized.")

    print("[Startup] Models ready. Server accepting connections.")
    yield
    
    # 3. Shutdown Cleanup
    print("[Shutdown] Stopping Asterisk Bridge...")
    app.state.bridge.stop()
    bridge_task.cancel()
    log_writer_task.cancel()
    try:
        await log_writer_task
    except (asyncio.CancelledError, Exception):
        pass

# [Phase 20] Force High Priority for Crystal-Clear Voice on Windows
_priority_class_set = "normal"
try:
    p = psutil.Process(os.getpid())
    p.nice(psutil.HIGH_PRIORITY_CLASS)
    _priority_class_set = "high"
    print("[System] Windows High-Priority Mode: ACTIVE")
except Exception as e:
    print(f"[System] Could not set high priority: {e}")

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    return {"status": "ok", "gpu": "RTX 3060", "engine": "Vaani v4.2"}

# --- Voice Library API ---
VOICES_FILE = os.path.join(ROOT_DIR, "assets", "voices", "voices.json")

class VoiceUpdate(BaseModel):
    name: Optional[str] = None
    gender: Optional[str] = None
    style: Optional[str] = None
    age: Optional[str] = None
    about: Optional[str] = None
    catchphrases: Optional[str] = None

class PersonaRequest(BaseModel):
    name: str
    gender: str = "male"

def load_voices():
    if not os.path.exists(VOICES_FILE):
        return []
    with open(VOICES_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_voices(voices):
    with open(VOICES_FILE, "w", encoding="utf-8") as f:
        json.dump(voices, f, indent=2, ensure_ascii=False)

# --- Call History Management ---
class DialRequest(BaseModel):
    phone_number: str
    voice_id: str = "ravi"

def load_call_history():
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return []

def save_call_history(history):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

@app.get("/calls/history")
async def get_call_history():
    return load_call_history()

@app.get("/calls/active")
async def get_active_calls():
    if hasattr(app.state, 'bridge'):
        # Return channel IDs and associated numbers if available in sessions
        return [{"channel_id": cid, "phone": s.channel_id} for cid, s in app.state.bridge.sessions.items()]
    return []

@app.post("/calls/hangup")
async def hangup_call(channel_id: Optional[str] = None):
    print(f"[Dialer] /calls/hangup CALLED channel_id={channel_id}", flush=True)
    hung_up = False

    # Hang up RPi bridge phone call if active
    phone_ws = getattr(app.state, 'active_phone_ws', None)
    if phone_ws is not None:
        try:
            await phone_ws.send_json({"type": "hangup"})
            await phone_ws.close()
        except Exception:
            pass
        app.state.active_phone_ws = None
        hung_up = True
        # Also kill the bridge process on RPi
        try:
            proc = await asyncio.create_subprocess_exec(
                "ssh",
                "-i", r"C:\Users\pc\.ssh\id_ed25519",
                "-o", "BatchMode=yes",
                "-o", "StrictHostKeyChecking=no",
                "pi@192.168.8.59",
                "pkill", "-9", "-f", "vaani_bridge.py",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(proc.wait(), timeout=3.0)
        except Exception:
            pass

    # Also hang up any legacy Windows-bridge sessions
    if hasattr(app.state, 'bridge'):
        if channel_id:
            success = await app.state.bridge.hangup_session(channel_id)
            hung_up = hung_up or success
        else:
            await app.state.bridge.hangup_all()
            hung_up = True

    # Mark any "dialing" history rows as ended
    history = load_call_history()
    changed = False
    for entry in history:
        if entry.get("status") == "dialing":
            entry["status"] = "ended"
            changed = True
    if changed:
        save_call_history(history)

    return {"status": "success" if hung_up else "not_found"}

@app.post("/calls/dial")
async def dial_number(req: DialRequest):
    phone = req.phone_number.strip()
    if not phone:
        raise HTTPException(status_code=400, detail="Phone number required")

    # Normalize: keep only digits, strip any '91' country-code prefix because
    # the Dinstar 8-port gateway expects bare 10-digit GSM MSISDN (the Ubuntu
    # prod system does the same — basic-context dialplan `Set(PHONE=${EXTEN:0:10})`).
    digits = "".join(ch for ch in phone if ch.isdigit())
    if len(digits) == 12 and digits.startswith("91"):
        digits = digits[2:]
    phone = digits

    # --- PROACTIVE LOCKING: Prevent Double/Parallel Calling ---
    if getattr(app.state, 'active_phone_ws', None) is not None:
        print("[Dialer] BLOCKING: A call is already in progress.")
        raise HTTPException(status_code=400, detail="A call is already in progress. End current call first.")

    # Openvox 32-port gateway at 192.168.8.60: ports 17 & 18 have live SIMs.
    # TEMP: swapped from 1018 → 1017 to isolate whether 480 Temporarily
    # Unavailable is an 8076536278-reachability issue or a 1018 SIM issue.
    sim_port = "1017"
    print(f"[Dialer] SSH-triggering vaani_bridge.py for {phone} via PJSIP/{sim_port}")

    # Trigger vaani_bridge.py on the RPi via SSH.
    # Independent copy at /home/pi/Documents/vaani_asterisk/ — not shared with Ubuntu prod.
    # It originates PJSIP/{phone}@{sim_port}, handles all ARI/RTP locally, and
    # connects back to our /ws/phone for the AI pipeline.
    ssh_key = r"C:\Users\pc\.ssh\id_ed25519"
    bridge_cmd = (
        "pkill -9 -f vaani_bridge.py 2>/dev/null; "
        "for i in 1 2 3 4 5 6 7 8 9 10; do "
        "  pgrep -f vaani_bridge.py >/dev/null || break; sleep 0.2; "
        "done; "
        "nohup /home/pi/Documents/vaani_asterisk/run_vaani_bridge.sh "
        f"--mode outbound --endpoint PJSIP/{phone}@{sim_port} "
        f"--server \"ws://192.168.8.2:8000/ws/phone?voice_id={req.voice_id}\" "
        "> /tmp/vaani_bridge.log 2>&1 &"
    )
    try:
        proc = await asyncio.create_subprocess_exec(
            "ssh",
            "-i", ssh_key,
            "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=no",
            "pi@192.168.8.59",
            "bash", "-c", bridge_cmd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await asyncio.wait_for(proc.wait(), timeout=6.0)
        status = "dialing"
        print(f"[Dialer] vaani_bridge.py launched for {phone}", flush=True)
    except Exception as e:
        print(f"[Dialer] SSH Error: {e}", flush=True)
        status = "error"

    # 2. Record in History — sweep stale "dialing" entries first (only one call can
    # be active at a time, so anything older than this new one is by definition over).
    history = load_call_history()
    for entry in history:
        if entry.get("status") == "dialing":
            entry["status"] = "ended"
    new_call = {
        "id": str(uuid.uuid4())[:8],
        "phone": phone,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": status,
        "type": "outbound"
    }
    history.insert(0, new_call)
    save_call_history(history[:50]) # Keep last 50
    
    return new_call

@app.get("/voices")
async def get_voices():
    return load_voices()

@app.get("/models/llm")
async def get_llm_models():
    """Returns a list of available LLM models for the settings UI.

    The list is provider-dependent: local Ollama exposes the GGUF models we've
    verified, remote Sarvam exposes the Indic tiers from their API docs.
    """
    # Prefer the live, possibly-switched provider over the env-resolved one so
    # the UI reflects post-/provider/switch state instead of boot-time state.
    provider = getattr(app.state, "llm_provider", None) or resolve_provider()
    if provider.name == "sarvam":
        return [
            {"id": "sarvam-m",    "name": "Sarvam M (24B) - Average"},
            {"id": "sarvam-30b",  "name": "Sarvam 30B - High IQ"},
            {"id": "sarvam-105b", "name": "Sarvam 105B - Flagship"},
        ]
    return [
        {"id": "gemma3:4b", "name": "Gemma 3 (4B) - Default"},
        {"id": "qwen2:7b", "name": "Qwen 2 (7B) - High IQ"},
        {"id": "gemma4:e4b", "name": "Gemma 4 (Large)"},
        {"id": "mashriram/sarvam-m:latest", "name": "Sarvam (M-Hinglish)"}
    ]

def _read_nvidia_smi():
    """Snapshot all visible NVIDIA GPUs. Returns [] if nvidia-smi isn't on PATH
    or the call fails — the frontend handles an empty list gracefully."""
    import subprocess
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2
        )
    except Exception:
        return []
    gpus = []
    for line in (out.stdout or "").strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue
        def _num(v, cast=int, default=0):
            try:
                return cast(v)
            except Exception:
                return default
        gpus.append({
            "index":        _num(parts[0]),
            "name":         parts[1],
            "util_pct":     _num(parts[2]),
            "mem_used_mb":  _num(parts[3]),
            "mem_total_mb": _num(parts[4]),
            "temp_c":       _num(parts[5]),
            "power_w":      _num(parts[6], cast=float, default=0.0) if len(parts) > 6 else 0.0,
            "power_limit_w":_num(parts[7], cast=float, default=0.0) if len(parts) > 7 else 0.0,
        })
    return gpus


async def _ollama_loaded_models():
    """Ask Ollama which models are currently resident in GPU memory. Best-effort —
    timeouts and non-200s just yield []."""
    try:
        async with httpx.AsyncClient(timeout=1.5) as client:
            r = await client.get("http://127.0.0.1:11434/api/ps")
            if r.status_code == 200:
                return [
                    {
                        "name": m.get("name"),
                        "size_mb": int(m.get("size", 0) // (1024 * 1024)),
                        "vram_mb": int(m.get("size_vram", 0) // (1024 * 1024)),
                    }
                    for m in r.json().get("models", [])
                ]
    except Exception:
        pass
    return []


@app.get("/system/status")
async def system_status():
    """Live snapshot of GPUs + loaded models for the frontend monitor panel.
    Polled by the UI at ~1.5s cadence; keep the work light."""
    provider = getattr(app.state, "llm_provider", None)
    gpus = _read_nvidia_smi()
    ollama_models = await _ollama_loaded_models() if (provider and getattr(provider, "name", "") == "ollama") else []
    
    # System Metrics
    cpu_percent = psutil.cpu_percent()
    ram = psutil.virtual_memory()
    ram_used = round(ram.used / (1024**3), 2)
    ram_total = round(ram.total / (1024**3), 2)

    # Pin assignments are defined by the launcher: STT on cuda:1 (GTX 1650),
    # TTS + LLM on cuda:0 (RTX 3060). If someone reshuffles CUDA_VISIBLE_DEVICES,
    # this will drift, but it mirrors start_vaani_web.py's current pinning.
    stt_device = "cuda:0"  # WhisperModel(..., device_index=0) in engine.py
    tts_device = "cuda:0"  # OmniVoice loads on cuda:0
    llm_device = "cuda:0" if (provider and getattr(provider, "uses_local_gpu", False)) else "remote"

    models = {
        "stt": {
            "name": "faster-whisper large-v3-turbo",
            "device": stt_device,
            "loaded": bool(getattr(_models, "asr_model", None)),
            "role": "Speech-to-Text",
        },
        "tts": {
            "name": "OmniVoice",
            "device": tts_device,
            "loaded": bool(getattr(_models, "tts_model", None)),
            "role": "Text-to-Speech",
        },
        "llm": {
            "name": (provider.model if provider else "unknown"),
            "provider": (getattr(provider, "name", "unknown") if provider else "unknown"),
            "device": llm_device,
            "url": (provider.url if provider else ""),
            "loaded": True if (provider and not getattr(provider, "uses_local_gpu", False)) else bool(ollama_models),
            "role": "Language Model",
            "resident_models": ollama_models,
        },
    }

    active_calls = 0
    try:
        if hasattr(app.state, "bridge"):
            active_calls = len(app.state.bridge.sessions)
    except Exception:
        pass

    return {
        "ts": time.time(),
        "cpu_percent": cpu_percent,
        "ram_used_gb": ram_used,
        "ram_total_gb": ram_total,
        "gpus": gpus,
        "models": models,
        "active_calls": active_calls,
    }


@app.post("/ollama/unload")
async def force_unload_ollama():
    """Manual escape hatch: evict every known Ollama model from GPU regardless
    of the currently-active provider. The System Monitor panel calls this when
    the user explicitly wants VRAM back (e.g. Sarvam is active but Ollama
    didn't get freed during the switch)."""
    async with httpx.AsyncClient() as client:
        unloaded = await ollama_unload_all(client)
    print(f"[Ollama] Force-unload requested: {unloaded}", flush=True)
    return {"ok": True, "unloaded": unloaded}


class ProviderSwitchBody(BaseModel):
    provider: str            # "ollama" | "sarvam"
    model: str | None = None


@app.post("/provider/switch")
async def switch_provider(body: ProviderSwitchBody, request: Request):
    """Hot-swap the active LLM backend.

    Contract:
      - Switching AWAY from ollama: evict all known Ollama tags from GPU via
        keep_alive=0 so VRAM comes back to OmniVoice.
      - Switching TO ollama: resolve first, then warm the chosen model so the
        first real turn isn't cold.
      - If the target provider matches what's already active, this is a no-op:
        prevents the "accidentally clicked my own provider" UX from wiping the
        persisted model (was defaulting to gemma3:4b on any bare Ollama POST).
      - On failure (e.g. Sarvam key missing), the active provider is left
        untouched — the UI shouldn't end up pointing at a broken backend.
    """
    new_name = (body.provider or "").lower().strip()
    if new_name not in ("ollama", "sarvam", "minimax"):
        raise HTTPException(status_code=400, detail=f"unknown provider {new_name!r}")

    prev = getattr(app.state, "llm_provider", None)
    prev_name = getattr(prev, "name", None)
    prev_model = getattr(prev, "model", None)

    # Diagnostic: who triggered this? Unexpected flips have been observed; log
    # UA + referer so we can pin it to a component if it recurs.
    ua = request.headers.get("user-agent", "?")[:100]
    ref = request.headers.get("referer", "-")[:80]
    origin = request.headers.get("origin", "-")[:80]
    print(f"[Provider] /switch request prev={prev_name}/{prev_model} "
          f"target={new_name}/{body.model} origin={origin} referer={ref} ua={ua}",
          flush=True)

    # No-op guard: same provider + same model (or unspecified model) means the
    # caller didn't actually want to change anything. Just return the current
    # state so a stray click doesn't trigger an Ollama warmup + VRAM reload.
    if new_name == prev_name and (body.model is None or body.model == prev_model):
        print(f"[Provider] /switch no-op (already {prev_name}/{prev_model})", flush=True)
        return {
            "ok": True,
            "provider": prev_name,
            "model": prev_model,
            "unloaded": [],
            "warmed": None,
            "noop": True,
        }

    # Resolve target first so a bad config (missing SARVAM_API_KEY, etc.)
    # fails fast without touching GPU state.
    try:
        new_provider = resolve_provider(override_name=new_name, override_model=body.model)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    unloaded: list[str] = []
    warmed: bool | None = None
    async with httpx.AsyncClient() as client:
        if prev_name == "ollama" and new_name != "ollama":
            unloaded = await ollama_unload_all(client)
            print(f"[Provider] Unloaded Ollama models: {unloaded}", flush=True)
        app.state.llm_provider = new_provider
        print(f"[Provider] Switched {prev_name} -> {new_name} (model={new_provider.model})", flush=True)
        # Persist so the next boot starts here without the user re-switching
        # and, crucially, so the launcher skips Ollama startup when Sarvam is
        # the active backend (saves ~3GB of VRAM for OmniVoice).
        if save_provider_config(new_provider.name, new_provider.model):
            print(f"[Provider] Persisted choice -> provider_config.json", flush=True)
        if new_name == "ollama":
            # Warm on the way in so the next voice turn doesn't pay cold-load.
            warmed = await ollama_warmup(client, new_provider.model)
            print(f"[Provider] Ollama warmup ({new_provider.model}) -> {warmed}", flush=True)

    return {
        "ok": True,
        "provider": new_provider.name,
        "model": new_provider.model,
        "unloaded": unloaded,
        "warmed": warmed,
    }


@app.get("/stt/status")
async def get_stt_status():
    """Report the active STT backend. Used by the UI to show which one is
    driving transcription and to render the toggle state."""
    choice = getattr(app.state, "stt_choice", None)
    whisper_loaded = getattr(_models, "asr_model", None) is not None
    return {
        "provider": getattr(choice, "name", None),
        "model": getattr(choice, "model", None),
        "whisper_loaded": whisper_loaded,
    }


class STTSwitchBody(BaseModel):
    provider: str                 # "sarvam" | "whisper"
    model: Optional[str] = None


@app.post("/stt/switch")
async def switch_stt(body: STTSwitchBody, request: Request):
    """Switch the STT backend at runtime.

    - sarvam → whisper: lazy-loads faster-whisper onto cuda:0 on first use.
    - whisper → sarvam: frees the Whisper GPU weights so OmniVoice gets the
      full, clean allocator back. This is what the pitch-drift fix hinged on.
    - Same-provider no-op short-circuits so a stray click doesn't thrash
      VRAM.
    """
    new_name = (body.provider or "").lower().strip()
    if new_name not in ("sarvam", "whisper"):
        raise HTTPException(status_code=400, detail=f"unknown STT provider {new_name!r}")

    prev = getattr(app.state, "stt_choice", None)
    prev_name = getattr(prev, "name", None)
    prev_model = getattr(prev, "model", None)

    ua = request.headers.get("user-agent", "?")[:100]
    print(f"[STT] /switch request prev={prev_name}/{prev_model} "
          f"target={new_name}/{body.model} ua={ua}", flush=True)

    if new_name == prev_name and (body.model is None or body.model == prev_model):
        print(f"[STT] /switch no-op (already {prev_name}/{prev_model})", flush=True)
        return {"ok": True, "provider": prev_name, "model": prev_model, "noop": True}

    try:
        new_choice = resolve_stt(override_name=new_name, override_model=body.model)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Close the previous client cleanly (Sarvam holds an httpx.AsyncClient).
    prev_client = getattr(app.state, "stt_client", None)

    if new_choice.uses_local_gpu:
        # Load Whisper now so the first utterance isn't a cold-load gamble.
        _models.load_whisper(new_choice.model)
    else:
        # Switching to Sarvam — free Whisper's VRAM so OmniVoice runs on a
        # clean allocator again.
        _models.unload_whisper()

    new_client = build_stt(new_choice, whisper_model=_models.asr_model)
    app.state.stt_choice = new_choice
    app.state.stt_client = new_client

    if prev_client is not None and prev_client is not new_client:
        try:
            await prev_client.aclose()
        except Exception:
            pass

    if save_stt_config(new_choice.name, new_choice.model):
        print(f"[STT] Persisted choice -> stt_config.json", flush=True)

    return {"ok": True, "provider": new_choice.name, "model": new_choice.model}


@app.get("/campaigns/default-prompt")
async def get_default_prompt():
    """Return the baseline political system prompt text."""
    from web_backend.campaigns import DEFAULT_POLITICAL_PROMPT
    return {"prompt": DEFAULT_POLITICAL_PROMPT}

@app.get("/campaigns")
async def get_all_campaigns():
    """Return all campaigns from campaigns.json and the active name."""
    try:
        from web_backend.campaigns import DEFAULT_PATH
        with open(DEFAULT_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as e:
        return {"active": "default", "campaigns": {}}

@app.post("/campaigns/{name}/switch")
async def switch_campaign(name: str):
    """Switch the active campaign to the one named {name}."""
    try:
        from web_backend.campaigns import DEFAULT_PATH, load_campaign
        with open(DEFAULT_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if name not in data.get("campaigns", {}):
            raise HTTPException(status_code=404, detail=f"campaign {name!r} not found")
        data["active"] = name
        with open(DEFAULT_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        # Refresh cache
        app.state.campaign = load_campaign(name=name)
        print(f"[Campaign] switched to {name!r}", flush=True)
        return {"ok": True, "active": name, "campaign": app.state.campaign}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/campaign")
async def get_campaign():
    """Return the currently active campaign config. Cached on app.state so
    repeated GETs don't hit disk; the cache is primed at startup and refreshed
    on PUT."""
    cached = getattr(app.state, "campaign", None)
    if cached is None:
        cached = load_campaign()
        app.state.campaign = cached
    return cached or {}


@app.put("/campaign")
async def put_campaign(body: dict, name: str = "default"):
    """Replace a campaign config. Persists to campaigns.json and
    updates the in-memory cache if it's the active one."""
    if not isinstance(body, dict) or not body:
        raise HTTPException(status_code=400, detail="campaign body must be a non-empty object")
    try:
        save_campaign(body, name=name)
        
        # If we updated the active one, refresh cache
        from web_backend.campaigns import DEFAULT_PATH
        with open(DEFAULT_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("active") == name:
            app.state.campaign = body
            
        print(f"[Campaign] saved: name={name!r} label={body.get('label')!r}", flush=True)
        return {"ok": True, "campaign": body}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"save failed: {e}")


@app.post("/debug/ws_close")
async def debug_ws_close(payload: dict):
    """Browser posts the WebSocket close event here so we can log it in backend.log.

    Lets us see the close-code from the browser's perspective, which the server
    doesn't otherwise know (it just sees "websocket.disconnect").
    """
    code = payload.get("code")
    reason = payload.get("reason")
    was_clean = payload.get("wasClean")
    ts = payload.get("ts")
    print(f"[FE-WS-CLOSE] code={code} reason={reason!r} wasClean={was_clean} ts={ts}")
    return {"ok": True}

@app.post("/voices/upload")
async def upload_voice(
    name: str = Form(...),
    gender: str = Form("male"),
    style: str = Form("natural"),
    age: str = Form(""),
    about: str = Form(""),
    catchphrases: str = Form(""),
    file: UploadFile = File(...)
):
    voice_id = str(uuid.uuid4())[:8]
    filename = f"{voice_id}_{file.filename}"
    file_path = os.path.join("assets", "voices", filename)
    abs_path = os.path.join(ROOT_DIR, file_path)
    
    with open(abs_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # --- Auto-Trim (OPTIMIZATION) ---
    # Clone prompts work best at 10s. Trimming here ensures high-speed, stable cloning.
    final_path = trim_audio_file(abs_path, 10)
    # Store the relative path for persistence
    relative_final_path = os.path.relpath(final_path, ROOT_DIR)

    voices = load_voices()
    new_voice = {
        "id": voice_id,
        "name": name,
        "gender": gender,
        "style": style,
        "age": age,
        "about": about,
        "catchphrases": catchphrases,
        "file_path": relative_final_path
    }
    voices.append(new_voice)
    save_voices(voices)
    return new_voice

@app.delete("/voices/{voice_id}")
async def delete_voice(voice_id: str):
    voices = load_voices()
    voice = next((v for v in voices if v["id"] == voice_id), None)
    if not voice:
        return {"error": "Voice not found"}
    
    # Don't delete Ravi
    if voice_id == "ravi":
        return {"error": "Cannot delete default voice"}
        
    voices = [v for v in voices if v["id"] != voice_id]
    save_voices(voices)
    
    abs_path = os.path.join(ROOT_DIR, voice["file_path"])
    if os.path.exists(abs_path):
        os.remove(abs_path)
        
    return {"status": "success"}

@app.put("/voices/{voice_id}")
async def update_voice(voice_id: str, update: VoiceUpdate):
    voices = load_voices()
    voice = next((v for v in voices if v["id"] == voice_id), None)
    
    if not voice:
        raise HTTPException(status_code=404, detail="Voice not found")
        
    if voice_id == "ravi":
        raise HTTPException(status_code=400, detail="Cannot edit the default voice")
    
    # Update only provided fields
    if update.name is not None: voice["name"] = update.name
    if update.gender is not None: voice["gender"] = update.gender
    if update.style is not None: voice["style"] = update.style
    if update.age is not None: voice["age"] = update.age
    if update.about is not None: voice["about"] = update.about
    if update.catchphrases is not None: voice["catchphrases"] = update.catchphrases
    
    save_voices(voices)
    return voice

# --- VIP Persona Database (Zero-Hallucination & Research Backed) ---
VIP_PERSONAS = {
    "amitabh bachchan": {
        "age": "81",
        "gender": "male",
        "about": "अमिताभ बच्चन भारतीय फिल्म जगत के सबसे महान अभिनेता हैं। उन्हें 'शहंशाह' और 'सदी के महानायक' के रूप में जाना जाता है। उन्होंने अपनी आवाज़ और अभिनय से दुनिया भर में करोड़ों प्रशंसकों का दिल जीता है।",
        "catchphrases": "रिश्ते में तो हम तुम्हारे बाप लगते हैं, नाम है शहंशाह।\n\nआज मेरे पास बंगला है, गाड़ी है, बैंक बैलेंस है... तुम्हारे पास क्या है?\n\nपरंपरा, प्रतिष्ठा, अनुशासन... ये इस गुरुकुल के तीन स्तंभ हैं।\n\nविजय दीनानाथ चौहान! पूरा नाम, बाप का नाम दीनानाथ चौहान, गाँव मांडवा。"
    },
    "shah rukh khan": {
        "age": "58",
        "gender": "male",
        "about": "शाहरुख खान, जिन्हें 'किंग खान' और 'बादशाह' कहा जाता है, हिंदी फिल्मों के सबसे लोकप्रिय अभिनेता हैं। वे अपनी रोमांटिक भूमिकाओं और आकर्षण के लिए पूरी दुनिया में प्रसिद्ध हैं।",
        "catchphrases": "हारकर जीतने वाले को बाजीगर कहते हैं।\n\nबड़े-बड़े देशों में ऐसी छोटी-छोटी बातें होती रहती हैं, सेनोरिटा।\n\nनाम तो सुना होगा?\n\nपिक्चर अभी बाकी है मेरे दोस्त!"
    },
    "salman khan": {
        "age": "58",
        "gender": "male",
        "about": "सलमान खान बॉलीवुड के 'दबंग' और 'भाईजान' हैं। वे अपने एक्शन, बॉडीबिल्डिंग और ब्लॉकबस्टर फिल्मों के लिए जाने जाते हैं। वे भारत के सबसे बड़े सितारों में से एक हैं।",
        "catchphrases": "एक बार जो मैंने कमिटमेंट कर दी, फिर तो मैं अपने आप की भी नहीं सुनता।\n\nमुझ पर एक अहसान करना कि मुझ पर कोई अहसान मत करना।\n\nस्वागत नहीं करोगे हमारा?\n\nअभी हम यहाँ के नए कोतवाल हैं!"
    },
    "narendra modi": {
        "age": "73",
        "gender": "male",
        "about": "नरेन्द्र मोदी भारत के 14वें प्रधानमंत्री हैं। वे अपने ओजस्वी भाषणों, विकास कार्यों और वैश्विक मंच पर भारत के सशक्त नेतृत्व के लिए जाने जाते हैं।",
        "catchphrases": "भाइयों और बहनों!\n\nमित्रों, मैं आपको विश्वास दिलाता हूँ... अच्छे दिन आने वाले हैं।\n\nसबका साथ, सबका विकास, सबका विश्वास और सबका प्रयास।\n\nआत्मनिर्भर भारत ही हमारा संकल्प है।"
    },
    "alia bhatt": {
        "age": "31",
        "gender": "female",
        "about": "आलिया भट्ट बॉलीवुड की एक बेहद प्रतिभाशाली और बहुमुखी अभिनेत्री हैं। उन्होंने 'हाईवे', 'राज़ी' और 'गंगूबाई काठियावाड़ी' जैसी फिल्मों में अपने दमदार अभिनय के लिए कई राष्ट्रीय पुरस्कार जीते हैं।",
        "catchphrases": "शिवा! क्या हो रहा है शिवा?\n\nमैं तो पैदा ही हॉट हुई थी!\n\nधप्पा!"
    },
    "amrish puri": {
        "age": "72",
        "gender": "male",
        "about": "अमरीश पुरी भारतीय सिनेमा के सबसे प्रतिष्ठित खलनायकों में से एक थे। उनकी गहरी, बुलंद आवाज़ और 'मोगैम्बो' जैसे किरदारों ने उन्हें अमर बना दिया।",
        "catchphrases": "मोगैम्बो खुश हुआ!\n\nजा सिमरन जा, जी ले अपनी ज़िन्दगी।\n\nइतने टुकड़े करूँगा कि पहचाना नहीं जाएगा।"
    },
    "kader khan": {
        "age": "79",
        "gender": "male",
        "about": "कादर खान एक प्रख्यात अभिनेता, पटकथा लेखक और संवाद लेखक थे। उन्होंने 300 से अधिक फिल्मों में अभिनय किया और बेमिसाल संवाद लिखे।",
        "catchphrases": "क्या बात है, क्या बात है, क्या बात है!\n\nये क्या हो रहा है भाई? ये दुनिया है, यहाँ सब चलता है।"
    },
    "pankaj tripathi": {
        "age": "47",
        "gender": "male",
        "about": "पंकज त्रिपाठी अपनी सहज और प्राकृतिक अभिनय शैली के लिए जाने जाते हैं। 'मिर्जापुर' के कालीन भैया के रूप में उन्होंने वैश्विक पहचान बनाई है।",
        "catchphrases": "बैठिए, समझाते हैं।\n\nई राजनीति है, यहाँ कोई किसी का नहीं होता।\n\nजीवन में थोड़ा बहुत रिस्क तो लेना ही पड़ता है।"
    },
    "suniel shetty": {
        "age": "62",
        "gender": "male",
        "about": "सुनील शेट्टी बॉलीवुड के एक प्रमुख एक्शन स्टार और सफल उद्यमी हैं। उन्हें 'अन्ना' के नाम से भी जाना जाता है। 'धड़कन' और 'बॉर्डर' जैसी फिल्मों में उन्हें काफी सराहा गया।",
        "catchphrases": "अंजलि! मैं तुम्हें भूल जाऊं ये हो नहीं सकता... और तुम मुझे भूल जाओ ये मैं होने नहीं दूंगा।\n\nये धरती मेरी माँ है अन्ना!"
    },
    "sunny deol": {
        "age": "66",
        "gender": "male",
        "about": "सनी देओल अपने 'ढाई किलो के हाथ' और दमदार डायलॉग डिलीवरी के लिए मशहूर हैं। 'घायल' और 'गदर' जैसी फिल्मों ने उन्हें लीजेंड बना दिया है।",
        "catchphrases": "तारीख पे तारीख, तारीख पे तारीख!\n\nये ढाई किलो का हाथ जब किसी पे पड़ता है न, तो आदमी उठता नहीं, उठ जाता है।\n\nहिंदुस्तान ज़िंदाबाद था, ज़िंदाबाद है, और ज़िंदाबाद रहेगा!"
    },
    "deepika padukone": {
        "age": "38",
        "gender": "female",
        "about": "दीपिका पादुकोण वर्तमान समय की सबसे प्रभावशाली और सफल अभिनेत्रियों में से एक हैं। 'ओम शांति ओम' से शुरुआत कर उन्होंने कई ब्लॉकबस्टर फिल्में दी हैं।",
        "catchphrases": "एक चुटकी सिंदूर की कीमत तुम क्या जानो रमेश बाबू?\n\nये जवानी है दीवानी!\n\nकिस्मत बड़ी कुत्ती चीज़ है।"
    },
    "samay raina": {
        "age": "26",
        "gender": "male",
        "about": "समय रैना एक लोकप्रिय स्टैंड-अप कॉमेडियन, यूट्यूब स्ट्रीमर और 'कॉमिकस्तान 2' के विजेता हैं। वे अपने शतरंज स्ट्रीम्स और बेबाक कॉमेडी के लिए मशहूर हैं।",
        "catchphrases": "ओह भाई साहब! मस्ती नहीं रुकनी चाहिए।\n\nचेक एंड मेट!\n\nक्या कर रहा है यार तू?"
    },
    "kcr": {
        "age": "70",
        "gender": "male",
        "about": "के. चंद्रशेखर राव (KCR) तेलंगाना राष्ट्र समिति (BRS) के संस्थापक और तेलंगाना के प्रथम मुख्यमंत्री हैं। वे अपनी प्रभावशाली वाकपटुता के लिए जाने जाते हैं।",
        "catchphrases": "जय तेलंगाना!\n\nएवनी बातें नहीं, काम होना चाहिए।\n\nतेलंगाना वचिनदी, अभिवृद्धि जुरिगिनदी।"
    }
}

@app.post("/voices/generate-persona")
async def generate_persona(req: PersonaRequest):
    # Normalize name for VIP lookup
    search_name = req.name.lower().strip()
    # Handle common typos/variations
    search_name = search_name.replace("bachan", "bachchan")
    search_name = search_name.replace("bachan ", "bachchan")
    
    if search_name in VIP_PERSONAS:
        return VIP_PERSONAS[search_name]

    # Fallback to LLM for unknown names (Using MiniMax-M2.7-HighSpeed)
    prompt = (
        f"Generate a celebrity persona for {req.name}. Output ONLY raw JSON matching this schema:\n"
        '{"age": "actual or estimated age", '
        '"about": "3 sentences in Devanagari Hindi about their professional achievements and style. Use formal language.", '
        '"dialogues": ["4 most iconic signature cinematic dialogues or catchphrases in Devanagari Hindi"]}\n\n'
        f"Generate for: {req.name}"
    )
    
    minimax_key = os.environ.get("MINIMAX_API_KEY", "").strip()
    if not minimax_key:
        # Emergency fallback to local gemma if key is missing
        llm_url = "http://127.0.0.1:11434/api/generate"
        llm_payload = {
            "model": "gemma3:4b",
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": 1024, "temperature": 0.2}
        }
    else:
        llm_url = "https://api.minimax.io/v1/chat/completions"
        llm_payload = {
            "model": "MiniMax-M2.7-HighSpeed",
            "messages": [{"role": "system", "content": "You are a helpful assistant that outputs only valid JSON."}, {"role": "user", "content": prompt}],
            "stream": False,
            "max_tokens": 1024,
            "temperature": 0.2
        }
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            headers = {"Authorization": f"Bearer {minimax_key}"} if minimax_key else None
            resp = await client.post(llm_url, json=llm_payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            
            if minimax_key:
                response_text = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            else:
                response_text = data.get("response", "").strip()
            
            # Clean up markdown code blocks if present
            if response_text.startswith("```json"):
                response_text = response_text.split("```json")[-1].split("```")[0].strip()
            elif response_text.startswith("```"):
                response_text = response_text.split("```")[-1].split("```")[0].strip()
            
            try:
                persona_data = json.loads(response_text)
            except Exception as parse_e:
                print(f"JSON Parse fallback triggered! Raw response: {response_text}")
                persona_data = {"age": "40", "about": "Failed to parse API output.", "dialogues": response_text.replace('"', '')}
                
            dialogues_raw = persona_data.get("dialogues", "")
            if isinstance(dialogues_raw, list):
                dialogues_fmt = "\n\n".join([str(d).strip() for d in dialogues_raw])
            else:
                dialogues_fmt = str(dialogues_raw)

            return {
                "age": str(persona_data.get("age", "30")),
                "about": str(persona_data.get("about", "")),
                "catchphrases": dialogues_fmt
            }
    except Exception as e:
        print(f"Error generating persona: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate persona from LLM")
    except Exception as e:
        print(f"Error generating persona: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate persona from LLM")

@app.websocket("/ws/phone")
async def phone_websocket(websocket: WebSocket, voice_id: str = "ravi"):
    """AI server endpoint for vaani_bridge.py connections.

    Protocol (from bridge → us):
      binary bytes  — ulaw 8kHz audio (only during speech, VAD-bracketed)
      {"type":"call_start"}  — call connected
      {"type":"speech_start"} — VAD triggered start of utterance
      {"type":"speech_end"}   — VAD end; process accumulated audio
      {"type":"call_end"}     — call finished by bridge

    Protocol (us → bridge):
      binary bytes  — ulaw 8kHz TTS response audio
      {"type":"hangup"} — request bridge to hang up the call
    """
    await websocket.accept()
    print(f"[Phone] RPi bridge connected with voice={voice_id}", flush=True)
    app.state.active_phone_ws = websocket

    import numpy as np
    from omnivoice import OmniVoiceGenerationConfig
    import torch

    voices = load_voices()
    voice_meta = next((v for v in voices if v["id"] == voice_id), None)
    if not voice_meta:
        voice_meta = {"name": "Vaani", "gender": "male", "id": "ravi"}
    
    # Load dynamic voice prompt
    if voice_id != "ravi":
        full_audio_path = os.path.join(ROOT_DIR, voice_meta.get("file_path", ""))
        if os.path.exists(full_audio_path):
            current_voice_prompt = await asyncio.to_thread(
                _models.tts_model.create_voice_clone_prompt,
                ref_audio=full_audio_path,
                preprocess_prompt=False
            )
        else:
            current_voice_prompt = _models.voice_prompt
    else:
        current_voice_prompt = _models.voice_prompt

    stt_buffer = bytearray()
    stt_resample_state = None
    call_history: list = []
    processing = False
    current_utterance_task: asyncio.Task | None = None
    # TTS sub-tasks (synth + player) registered by process_utterance so _barge_in
    # can hard-cancel them. Without this the player keeps draining queued audio
    # for several seconds even after the user has interrupted.
    current_tts_subtasks: list[asyncio.Task] = []
    # Endpointing debounce: speech_end schedules processing 500ms in the future.
    # If a new speech_start fires within that window, cancel and keep accumulating
    # — this prevents chopping a sentence with brief mid-pauses into many turns.
    pending_process_task: asyncio.Task | None = None
    ENDPOINT_DEBOUNCE_S = 0.4  # [Refinement] Reduced from 0.5 for snappier response
    MAX_BUFFER_BYTES = 32000 * 30  # 30s ceiling; older audio dropped if exceeded
    # Barge-in state: ai_speaking flips True inside tts_worker while audio is
    # actively streaming to the RPi. speech_end uses this to decide whether
    # to treat a user utterance as a normal turn or as a mid-response interrupt.
    ai_speaking = False

    async def _synthesize(text: str) -> bytes | None:
        """High-speed streaming synthesis. Returns 24kHz PCM bytes."""
        def _gen():
            audio_list = _models.tts_model.generate(
                text=text,
                voice_clone_prompt=current_voice_prompt,
                language="hi",
                generation_config=OmniVoiceGenerationConfig(
                    num_step=8, guidance_scale=2.0, t_shift=0.1,
                    audio_chunk_threshold=600.0, postprocess_output=False,
                ),
            )
            if not audio_list: return None
            
            audio_data = audio_list[0]
            if hasattr(audio_data, "cpu"):
                audio_data = audio_data.cpu().numpy()
            audio_data = np.asarray(audio_data, dtype=np.float32)
            
            # Return raw PCM 16-bit 24kHz (Standard for internal bridges)
            return (np.clip(audio_data, -1.0, 1.0) * 32767).astype(np.int16).tobytes()

        try:
            return await asyncio.to_thread(_gen)
        except Exception as e:
            print(f"[Phone] Synthesis error: {e}", flush=True)
            return None

    async def _send_to_bridge(audio: bytes):
        """Send 16kHz LE PCM16 audio to bridge in paced chunks. Bridge converts to ulaw on wire."""
        chunk_size = 8000
        for i in range(0, len(audio), chunk_size):
            await websocket.send_bytes(audio[i : i + chunk_size])

    async def say(text: str):
        """Legacy helper for greeting."""
        audio = await _synthesize(text)
        if audio:
            # 24kHz PCM -> 16kHz PCM (bridge handles 16k -> 8k ulaw on wire)
            audio_16k, _ = audioop.ratecv(audio, 2, 1, 24000, 16000, None)
            await _send_to_bridge(audio_16k)

    # Matches lone backchannels — 1-2 word affirmative fillers that should NOT
    # barge in on the AI. Covers Hindi + common English equivalents.
    _BACKCHANNEL_RE = re.compile(
        r'^(?:हम+|हम्म|हाँ|हां|हन+|जी|अच्छा|ठीक|ओह+|ओ|ok|okay|yeah|yes|hmm+|mmm+|uh+|umm?)[।\s.!?,]*$',
        re.IGNORECASE
    )

    def _is_backchannel_or_noise(text: str) -> bool:
        """Filters out tiny noise fragments or common backchannels."""
        if not text: return True
        stripped = text.strip().rstrip('।.!?, ').strip()
        if not stripped: return True
        if len(stripped) <= 2:  # e.g. 'हा', 'ok'
            return True
        if len(stripped.split()) == 1 and _BACKCHANNEL_RE.match(stripped):
            return True
        return False

    def _preprocess_for_stt(pcm_16k: bytes) -> np.ndarray:
        """Clean up phone audio before STT to lift accuracy on 8kHz-upsampled
        telephony audio:
          - DC offset removal (Asterisk ulaw->slin leaves a small bias)
          - Trim leading/trailing low-energy frames (pre-roll noise dilutes STT)
          - Peak-normalize to -3 dBFS so quiet calls don't get under-decoded
        Returns float32 mono in [-1, 1] expected by both Sarvam and Whisper."""
        audio = np.frombuffer(pcm_16k, dtype=np.int16).astype(np.float32) / 32768.0
        if audio.size == 0:
            return audio
        # 1. DC offset removal
        audio = audio - audio.mean()
        # 2. Trim silence at edges using a simple energy gate (32ms windows).
        win = 512  # 32 ms at 16 kHz
        if audio.size > win * 2:
            n_frames = audio.size // win
            frames = audio[: n_frames * win].reshape(n_frames, win)
            energy = np.abs(frames).mean(axis=1)
            thresh = max(0.005, energy.max() * 0.05)
            voiced = np.where(energy > thresh)[0]
            if voiced.size > 0:
                # Keep ~100ms of leading context so we don't clip the first phoneme.
                start = max(0, voiced[0] - 3) * win
                end = min(n_frames, voiced[-1] + 4) * win
                audio = audio[start:end]
        # 3. Peak-normalize to -3 dBFS (~0.707) — boosts quiet voices without clipping
        peak = float(np.abs(audio).max())
        if peak > 1e-4:
            audio = audio * (0.707 / peak)
            np.clip(audio, -1.0, 1.0, out=audio)
        return audio

    async def _run_stt(pcm_16k: bytes) -> str:
        """Standalone STT — returns transcript or empty string on failure."""
        audio_np = _preprocess_for_stt(pcm_16k)
        if audio_np.size < 1600:  # <100ms after trim — nothing to transcribe
            return ""
        stt_client = getattr(app.state, 'stt_client', None)
        try:
            if stt_client is not None and hasattr(stt_client, 'transcribe'):
                return (await stt_client.transcribe(audio_np)) or ""
            def _transcribe():
                model = _models.load_whisper()
                segs, _ = model.transcribe(audio_np, language="hi", beam_size=1,
                                           vad_filter=True, no_speech_threshold=0.6)
                return " ".join(s.text.strip() for s in segs).strip()
            return (await asyncio.to_thread(_transcribe)) or ""
        except Exception as e:
            print(f"[Phone] STT error: {e}", flush=True)
            return ""

    async def _barge_in(reason: str):
        """User interrupted with substantive speech — silence AI immediately.
        1. Tell RPi to flush its RTP sender queue (line goes silent instantly).
        2. Hard-cancel the synth and player sub-tasks (otherwise the player would
           drain whatever's already in audio_ready_queue, playing several more
           seconds of AI speech after the user interrupts).
        3. Cancel the outer utterance task (LLM stream).
        Caller is responsible for starting the new user turn afterwards."""
        nonlocal processing, ai_speaking, current_utterance_task
        print(f"[Phone] BARGE-IN: {reason}", flush=True)
        await broadcast_log("status", f"[Phone] BARGE-IN: {reason}")
        try:
            await websocket.send_json({"type": "barge_in"})
        except Exception as e:
            print(f"[Phone] barge_in signal send failed: {e}", flush=True)
        # Kill TTS sub-tasks first so no more bytes go on the wire.
        for st in list(current_tts_subtasks):
            if not st.done():
                st.cancel()
        # Cancelling the outer task propagates through the LLM streaming loop.
        task = current_utterance_task
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        # Wait for sub-tasks to finish unwinding so ai_speaking flips back cleanly.
        for st in list(current_tts_subtasks):
            try:
                await st
            except (asyncio.CancelledError, Exception):
                pass
        current_tts_subtasks.clear()
        ai_speaking = False
        processing = False
        current_utterance_task = None

    async def _handle_mid_speech_interrupt(pcm_16k: bytes):
        """Runs when speech_end arrives while AI is speaking.
        STT the utterance, classify, and either ignore (backchannel) or
        barge in and process as a new turn (substantive)."""
        transcript = await _run_stt(pcm_16k)
        if _is_backchannel_or_noise(transcript):
            await broadcast_log("status",
                f"[Phone] Mid-speech ignored (backchannel/noise): {transcript!r}")
            return
        await _barge_in(f"user said {transcript!r}")
        # Reuse the STT result — no need to transcribe the same audio twice.
        await process_utterance(pcm_16k, precomputed_transcript=transcript)

    def _clean_tags(text: str) -> str:
        """Removes incomplete emotion tags like '[lau' or lone '[' from the end of text."""
        text = text.strip()
        # Case 1: Ends with a naked opening bracket
        if text.endswith("["):
            return text[:-1].strip()
        # Case 2: Contains an opening bracket that is never closed
        last_open = text.rfind("[")
        if last_open != -1:
            last_close = text.rfind("]", last_open)
            if last_close == -1:
                # The tag is incomplete, e.g. "Hello [lau"
                return text[:last_open].strip()
        return text

    async def process_utterance(pcm_16k: bytes, precomputed_transcript: str | None = None):
        nonlocal processing, call_history, ai_speaking
        if processing:
            return
        processing = True
        try:
            t0 = time.time()
            if precomputed_transcript is not None:
                # Came from the barge-in path — STT already ran.
                transcript = precomputed_transcript
                stt_ms = 0
            else:
                dur_secs = round(len(pcm_16k) / 32000, 1)
                await broadcast_log("status", f"[Phone] STT starting ({dur_secs}s audio)...")
                transcript = await _run_stt(pcm_16k)
                stt_ms = int((time.time() - t0) * 1000)
            if not transcript:
                await broadcast_log("status", f"[Phone] STT returned empty ({stt_ms}ms)")
                return
            print(f"[Phone] User ({stt_ms}ms): {transcript}", flush=True)
            await broadcast_log("phone_user", transcript)
            await push_phone_transcript("user", transcript)

            # LLM → TTS streaming pipeline (mirrors browser voice assistant)
            # LLM streams text; as each sentence completes it's enqueued for TTS.
            # TTS worker synthesizes and sends audio concurrently with LLM generation.
            # First audio reaches the phone in ~(STT + LLM-first-sentence + TTS-chunk) ≈ 3-4s
            # instead of waiting for the full response (~9s in batch mode).
            provider = getattr(app.state, 'llm_provider', None) or resolve_provider()
            campaign = getattr(app.state, 'campaign', None) or load_campaign()
            system_prompt = build_system_prompt(
                campaign,
                voice_name=voice_meta.get("name", "AI"),
                voice_gender=voice_meta.get("gender", "male"),
                voice_metadata=voice_meta
            )

            await broadcast_log("status", "[Phone] LLM thinking...")
            tts_queue: asyncio.Queue = asyncio.Queue()
            audio_ready_queue: asyncio.Queue = asyncio.Queue(maxsize=3)
            tts_chunk_count = [0]

            async def tts_synthesizer_worker():
                """Synthesizes text sentences into audio buffers in background."""
                try:
                    while True:
                        text = await tts_queue.get()
                        if text is None:
                            await audio_ready_queue.put(None)
                            break
                        audio = await _synthesize(text)
                        if audio:
                            # 24kHz PCM -> 16kHz PCM. Bridge does 16k -> 8k ulaw on the wire.
                            audio_16k, _ = audioop.ratecv(audio, 2, 1, 24000, 16000, None)
                            await audio_ready_queue.put((text, audio_16k))
                except Exception as e:
                    print(f"[Phone] Synthesizer worker failed: {e}", flush=True)

            async def tts_player_worker():
                """Plays synthesized audio buffers sequentially to the bridge."""
                nonlocal ai_speaking
                try:
                    while True:
                        item = await audio_ready_queue.get()
                        if item is None: break
                        text, audio = item
                        ai_speaking = True
                        tts_chunk_count[0] += 1
                        await broadcast_log("status", f"[Phone] Playing TTS chunk {tts_chunk_count[0]}: {text[:50]!r}")
                        # Emit the exact text being spoken for karaoke-style word highlighting
                        await broadcast_log("phone_tts_speaking", text)
                        
                        # Send audio and WAIT for it to play out on the phone.
                        # This ensures ai_speaking remains True during the actual speech,
                        # which is critical for the barge-in (interrupt) system to work.
                        await _send_to_bridge(audio)
                        # 16kHz LE PCM16 = 32000 B/s; sleep so ai_speaking stays True for the whole playback.
                        duration = len(audio) / 32000.0
                        await asyncio.sleep(duration)
                        
                        await broadcast_log("status", f"[Phone] Played chunk {tts_chunk_count[0]} ({int(duration*1000)}ms)")
                finally:
                    ai_speaking = False

            synth_task = asyncio.create_task(tts_synthesizer_worker())
            play_task = asyncio.create_task(tts_player_worker())
            # Register so _barge_in can hard-cancel them on user interrupt.
            current_tts_subtasks.append(synth_task)
            current_tts_subtasks.append(play_task)
            tts_task = asyncio.gather(synth_task, play_task)

            llm_t0 = time.time()
            # Sentence delimiters: Hindi danda, !, ?, and period not between digits. 
            # Using capturing groups () so punctuation is preserved in the split list.
            SENT_RE = re.compile(r'([।!?]|(?<=[^\d])\.(?=[^\d]))')

            try:
                messages = [
                    *call_history[-30:],
                    {"role": "user", "content": transcript},
                ]
                payload = provider.build_payload(system_prompt, messages)
                extras = provider.payload_extras()
                if extras:
                    payload.update(extras)
                
                print(f"[Phone] Sending payload to {provider.url}: {json.dumps(payload, ensure_ascii=False)[:300]}...", flush=True)
                
                full_response = ""
                sentence_buf = ""
                # Batching logic: combine short phrases/tags to avoid robotic 'ruk-ruk' speech.
                current_batch = []
                first_token = True
                
                # Strip think tags if model is reasoning-capable (engine-style)
                think_state = {"in_think": False, "tail": ""}

                async with httpx.AsyncClient(timeout=60.0) as client:
                    async with client.stream("POST", provider.url,
                                             headers=provider.headers(),
                                             json=payload) as resp:
                        import typing
                        resp = typing.cast(httpx.Response, resp)
                        if resp.status_code != 200:
                            err_txt = await resp.aread()
                            print(f"[Phone] LLM Error {resp.status_code}: {err_txt.decode()}", flush=True)
                            await broadcast_log("system", f"[Phone] LLM Error {resp.status_code}")
                            return

                        async for line in resp.aiter_lines():
                            if not line or line == "data: [DONE]":
                                continue
                            if line.startswith("data: "):
                                try:
                                    chunk_data = json.loads(line[6:])
                                    delta = (chunk_data.get("choices", [{}])[0]
                                             .get("delta", {}).get("content") or "")
                                    if not delta:
                                        continue
                                    
                                    # Strip thinking tokens (engine-style)
                                    clean_delta = _strip_think_streaming(delta, think_state)
                                    if not clean_delta:
                                        continue

                                    if first_token:
                                        ttft_ms = int((time.time() - llm_t0) * 1000)
                                        await broadcast_log("status", f"[Phone] LLM TTFT {ttft_ms}ms")
                                        first_token = False
                                    
                                    full_response += clean_delta
                                    # [LIVE STREAM] Push partial AI text to both voice clients AND log clients
                                    await push_phone_transcript("ai", full_response, is_partial=True)
                                    await broadcast_log("phone_ai_partial", full_response)
                                    sentence_buf += clean_delta
                                    
                                    # Use 'Split and Pop' logic but PRESERVE the punctuation.
                                    # Capturing groups () in re.split keep the delimiters in the list.
                                    parts = SENT_RE.split(sentence_buf)
                                    if len(parts) > 2: # At least one [text, delim, text]
                                        # We can join pairs of (text, delim)
                                        # e.g. ['जी', '!', ' बताइए', '।', ' यार']
                                        # to_process will be [parts[0]+parts[1], parts[2]+parts[3], ...]
                                        to_process = []
                                        # The last part is always the incomplete trailing text
                                        for i in range(0, len(parts) - 1, 2):
                                            to_process.append(parts[i] + parts[i+1])
                                        
                                        sentence_buf = parts[-1]
                                        
                                        for s in to_process:
                                            s = _clean_tags(s)
                                            if not s: continue
                                            current_batch.append(s)
                                            
                                            batch_txt = " ".join(current_batch)
                                            # Flush if the batch is long enough for natural intonation
                                            if len(batch_txt) > 30: # Reduced from 50 for snappier endings
                                                await tts_queue.put(batch_txt)
                                                current_batch = []
                                except (json.JSONDecodeError, IndexError, KeyError):
                                    pass

                        # Final flush: combine everything left
                        final_txt = _clean_tags(" ".join(current_batch) + " " + sentence_buf)
                        if final_txt:
                            await tts_queue.put(final_txt)
                        sentence_buf = ""
                        current_batch = []
            except asyncio.CancelledError:
                # Barge-in kills this task. We MUST kill the tts_worker too, 
                # otherwise it keeps playing the remaining queue!
                if not tts_task.done():
                    tts_task.cancel()
                raise
            finally:
                # If _barge_in() already cancelled tts_task, putting None is
                # a no-op (queue is drained anyway) and awaiting raises
                # CancelledError — swallow it so the pipeline can unwind cleanly.
                if not tts_task.done():
                    try:
                        await tts_queue.put(None)  # sentinel — stop TTS worker
                    except Exception:
                        pass
                try:
                    # Give the player a tiny moment to finish the last byte
                    await asyncio.sleep(0.2)
                    await tts_task
                except (asyncio.CancelledError, Exception):
                    pass

            # If we got cancelled mid-stream by barge-in, we don't want to log
            # a truncated AI response as if it was a completed turn. Skip the
            # transcript/history write when the task was cancelled.
            if tts_task.cancelled():
                return
            response_text = _clean_tags(full_response)
            if not response_text:
                await broadcast_log("system", "[Phone] LLM returned empty response")
                return
            total_ms = int((time.time() - t0) * 1000)
            call_history.append({"role": "user", "content": transcript})
            call_history.append({"role": "assistant", "content": response_text})
            print(f"[Phone] AI ({total_ms}ms total): {response_text[:80]}", flush=True)
            await broadcast_log("phone_ai", response_text)
            # Final transcript push (is_partial=False)
            await push_phone_transcript("ai", response_text, is_partial=False)
            await broadcast_log("status", f"[Phone] Turn done — total {total_ms}ms, {tts_chunk_count[0]} TTS chunks")

        except Exception as e:
            await broadcast_log("system", f"[Phone] Pipeline error: {e}")
            print(f"[Phone] Pipeline error: {e}", flush=True)
        finally:
            # Deregister TTS subtasks so a stale reference doesn't outlive this
            # turn. Sub-tasks may not be defined if we failed before creating them.
            for name in ("synth_task", "play_task"):
                st = locals().get(name)
                if st is not None and st in current_tts_subtasks:
                    current_tts_subtasks.remove(st)
            ai_speaking = False
            processing = False

    try:
        msg_count = 0
        while True:
            message = await websocket.receive()
            msg_count += 1
            if message.get("type") == "websocket.disconnect":
                print(f"[Phone] WS disconnect received after {msg_count} messages, "
                      f"code={message.get('code')}", flush=True)
                break

            if "bytes" in message:
                # Bridge sends 16kHz LE PCM16 directly (RPi-side ulaw->16k conversion).
                stt_buffer.extend(message["bytes"])
                if len(stt_buffer) > MAX_BUFFER_BYTES:
                    # Drop oldest samples; keep last MAX_BUFFER_BYTES so a stuck
                    # debounce never grows the buffer without bound.
                    del stt_buffer[: len(stt_buffer) - MAX_BUFFER_BYTES]

            elif "text" in message:
                ev = json.loads(message["text"])
                t = ev.get("type")
                print(f"[Phone] Got text event: {t}", flush=True)

                if t == "call_start":
                    call_history.clear()
                    stt_buffer.clear()
                    mode = ev.get("mode", "outbound")
                    print(f"[Phone] Call started mode={mode} voice={voice_id}", flush=True)
                    # Greeting is now ALWAYS live to support dynamic personality/gender
                    name = voice_meta.get("name", "Vaani")
                    gender_verb = "रहा" if voice_meta.get("gender") == "male" else "रही"
                    asyncio.create_task(say(
                        f"नमस्ते! मैं वाणी एआई से {name} बोल {gender_verb} हूँ। आज किस टॉपिक पर बात करें?"
                    ))

                elif t == "speech_start":
                    # Cancel any pending debounced processor — the user is continuing
                    # to speak after a brief pause, so we should accumulate not process.
                    if pending_process_task is not None and not pending_process_task.done():
                        pending_process_task.cancel()
                        pending_process_task = None
                    # NOTE: do NOT clear stt_buffer here. Continuations within the
                    # debounce window must concatenate so STT sees a complete sentence.
                    stt_resample_state = None

                elif t == "speech_end":
                    buf_bytes = len(stt_buffer)
                    buf_secs = round(buf_bytes / 32000, 1)  # 16kHz PCM16 = 32000 B/s
                    print(f"[Phone] speech_end — buffer={buf_bytes}B ({buf_secs}s)", flush=True)
                    await broadcast_log("status", f"[Phone] speech_end buffer={buf_secs}s")
                    if buf_bytes <= 3200:
                        # Tiny fragment, drop and wait for real speech
                        stt_buffer.clear()
                    else:
                        # Cancel any earlier pending processor — this newer speech_end
                        # supersedes it (longer accumulated buffer).
                        if pending_process_task is not None and not pending_process_task.done():
                            pending_process_task.cancel()

                        # Snapshot for speculative STT (run in parallel with debounce).
                        # If no new audio arrives during the wait, this result is valid
                        # and saves the STT round-trip after the debounce fires.
                        snap = bytes(stt_buffer)

                        async def _process_after_debounce(snap_at_fire: bytes):
                            nonlocal current_utterance_task, stt_resample_state
                            stt_client = getattr(app.state, 'stt_client', None)
                            spec_task: asyncio.Task | None = None
                            if stt_client is not None and hasattr(stt_client, 'transcribe'):
                                spec_audio = _preprocess_for_stt(snap_at_fire)
                                if spec_audio.size >= 1600:
                                    spec_task = asyncio.create_task(stt_client.transcribe(spec_audio))

                            try:
                                await asyncio.sleep(ENDPOINT_DEBOUNCE_S)
                            except asyncio.CancelledError:
                                if spec_task and not spec_task.done():
                                    spec_task.cancel()
                                return

                            chunk_bytes = len(stt_buffer)
                            if chunk_bytes <= 3200:
                                stt_buffer.clear()
                                if spec_task and not spec_task.done():
                                    spec_task.cancel()
                                return
                            MAX_STT_BYTES = 928000  # 29s at 16kHz PCM16 (Sarvam cap)
                            if chunk_bytes > MAX_STT_BYTES:
                                secs = round(chunk_bytes / 32000, 1)
                                print(f"[Phone] Buffer too large ({secs}s), trimming to 29s", flush=True)
                                await broadcast_log("system", f"[Phone] WARNING: trimming {secs}s audio to 29s for STT")
                                chunk = bytes(stt_buffer[-MAX_STT_BYTES:])
                            else:
                                chunk = bytes(stt_buffer)
                            stt_buffer.clear()
                            stt_resample_state = None

                            transcript: Optional[str] = None
                            if spec_task is not None and chunk == snap_at_fire:
                                # No new audio came during the debounce — speculative
                                # transcription is a valid hit; saves an STT round trip.
                                try:
                                    # Ensure result is typed as str for the checker
                                    transcript = str(await spec_task)
                                    if transcript:
                                        print(f"[Phone] Speculative STT hit", flush=True)
                                except Exception:
                                    transcript = None
                            elif spec_task is not None:
                                # Audio grew during debounce; drop the stale speculative.
                                spec_task.cancel()

                            if ai_speaking:
                                if transcript is not None:
                                    # Already have transcript — classify here to avoid double STT.
                                    if _is_backchannel_or_noise(transcript):
                                        await broadcast_log("status",
                                            f"[Phone] Mid-speech ignored (backchannel/noise): {transcript!r}")
                                        return
                                    await _barge_in(f"user said {transcript!r}")
                                    current_utterance_task = asyncio.create_task(
                                        process_utterance(chunk, precomputed_transcript=transcript))
                                else:
                                    current_utterance_task = asyncio.create_task(
                                        _handle_mid_speech_interrupt(chunk))
                            else:
                                current_utterance_task = asyncio.create_task(
                                    process_utterance(chunk, precomputed_transcript=transcript))

                        pending_process_task = asyncio.create_task(_process_after_debounce(snap))

                elif t in ("call_end", "hangup"):
                    print(f"[Phone] Call ended ({t})", flush=True)
                    # Cancel any pending debounced processor — call is over
                    if pending_process_task is not None and not pending_process_task.done():
                        pending_process_task.cancel()
                    # Wait for any in-flight STT→LLM→TTS task so response reaches phone
                    if current_utterance_task and not current_utterance_task.done():
                        try:
                            await asyncio.wait_for(current_utterance_task, timeout=20.0)
                        except (asyncio.TimeoutError, Exception):
                            pass
                    break

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[Phone] WS error: {e}", flush=True)
    finally:
        app.state.active_phone_ws = None
        # Mark history as ended
        history = load_call_history()
        for entry in history:
            if entry.get("status") == "dialing":
                entry["status"] = "ended"
        save_call_history(history)
        print("[Phone] RPi bridge disconnected", flush=True)


@app.websocket("/ws/logs")
async def logs_websocket(websocket: WebSocket):
    """Endpoint for the Live Debug Dashboard."""
    await websocket.accept()
    log_clients.add(websocket)
    try:
        await broadcast_log("system", "Log stream connected.")
        while True:
            await websocket.receive_text() # Keep alive
    except WebSocketDisconnect:
        if websocket in log_clients: log_clients.remove(websocket)
    except:
        if websocket in log_clients: log_clients.remove(websocket)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    voice_clients.add(websocket)
    print("[WS] Client connected.")

    async def event_handler(msg_type, data):
        """Callback from engine to send data to WebSocket."""
        try:
            # Clean transcript for UI (remove emotional tags like [laughter])
            if msg_type == "transcript" and "text" in data:
                data["text"] = re.sub(r'\[.*?\]', '', data["text"]).strip()

            # Broadcast all non-audio events to the LOG viewer too
            if msg_type != "audio_chunk":
                await broadcast_log(msg_type, data)
                
            if msg_type == "audio_chunk":
                await websocket.send_bytes(data)
            else:
                await websocket.send_json({"type": msg_type, "data": data})
        except Exception as e:
            print(f"[WS] Send Error: {e}")

    # Get initial voice preference from query params (Persistence Fix)
    initial_voice_id = websocket.query_params.get("voice_id", "ravi")
    voices = load_voices()
    initial_voice = next((v for v in voices if v["id"] == initial_voice_id), None)

    provider = getattr(app.state, 'llm_provider', None) or resolve_provider()
    # Per-turn provider read so /provider/switch takes effect mid-session.
    def _current_provider():
        p = getattr(app.state, 'llm_provider', None)
        return p if p is not None else resolve_provider()
    # Per-turn campaign read so /campaign edits take effect on next user turn.
    def _current_campaign():
        cached = getattr(app.state, 'campaign', None)
        return cached if cached is not None else load_campaign()
    # Per-turn STT read so /stt/switch flips sarvam<->whisper without
    # reopening the socket. asr_worker calls this before each transcription.
    def _current_stt():
        s = getattr(app.state, 'stt_client', None)
        return s if s is not None else build_stt(resolve_stt())
    assistant = WebAssistant(
        event_handler=event_handler,
        llm_url=provider.url if provider else None,
        llm_model=provider.model if provider else None,
        llm_headers=provider.headers() if provider else None,
        llm_extras=provider.payload_extras() if provider else None,
        get_provider=_current_provider,
        get_campaign=_current_campaign,
        get_stt=_current_stt,
    )
    print(f"[WS] Assistant initialized with provider={provider.name if provider else 'None'}", flush=True)
    
    if initial_voice:
        # Use .get() for every field — voices.json is user-edited and some entries
        # (notably the default "ravi") ship with "accent" instead of "style", which
        # previously crashed the /ws endpoint immediately with KeyError: 'style'.
        assistant.active_voice_metadata = {
            "name": initial_voice.get("name", "Vaani"),
            "gender": initial_voice.get("gender", "male"),
            "style": initial_voice.get("style") or initial_voice.get("accent", "conversational"),
            "id": initial_voice.get("id", "ravi"),
            "age": initial_voice.get("age", "30"),
            "about": initial_voice.get("about", ""),
            "catchphrases": initial_voice.get("catchphrases", ""),
        }
        if initial_voice_id != "ravi":
            print(f"[WS] Initializing assistant with voice: {initial_voice['name']}")
            # Pre-generate voice prompt for the initial audio
            full_audio_path = os.path.join(ROOT_DIR, initial_voice["file_path"])
            new_prompt = await asyncio.to_thread(
                assistant.tts_model.create_voice_clone_prompt,
                ref_audio=full_audio_path,
                preprocess_prompt=True
            )
            # Goes through set_voice_prompt so backchannels get re-synthesized
            # in the selected voice (otherwise the pre-synth "अच्छा" plays in
            # Ravi's voice right before a Pankaj/etc reply — audible glitch).
            assistant.set_voice_prompt(new_prompt)

    print("[WS] Starting Engine run() task...", flush=True)
    engine_task = asyncio.create_task(assistant.run())
    
    # Auto-Greet (Prioritize template-defined example)
    campaign = assistant.get_campaign() if assistant.get_campaign else {}
    greet_text = campaign.get("call_goal_example") if campaign else ""
    if not greet_text:
        name = assistant.active_voice_metadata['name']
        gender_verb = 'रहा' if assistant.active_voice_metadata['gender']=='male' else 'रही'
        greet_text = f"नमस्ते! मैं {name} बोल {gender_verb} हूँ, वाणी से। आज मैं आपकी क्या सहायता कर सकता हूँ?"
    
    await assistant.tts_queue.put(greet_text)
    assistant.history.append({"role": "assistant", "content": greet_text})
    # [FIX] Without __END_RESPONSE__, is_speaking stays True forever and the ASR
    # adaptive threshold (4×) masks quiet user voice. Close the turn cleanly.
    await assistant.tts_queue.put("__END_RESPONSE__")

    # Diagnostic: log first 3 chunks + every 500th chunk so we can confirm PCM is
    # non-zero when the user speaks (not just at idle). Visible in LogViewer.
    diag_count = 0

    import struct

    def _mic_diag(raw: bytes, label: str) -> str:
        sample_count = len(raw) // 2
        head16 = raw[:16].hex()
        try:
            peak_i16 = max(abs(s) for s in struct.unpack(f"<{min(sample_count, 256)}h", raw[:min(len(raw), 512)])) if sample_count else 0
        except Exception:
            peak_i16 = -1
        return f"[WS-MIC-DIAG {label}] bytes={len(raw)} samples={sample_count} peak_int16={peak_i16} head_hex={head16}"

    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                 # Print exact close code + reason so we can tell server-initiated
                 # (ping timeout → 1011) from client-initiated (1000 normal,
                 # 1001 going-away, 1006 abnormal) closes. Turn 'connection closed'
                 # from uvicorn into something actionable.
                 code = message.get("code")
                 reason = message.get("reason")
                 print(f"[WS] disconnect msg code={code} reason={reason!r} is_speaking={assistant.is_speaking} tts_qsize={assistant.tts_queue.qsize()} llm_qsize={assistant.llm_queue.qsize()}")
                 break
            if "bytes" in message:
                raw = message["bytes"]
                diag_count += 1
                if diag_count <= 3 or diag_count % 500 == 0:
                    label = f"#{diag_count}" if diag_count <= 3 else f"periodic#{diag_count}"
                    await broadcast_log("log", _mic_diag(raw, label))
                await assistant.asr_queue.put(raw)
            elif "text" in message:
                data = json.loads(message["text"])
                if data.get("type") == "interrupt":
                    assistant.interrupt_event.set()
                    while not assistant.tts_queue.empty(): assistant.tts_queue.get_nowait()
                    while not assistant.llm_queue.empty(): assistant.llm_queue.get_nowait()
                    assistant.is_speaking = False
                    # Note: Do not clear interrupt_event here; llm_worker clears it upon next task
                    await websocket.send_json({"type": "status", "data": "interrupted"})
                
                elif data.get("type") == "text_input":
                    text = data.get("text")
                    if text:
                        print(f"[WS] User typed: {text}")
                        # Immediately emit the transcript item so the UI can render it the same way as voice
                        await event_handler("transcript", {"role": "user", "text": text})
                        # Interrupt any ongoing AI speech
                        assistant.interrupt_event.set()
                        await websocket.send_json({"type": "status", "data": "interrupted"})
                        while not assistant.tts_queue.empty(): assistant.tts_queue.get_nowait()
                        while not assistant.llm_queue.empty(): assistant.llm_queue.get_nowait()
                        assistant.is_speaking = False
                        # Note: _flush_coalesced or llm_worker clears the flag, we shouldn't do it synchronously
                        # otherwise background threads might not register the interrupt.
                        # Queue the text to LLM directly
                        await assistant.llm_queue.put(text)
                
                elif data.get("type") == "switch_model":
                    new_model = data.get("model_id")
                    if new_model:
                        # Go through resolve_provider so the new model inherits
                        # the current provider's URL, auth, and payload extras
                        # — otherwise assistant.llm_model drifts away from
                        # app.state.llm_provider and /system/status reports the
                        # wrong value on UI refresh.
                        current = getattr(app.state, "llm_provider", None) or resolve_provider()
                        try:
                            new_provider = resolve_provider(
                                override_name=current.name,
                                override_model=new_model,
                            )
                        except Exception as e:
                            await websocket.send_json({"type": "status", "data": f"Model switch failed: {e}"})
                        else:
                            app.state.llm_provider = new_provider
                            assistant.llm_model = new_model
                            save_provider_config(new_provider.name, new_provider.model)
                            print(f"[WS] switch_model -> provider={new_provider.name} model={new_model} (persisted)", flush=True)
                            await websocket.send_json({"type": "status", "data": f"Model switched to {new_model}"})

                elif data.get("type") == "reset_chat":
                    print(f"[WS] Resetting chat session and memory")
                    assistant.reset_session()
                    await websocket.send_json({"type": "status", "data": "Chat history cleared"})

                elif data.get("type") == "log_forward":
                    await broadcast_log("log", f"[FRONTEND] {data.get('data')}")

                elif data.get("type") == "switch_voice":
                    voice_id = data.get("voice_id")
                    voices = load_voices()
                    voice = next((v for v in voices if v["id"] == voice_id), None)
                    if voice:
                        print(f"[WS] Attempting to switch voice: {voice.get('name','?')} (ID: {voice_id})")
                        try:
                            # Update Assistant Metadata — defensive lookups
                            assistant.active_voice_metadata = {
                                "name": voice.get("name", "Vaani"),
                                "gender": voice.get("gender", "male"),
                                "style": voice.get("style") or voice.get("accent", "conversational"),
                                "id": voice.get("id", "ravi"),
                                "age": voice.get("age", "30"),
                                "about": voice.get("about", ""),
                                "catchphrases": voice.get("catchphrases", ""),
                            }
                            # Re-generate voice prompt for the new audio
                            full_audio_path = os.path.join(ROOT_DIR, voice["file_path"])
                            print(f"[WS] -> Cloning from {full_audio_path}...")
                            
                            new_prompt = await asyncio.to_thread(
                                assistant.tts_model.create_voice_clone_prompt,
                                ref_audio=full_audio_path,
                                preprocess_prompt=True
                            )
                            # Triggers async backchannel re-synth in the new
                            # voice — prevents the wrong-speaker glitch on
                            # the next turn's opening "अच्छा".
                            assistant.set_voice_prompt(new_prompt)
                            print(f"[WS] -> SUCCESS: Voice changed to {voice['name']}")
                            await websocket.send_json({"type": "status", "data": f"Voice switched to {voice['name']}"})
                            
                            # Auto-Greet in new voice
                            greet_text = f"Namaste! Main {voice['name']} bol {'raha' if voice['gender']=='male' else 'rahi'} hoon Vaani AI se. Aaj kaise help karu aapki?"
                            await assistant.tts_queue.put(greet_text)
                            await assistant.tts_queue.put("__END_RESPONSE__")
                        except Exception as e:
                            print(f"[WS] -> ERROR during voice switch: {e}")
                            await websocket.send_json({"type": "status", "data": f"Error switching voice: {str(e)}"})

    except WebSocketDisconnect:
        print("[WS] Client disconnected.", flush=True)
    except Exception as e:
        import traceback
        print(f"[WS] Error: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
    finally:
        voice_clients.discard(websocket)
        print(f"[WS] Entering finally. engine_task.done={engine_task.done()} is_running={assistant.is_running}", flush=True)
        assistant.stop()
        engine_task.cancel()
        print(f"[WS] After cancel: engine_task.done={engine_task.done()} cancelled={engine_task.cancelled()}", flush=True)

@app.websocket("/ws/openai")
async def openai_realtime(websocket: WebSocket):
    await websocket.accept()
    print("[OpenAI-WS] Client connected.")

    if not OPENAI_API_KEY:
        print("[OpenAI-WS] ERROR: API Key missing.")
        await websocket.close(code=4000, reason="API Key Missing")
        return

    try:
        async with websockets.connect(
            OPENAI_URL,
            extra_headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "OpenAI-Beta": "realtime=v1"
            }
        ) as openai_ws:
            print("[OpenAI-WS] Connected to OpenAI.")

            # Send session update for Server-Side VAD
            session_update = {
                "type": "session.update",
                "session": {
                    "modalities": ["text", "audio"],
                    "instructions": "You are Vaani, a helpful cinematic AI assistant. Reply in Hinglish (Hindi + English) using Devanagari script for Hindi words. Be conversational and warm.",
                    "voice": "alloy",
                    "input_audio_format": "pcm16",
                    "output_audio_format": "pcm16",
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": 0.5,
                        "prefix_padding_ms": 300,
                        "silence_duration_ms": 500
                    }
                }
            }
            await openai_ws.send(json.dumps(session_update))

            async def client_to_openai():
                try:
                    while True:
                        message = await websocket.receive()
                        if "bytes" in message:
                            # OpenAI expects base64 for audio chunks
                            audio_b64 = base64.b64encode(message["bytes"]).decode("utf-8")
                            await openai_ws.send(json.dumps({
                                "type": "input_audio_buffer.append",
                                "audio": audio_b64
                            }))
                        elif "text" in message:
                            data = json.loads(message["text"])
                            # Pass-through other events if needed
                            await openai_ws.send(json.dumps(data))
                except Exception as e:
                    print(f"[Relay] Client -> OpenAI error: {e}")

            async def openai_to_client():
                try:
                    async for openai_msg in openai_ws:
                        data = json.loads(openai_msg)
                        
                        # Handle specific events that the UI needs
                        msg_type = data.get("type")
                        
                        # VAD Start Event
                        if msg_type == "input_audio_buffer.speech_started":
                            await websocket.send_json({"type": "status", "data": "listening"})
                        
                        # VAD Stop/Transcript/Processing events
                        elif msg_type == "input_audio_buffer.speech_stopped":
                            await websocket.send_json({"type": "status", "data": "thinking"})

                        # Audio output from OpenAI
                        elif msg_type == "response.audio.delta":
                            audio_bytes = base64.b64decode(data["delta"])
                            await websocket.send_bytes(audio_bytes)
                        
                        # Text transcripts for UI
                        elif msg_type == "response.audio_transcript.delta":
                             await websocket.send_json({"type": "llm_chunk", "data": data["delta"]})
                        
                        elif msg_type == "response.audio_transcript.done":
                            await websocket.send_json({"type": "transcript", "data": {"role": "ai", "text": data["transcript"]}})

                except Exception as e:
                    print(f"[Relay] OpenAI -> Client error: {e}")

            # Run both relay tasks
            await asyncio.gather(client_to_openai(), openai_to_client())

    except Exception as e:
        print(f"[OpenAI-WS] Connection Failure: {e}")
    finally:
        print("[OpenAI-WS] Closed.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

