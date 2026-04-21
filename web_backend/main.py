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
import psutil # For High-Priority scheduling

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
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
try:
    from web_backend.engine import WebAssistant, _models
except ImportError:
    from engine import WebAssistant, _models

try:
    from web_backend.llm_providers import (
        resolve_provider, ollama_unload_all, ollama_warmup, save_provider_config,
    )
except ImportError:
    from llm_providers import (
        resolve_provider, ollama_unload_all, ollama_warmup, save_provider_config,
    )

try:
    from web_backend.stt_providers import (
        resolve_stt, build_stt, save_stt_config,
    )
except ImportError:
    from stt_providers import (
        resolve_stt, build_stt, save_stt_config,
    )

try:
    from web_backend.campaigns import load_campaign, save_campaign
except ImportError:
    from campaigns import load_campaign, save_campaign

try:
    from web_backend.audio_utils import trim_audio_file
except ImportError:
    from audio_utils import trim_audio_file

from datetime import datetime

try:
    from web_backend.asterisk_bridge_helper import VaaniAsteriskBridge
except ImportError:
    from asterisk_bridge_helper import VaaniAsteriskBridge


ARI_BASE_URL = "http://192.168.8.59:8088/ari"
ARI_AUTH = ("ari_user", "ari_pass")
HISTORY_FILE = os.path.join(ROOT_DIR, "calls_history.json")

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_URL = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01"

from contextlib import asynccontextmanager

# --- Log Management ---
log_clients = set()
SESSION_LOG = os.path.join(ROOT_DIR, "session_trace.log")
# Non-blocking file sink: broadcast_log pushes lines to this queue and returns
# immediately. A background task (started in lifespan) batches flushes to disk
# so the LLM/TTS hot path never waits on I/O. The queue is bounded so a stalled
# disk doesn't cause unbounded memory growth — it just drops oldest lines.
_log_file_queue: asyncio.Queue | None = None
_LOG_QUEUE_MAX = 5000


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

    # Resolve the active LLM provider once at startup. Local Ollama gets a GPU
    # prewarm; remote APIs (Sarvam) are HTTP-only so we skip all local warmup.
    active_provider = resolve_provider()
    print(f"[Startup] LLM provider: {active_provider.name} model={active_provider.model}")

    if active_provider.uses_local_gpu:
        # Prewarm Ollama with FULL GPU offload (num_gpu=99). Cuts TTFT from
        # ~400 ms (CPU) to ~100 ms (GPU). num_ctx=1024 keeps KV cache small.
        print("[Startup] Pre-warming Ollama on GPU (num_gpu=99, num_ctx=1024)...")
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                await client.post("http://127.0.0.1:11434/api/generate", json={
                    "model": active_provider.model,
                    "prompt": "test",
                    "stream": False,
                    "options": {"num_predict": 1, "num_gpu": 99, "num_ctx": 1024}
                })
            print("[Startup] Ollama pre-warmed on GPU.")
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
    app.state.bridge = VaaniAsteriskBridge(log_handler=broadcast_log)
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
    name: str = None
    gender: str = None
    style: str = None
    age: str = None
    about: str = None
    catchphrases: str = None

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
async def hangup_call(channel_id: str = None):
    if not hasattr(app.state, 'bridge'):
        return {"status": "error", "message": "Bridge not running"}

    if channel_id:
        success = await app.state.bridge.hangup_session(channel_id)
    else:
        await app.state.bridge.hangup_all()
        success = True

    # Mark any "dialing" history rows as ended — the active call is gone.
    history = load_call_history()
    changed = False
    for entry in history:
        if entry.get("status") == "dialing":
            entry["status"] = "ended"
            changed = True
    if changed:
        save_call_history(history)

    if channel_id:
        return {"status": "success" if success else "not_found"}
    return {"status": "success", "message": "All calls cleared"}

@app.post("/calls/dial")
async def dial_number(req: DialRequest):
    phone = req.phone_number.strip()
    if not phone:
        raise HTTPException(status_code=400, detail="Phone number required")
    
    # --- PROACTIVE LOCKING: Prevent Double/Parallel Calling ---
    if hasattr(app.state, 'bridge') and len(app.state.bridge.sessions) > 0:
        print("[Dialer] BLOCKING: A call is already in progress.")
        raise HTTPException(status_code=400, detail="A call is already in progress. End current call first.")

    # Standardize phone number for the Gateway (assuming direct dial)
    # Target endpoint: PJSIP/1001/sip:<number>@192.168.8.60
    # Note: 192.168.8.60 is the Dinstar Gateway
    target_endpoint = f"PJSIP/1001/sip:{phone}@192.168.8.60"
    
    print(f"[Dialer] Requesting outbound call to {phone} via {target_endpoint}")
    
    # 1. Originate via ARI
    # app: ai-call-app (what our bridge listens to)
    originate_url = f"{ARI_BASE_URL}/channels"
    payload = {
        "endpoint": target_endpoint,
        "app": "ai-call-app",
        "appArgs": f"dialed_from_web,{phone},{req.voice_id}",
        "callerId": f"Vaani AI <1001>"
    }
    
    try:
        async with httpx.AsyncClient(auth=ARI_AUTH) as client:
            resp = await client.post(originate_url, json=payload)
            if resp.status_code not in (200, 201):
                print(f"[Dialer] Asterisk Error: {resp.text}")
                status = "failed"
            else:
                status = "dialing"
    except Exception as e:
        print(f"[Dialer] Connectivity Error: {e}")
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
    if new_name not in ("ollama", "sarvam"):
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
async def put_campaign(body: dict):
    """Replace the active campaign config. Persists to campaigns.json and
    updates the in-memory cache so the next LLM turn picks up the new prompt
    without a WebSocket reconnect."""
    if not isinstance(body, dict) or not body:
        raise HTTPException(status_code=400, detail="campaign body must be a non-empty object")
    try:
        save_campaign(body, name="default")
        app.state.campaign = body
        print(f"[Campaign] updated: label={body.get('label')!r} candidate={body.get('candidate_name')!r}", flush=True)
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

# --- VIP Persona Database (Zero-Hallucination) ---
VIP_PERSONAS = {
    "amitabh bachchan": {
        "age": "81",
        "gender": "male",
        "about": "अमिताभ बच्चन भारतीय फिल्म जगत के सबसे महान अभिनेता हैं। उन्हें 'शहंशाह' और 'एंग्री यंग मैन' के रूप में जाना जाता है। उन्होंने हिंदी सिनेमा को एक नई पहचान दी है।",
        "catchphrases": "रिश्ते में तो हम तुम्हारे बाप लगते हैं, नाम है शहंशाह।\n\nआज मेरे पास बंगला है, गाड़ी है, बैंक बैलेंस है... तुम्हारे पास क्या है?\n\nपरंपरा, प्रतिष्ठा, अनुशासन... ये इस गुरुकुल के तीन स्तंभ हैं।\n\nविजय दीनानाथ चौहान! पूरा नाम बाप का नाम दीनानाथ चौहान..."
    },
    "shah rukh khan": {
        "age": "58",
        "gender": "male",
        "about": "शाहरुख खान, जिन्हें 'किंग खान' और 'बादशाह' कहा जाता है, हिंदी फिल्मों के सबसे लोकप्रिय अभिनेता हैं। वे अपनी रोमांटिक भूमिकाओं और आकर्षण के लिए पूरी दुनिया में प्रसिद्ध हैं।",
        "catchphrases": "हारकर जीतने वाले को बाजीगर कहते हैं।\n\nबड़े-बड़े देशों में ऐसी छोटी-छोटी बातें होती रहती हैं, सेनोरिटा।\n\nसत्ते पे सत्ता... नहीं, सत्तर मिनट हैं तुम्हारे पास! शायद तुम्हारी जिंदगी के सबसे खास सत्तर मिनट।\n\nनाम तो सुना होगा?"
    },
    "salman khan": {
        "age": "58",
        "gender": "male",
        "about": "सलमान खान बॉलीवुड के 'दबंग' और 'भाईजान' हैं। वे अपने एक्शन, बॉडीबिल्डिंग और ब्लॉकबस्टर फिल्मों के लिए जाने जाते हैं। वे भारत के सबसे बड़े सितारों में से एक हैं।",
        "catchphrases": "एक बार जो मैंने कमिटमेंट कर दी, फिर तो मैं अपने आप की भी नहीं सुनता।\n\nमुझ पर एक अहसान करना कि मुझ पर कोई अहसान मत करना।\n\nस्वागत नहीं करोगे हमारा?\n\nअभी हम यहाँ के नए कोतवाल हैं... साले, तू हमारे घर में आ के हमें डराएगा?"
    },
    "narendra modi": {
        "age": "73",
        "gender": "male",
        "about": "नरेन्द्र मोदी भारत के 14वें प्रधानमंत्री हैं। वे अपने ओजस्वी भाषणों, विकास कार्यों और वैश्विक मंच पर भारत के सशक्त नेतृत्व के लिए जाने जाते हैं।",
        "catchphrases": "भाइयों और बहनों!\n\nमित्रों, मैं आपको विश्वास दिलाता हूँ... अच्छे दिन आने वाले हैं।\n\nसबका साथ, सबका विकास, सबका विश्वास और सबका प्रयास।\n\nआत्मनिर्भर भारत ही हमारा संकल्प है।"
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

    # Fallback to LLM for unknown names
    prompt = (
        f"Generate a persona for {req.name}. Output ONLY raw JSON matching this schema:\n"
        '{"age": "estimated numeric age", '
        '"about": "3 sentences in Devanagari Hindi about their career. Ensure sentences are COMPLETE.", '
        '"dialogues": ["4 most iconic signature cinematic dialogues in Devanagari Hindi"]}\n\n'
        "EXAMPLES of iconic dialogues:\n"
        "- Amitabh Bachchan: ['रिश्ते में तो हम तुम्हारे बाप लगते हैं, नाम है शहंशाह', 'आज मेरे पास बंगला है, गाड़ी है, बैंक बैलेंस है... तुम्हारे पास क्या है?']\n"
        "- Shah Rukh Khan: ['हारकर जीतने वाले को बाजीगर कहते हैं', 'बड़े-बड़े देशों में ऐसी छोटी-छोटी बातें होती रहती हैं']\n"
        f"Now generate for: {req.name}"
    )
    
    payload = {
        "model": "gemma3:4b",
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": 1024, "temperature": 0.2}
    }
    
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post("http://127.0.0.1:11434/api/generate", json=payload)
            resp.raise_for_status()
            data = resp.json()
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
                persona_data = {"age": "40", "about": "Failed to parse API output. But I am ready.", "dialogues": response_text.replace('"', '')}
                
            dialogues_raw = persona_data.get("dialogues", "")
            if isinstance(dialogues_raw, list):
                dialogues_fmt = "\n\n".join([str(d).strip() for d in dialogues_raw])
            else:
                dialogues_fmt = str(dialogues_raw)
                if dialogues_fmt.startswith("[") and dialogues_fmt.endswith("]"):
                    import ast
                    try:
                        # Try parsing stringified list
                        parsed_list = ast.literal_eval(dialogues_fmt)
                        if isinstance(parsed_list, list):
                            dialogues_fmt = "\n\n".join(parsed_list)
                    except:
                        dialogues_fmt = dialogues_fmt.strip("[]").replace("', '", "\n\n").replace('", "', "\n\n")

            return {
                "age": str(persona_data.get("age", "30")),
                "about": str(persona_data.get("about", "")),
                "catchphrases": dialogues_fmt
            }
    except Exception as e:
        print(f"Error generating persona: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate persona from LLM")

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

    provider = getattr(app.state, 'llm_provider', None)
    # Per-turn provider read so /provider/switch takes effect mid-session.
    def _current_provider():
        return getattr(app.state, 'llm_provider', None)
    # Per-turn campaign read so /campaign edits take effect on next user turn.
    def _current_campaign():
        cached = getattr(app.state, 'campaign', None)
        return cached if cached is not None else load_campaign()
    # Per-turn STT read so /stt/switch flips sarvam<->whisper without
    # reopening the socket. asr_worker calls this before each transcription.
    def _current_stt():
        return getattr(app.state, 'stt_client', None)
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
            assistant.voice_prompt = await asyncio.to_thread(
                assistant.tts_model.create_voice_clone_prompt,
                ref_audio=full_audio_path,
                preprocess_prompt=True
            )

    engine_task = asyncio.create_task(assistant.run())
    
    # Auto-Greet (now in the selected voice and DEVANAGARI to avoid being stripped)
    name = assistant.active_voice_metadata['name']
    gender_verb = 'रहा' if assistant.active_voice_metadata['gender']=='male' else 'रही'
    greet_text = f"नमस्ते! मैं {name} बोल {gender_verb} हूँ, वाणी से। आज मैं आपकी क्या सहायता कर सकता हूँ?"
    await assistant.tts_queue.put(greet_text)
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
                    assistant.interrupt_event.clear()
                    await websocket.send_json({"type": "status", "data": "interrupted"})
                
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
                            
                            assistant.voice_prompt = await asyncio.to_thread(
                                assistant.tts_model.create_voice_clone_prompt,
                                ref_audio=full_audio_path,
                                preprocess_prompt=True
                            )
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

