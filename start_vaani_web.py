import subprocess
import os
import sys
import time
import socket
import httpx

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# --- Environment Fixes (CRITICAL for Windows/NumPy Stability) ---
os.environ["OPENBLAS_MAIN_FREE"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_MAX_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["PYTHONFPEMASK"] = "1"
os.environ["NPY_DISABLE_CPU_FEATURES"] = "AVX512F"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # Ensure index matches hardware
os.environ["OLLAMA_MAX_LOADED_MODELS"] = "0"   # Force Ollama to be lean
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["HF_HUB_OFFLINE"] = "1"           # Force local model loading (Bypass DNS errors)
# -------------------------------------------------------------

def kill_port(port):
    """Finds and kills a process using the specified port on Windows."""
    try:
        # Get PID using netstat
        cmd = f'netstat -ano | findstr :{port}'
        output = subprocess.check_output(cmd, shell=True).decode()
        for line in output.strip().split('\n'):
            if f':{port}' in line and 'LISTENING' in line:
                pid = line.strip().split()[-1]
                print(f"[Cleanup] Killing process {pid} (and children) on port {port}...")
                subprocess.run(['taskkill', '/F', '/T', '/PID', pid], capture_output=True)
    except Exception:
        pass # Port probably already free

def restart_ollama_pinned():
    """Kill any running ollama.exe and relaunch `ollama serve` with CUDA_VISIBLE_DEVICES=0.
    This pins Gemma 3:4B to the RTX 3060 alongside OmniVoice; the GTX 1650 is reserved
    for Whisper and does not have enough VRAM for the LLM."""
    print("[Ollama] Restarting pinned to cuda:0 (RTX 3060)...")
    subprocess.run(['taskkill', '/F', '/IM', 'ollama.exe'], capture_output=True)
    time.sleep(1)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["OLLAMA_HOST"] = "127.0.0.1:11434"
    # Detach so the launcher can exit without killing the Ollama server.
    creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | \
                    getattr(subprocess, "DETACHED_PROCESS", 0x00000008)
    try:
        subprocess.Popen(["ollama", "serve"], env=env, creationflags=creationflags,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        print("[Ollama] WARNING: `ollama` not on PATH. Skipping restart — assuming external service.")
        return
    # Wait for API to come back.
    import urllib.request
    for i in range(30):
        try:
            urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=1).read()
            print(f"[Ollama] Pinned to cuda:0, API responding after {i+1}s.")
            return
        except Exception:
            time.sleep(1)
    print("[Ollama] WARNING: did not come back up in 30s. Check `ollama serve` manually.")


def wait_for_port(port, timeout=30):
    """Waits for a port to become active with progress feedback."""
    start_time = time.time()
    last_print = start_time
    print(f"[Wait] Monitoring port {port}...", end="", flush=True)
    
    while time.time() - start_time < timeout:
        try:
            with socket.create_connection(("localhost", port), timeout=1):
                print(" [READY]")
                return True
        except (socket.timeout, ConnectionRefusedError, TimeoutError, OSError):
            elapsed = int(time.time() - start_time)
            if time.time() - last_print > 10:
                print(f"\n[Wait] Still waiting... ({elapsed}s elapsed). AI models are loading, please do not close.", end="", flush=True)
                last_print = time.time()
            else:
                print(".", end="", flush=True)
            time.sleep(1)
    print("\n[Error] Timeout reached.")
    return False

def run():
    print("\n--- [Vaani Web] Super Launcher (One-Click Stability) ---")
    
    # 1. Cleanup Ghost Processes
    print("[0/3] Cleaning up old processes...")
    kill_port(8000)
    kill_port(3000)
    
    # Aggressive cleanup for any lingering Python processes running Omini scripts
    try:
        import subprocess
        subprocess.run(['taskkill', '/F', '/IM', 'python.exe', '/FI', 'WINDOWTITLE eq Vaani*'], capture_output=True)
        # Attempt to kill any process with 'asterisk_bridge' in cmdline if wmic is available
        subprocess.run('wmic process where "commandline like \'%vaani_asterisk_bridge%\'" delete', shell=True, capture_output=True)
    except:
        pass
        
    time.sleep(1)

    # Only manage the local Ollama server when it's actually the active LLM
    # backend. Remote providers (e.g. Sarvam) are HTTP-only, so launching
    # Ollama would just waste VRAM needed by OmniVoice + Whisper.
    # Priority: persisted provider_config.json (written by /provider/switch)
    # > env LLM_PROVIDER > default. Keeps the user's runtime choice sticky
    # across restarts without them editing .env.
    llm_provider = os.environ.get("LLM_PROVIDER", "ollama").lower()
    try:
        import json as _json
        _cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "provider_config.json")
        with open(_cfg_path, "r", encoding="utf-8") as _f:
            _cfg = _json.load(_f)
        _persisted = (_cfg.get("provider") or "").strip().lower()
        if _persisted in ("ollama", "sarvam"):
            llm_provider = _persisted
            print(f"[Launcher] Using persisted LLM provider: {llm_provider} "
                  f"(model={_cfg.get('model')})")
            # Mirror into env so the backend subprocess picks up the same choice
            # before resolve_provider() runs at startup.
            os.environ["LLM_PROVIDER"] = llm_provider
            if _cfg.get("model"):
                os.environ["LLM_MODEL"] = str(_cfg["model"])
    except FileNotFoundError:
        pass
    except Exception as _e:
        print(f"[Launcher] provider_config.json read skipped: {_e}")
    if llm_provider == "ollama":
        # Free any stale Ollama model caches before relaunching pinned to cuda:0.
        print("[0.5/3] Freeing Ollama GPU cache...")
        try:
            import urllib.request
            for model in ["gemma3:4b", "gemma4:e4b"]:
                req = urllib.request.Request("http://localhost:11434/api/generate",
                    data=f'{{"model":"{model}","keep_alive":0}}'.encode(), method="POST")
                urllib.request.urlopen(req, timeout=5)
        except Exception:
            pass
        time.sleep(2)

        # Restart Ollama pinned to RTX 3060 so gemma3:4b lives alongside OmniVoice.
        restart_ollama_pinned()
    else:
        print(f"[0.5/3] LLM_PROVIDER={llm_provider} — skipping Ollama startup (remote API mode).")
        # Best-effort: kill any lingering ollama.exe so it doesn't hold VRAM.
        try:
            subprocess.run(['taskkill', '/F', '/IM', 'ollama.exe'], capture_output=True)
        except Exception:
            pass

    # 2. Start Backend using UV
    print("[1/3] Starting FastAPI Backend on port 8000...")
    backend_log = open("backend.log", "w", encoding="utf-8")
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"  
    
    # [STABILITY FIX] Use BELOW_NORMAL_PRIORITY_CLASS (0x00004000) for Backend 
    # to prevent it from stalling the Windows UI during heavy GPU compute.
    # On Windows, this ensures the desktop remains responsive even if 3060/1650 are pegged.
    backend_p = subprocess.Popen(
        ["uv", "run", "python", "main.py"], 
        cwd="web_backend", 
        stdout=backend_log, 
        stderr=backend_log, 
        env=env,
        creationflags=0x00004000 # BELOW_NORMAL_PRIORITY_CLASS
    )
    
    # Wait for Backend to be ready
    print("[Wait] Waiting for AI Core to initialize (this may take 1-3 mins)...")
    if not wait_for_port(8000, timeout=300):
        print("Backend failed to start in time. Check web_backend logs.")
        backend_p.terminate()
        return

    # 3. Start Frontend
    print("[2/3] Starting Next.js Frontend on port 3000...")
    frontend_p = subprocess.Popen(["npm", "run", "dev"], cwd="web_frontend", shell=True)

    
    # Wait for Frontend
    if not wait_for_port(3000):
        print("Frontend failed to start. Check web_frontend logs.")
        backend_p.terminate()
        frontend_p.terminate()
        return


    print("\n--- ALL SYSTEMS ONLINE! ---")
    print("Dashboard: http://localhost:3000")
    print("Press Ctrl+C to stop all servers.")
    
    try:
        while True:
            time.sleep(2)
            if backend_p.poll() is not None:
                print("\n[Alert] Backend crashed. Stopping system...")
                break
            if frontend_p.poll() is not None:
                print("\n[Alert] Frontend crashed. Stopping system...")
                break

    except KeyboardInterrupt:
        print("\n[Shutting Down] Closing Vaani system...")
    finally:
        backend_p.terminate()
        frontend_p.terminate()



if __name__ == "__main__":
    run()
