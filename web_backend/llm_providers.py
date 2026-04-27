"""Pluggable LLM backends for the voice pipeline.

Both Ollama (local) and Sarvam (HTTP) speak OpenAI-compatible /chat/completions
with SSE streaming, so the streaming loop in `engine.llm_worker` stays
identical — only the URL, auth header, model ID, and a small provider-specific
tail on the request payload differ.

Selection is via env vars (evaluated per `resolve_provider` call, so a restart
or reload picks up changes to .env):

    LLM_PROVIDER     "ollama" (default) | "sarvam"
    LLM_MODEL        Optional override for the provider's default model
    SARVAM_API_KEY   Required when LLM_PROVIDER=sarvam
    OLLAMA_URL       Optional override for the local Ollama endpoint
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any


OLLAMA_DEFAULT_URL = "http://127.0.0.1:11434/v1/chat/completions"
OLLAMA_DEFAULT_MODEL = "gemma3:4b"
OLLAMA_API_BASE = "http://127.0.0.1:11434"

SARVAM_URL = "https://api.sarvam.ai/v1/chat/completions"
SARVAM_DEFAULT_MODEL = "sarvam-m"  # mid-tier ("average") Indic LLM

MINIMAX_URL = "https://api.minimax.io/v1/chat/completions"
MINIMAX_DEFAULT_MODEL = "MiniMax-M2.5"

# Persisted provider choice. Sits at project root next to campaigns.json so the
# user's runtime /provider/switch survives a restart — otherwise each boot
# falls back to env (LLM_PROVIDER=ollama by default) and wastes VRAM relaunching
# Ollama when the user had already moved to Sarvam.
PROVIDER_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "provider_config.json",
)


def load_provider_config(path: str = PROVIDER_CONFIG_PATH) -> Dict[str, Optional[str]]:
    """Read the persisted provider+model. Returns {} if missing or malformed so
    callers just fall through to env/defaults — never raises on a bad file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return {}
    except Exception as e:
        print(f"[llm_providers] config read error: {e}", flush=True)
        return {}
    if not isinstance(data, dict):
        return {}
    prov = (data.get("provider") or "").strip().lower() or None
    model = data.get("model") or None
    return {"provider": prov, "model": model}


def save_provider_config(provider: str, model: Optional[str],
                         path: str = PROVIDER_CONFIG_PATH) -> bool:
    """Persist provider+model atomically enough for a single-writer config.
    Best-effort: logs and returns False on failure, caller already mutated
    in-memory state so a disk hiccup shouldn't roll that back."""
    try:
        payload = {"provider": provider, "model": model}
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
        return True
    except Exception as e:
        print(f"[llm_providers] config save error: {e}", flush=True)
        return False

# All Ollama tags we might have pulled and want to evict on provider switch.
# Keeping this list centralised so the /provider/switch endpoint and the
# launcher's cache-freeing loop stay in sync.
KNOWN_OLLAMA_TAGS = [
    "gemma3:4b",
    "gemma4:e4b",
    "qwen2:7b",
    "mashriram/sarvam-m:latest",
]


async def ollama_unload_all(http_client) -> list[str]:
    """Evict every known Ollama model from GPU memory by sending keep_alive=0.

    Uses the resident list first (cheap, accurate), then falls back to the
    static tag list so we catch anything the /api/ps view missed. Returns the
    model names we successfully asked to unload; swallows network errors so a
    dead Ollama doesn't block a provider switch."""
    targets: set[str] = set()
    try:
        r = await http_client.get(f"{OLLAMA_API_BASE}/api/ps", timeout=1.5)
        if r.status_code == 200:
            for m in r.json().get("models", []):
                if m.get("name"):
                    targets.add(m["name"])
    except Exception:
        pass
    targets.update(KNOWN_OLLAMA_TAGS)

    unloaded: list[str] = []
    for name in targets:
        try:
            await http_client.post(
                f"{OLLAMA_API_BASE}/api/generate",
                json={"model": name, "keep_alive": 0},
                timeout=3.0,
            )
            unloaded.append(name)
        except Exception:
            pass
    return unloaded


async def ollama_warmup(http_client, model: str) -> bool:
    """Re-pin a model into GPU memory by asking for a single-token completion.
    Keeps keep_alive default (5m) so it stays resident. False on any error —
    caller logs and continues; the next real turn will just be slightly slower."""
    try:
        r = await http_client.post(
            f"{OLLAMA_API_BASE}/api/generate",
            json={"model": model, "prompt": "hi", "stream": False,
                  "options": {"num_predict": 1, "num_gpu": 0, "num_ctx": 4096}},
            timeout=60.0,
        )
        return r.status_code == 200
    except Exception:
        return False


@dataclass(frozen=True)
class LLMProvider:
    name: str                                  # "ollama" | "sarvam"
    url: str
    model: str
    auth_header: Optional[Tuple[str, str]] = None
    uses_local_gpu: bool = False               # launcher consults this to
                                               # decide whether to spin up
                                               # local Ollama

    def headers(self) -> Dict[str, str]:
        if self.auth_header is None:
            return {}
        return {self.auth_header[0]: self.auth_header[1]}

    def payload_extras(self) -> Dict:
        """Provider-specific request body fields merged into the base payload.

        Ollama honours its own `options` dict for GPU/context tuning; Sarvam
        follows OpenAI field names at the top level.
        """
        if self.name == "ollama":
            return {
                "options": {
                    "num_predict": 128,
                    "num_gpu": 0,
                    "num_ctx": 4096,
                },
            }
        if self.name == "sarvam":
            # 105B is a thinking model — it can burn 500+ tokens on internal
            # <think> reasoning before producing the spoken answer. 512 was
            # leaving nothing for actual output on longer reasoning turns,
            # causing silent responses. 1024 gives the model enough headroom.
            return {
                "max_tokens": 1024,
                "temperature": 0.5,
                "top_p": 0.9,
            }
        if self.name == "minimax":
            return {
                "max_tokens": 1024,
                "temperature": 0.7,
                "top_p": 0.9,
            }
        return {}

    def build_payload(self, system_prompt: str, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """Construct the base OpenAI-compatible request structure."""
        effective_model = self.model
        effective_system = system_prompt

        # Handle MiniMax "Non-Thinking" variants by stripping the UI suffix
        # and adding a strict "no-reasoning" instruction to the system prompt.
        if self.name == "minimax" and "-non-thinking" in self.model:
            effective_model = self.model.replace("-non-thinking", "")
            effective_system += "\n\nCRITICAL: Do NOT think or reason internally. Provide the direct answer immediately without any <think> tags or internal monologue."

        # Personality for Speed & Realism (Gemini-style: Robust, Intelligent, Human)
        effective_system += "\n\nPERSONALITY: You are Ravi, a highly intelligent and naturally conversational person. Speak like a close friend who is smart and helpful. "
        effective_system += "CRITICAL: Do NOT over-repeat movie dialogues. Use a famous dialogue ONLY if it perfectly fits the context or if the user asks. "
        effective_system += "Be concise, empathetic, and witty. Use [laughter], [sigh], [sniff] naturally to sound alive."
        
        effective_system += "\n\nSTYLE: Use fillers like 'हम्म...', 'यार...', 'मतलब...' sparingly and naturally. Keep responses short and impactful. "
        effective_system += "Use Devanagari words ONLY. No digits (use 'दो' instead of '2'), no Roman script."

        full_messages = [{"role": "system", "content": effective_system}] + messages
        return {
            "model": effective_model,
            "messages": full_messages,
            "stream": True,
        }


def resolve_provider(
    override_name: Optional[str] = None,
    override_model: Optional[str] = None,
) -> LLMProvider:
    """Resolve the active provider. Priority: explicit override > persisted
    config (provider_config.json) > env (LLM_PROVIDER/LLM_MODEL) > hardcoded
    default. The persisted file is written by /provider/switch so a user's
    runtime choice survives restarts without editing .env.

    Model resolution only reuses LLM_MODEL or the persisted model when it
    belongs to the same provider we've settled on — otherwise Sarvam could
    inherit an Ollama tag (or vice versa); in that case we fall back to the
    provider's own default model."""
    persisted = load_provider_config()
    persisted_name = persisted.get("provider")
    persisted_model = persisted.get("model")

    env_name = os.environ.get("LLM_PROVIDER", "ollama").lower()
    env_model = os.environ.get("LLM_MODEL")

    name = (override_name or persisted_name or env_name or "ollama").lower()

    if override_model is not None:
        model_override = override_model
    elif persisted_name and persisted_name == name and persisted_model:
        model_override = persisted_model
    elif name == env_name:
        model_override = env_model
    else:
        model_override = None

    if name == "sarvam":
        api_key = os.environ.get("SARVAM_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "LLM_PROVIDER=sarvam but SARVAM_API_KEY is not set. "
                "Add it to .env or export it in your shell."
            )
        return LLMProvider(
            name="sarvam",
            url=SARVAM_URL,
            model=model_override or SARVAM_DEFAULT_MODEL,
            auth_header=("Authorization", f"Bearer {api_key}"),
            uses_local_gpu=False,
        )

    if name == "minimax":
        api_key = os.environ.get("MINIMAX_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "LLM_PROVIDER=minimax but MINIMAX_API_KEY is not set. "
                "Add it to .env or export it in your shell."
            )
        return LLMProvider(
            name="minimax",
            url=MINIMAX_URL,
            model=model_override or MINIMAX_DEFAULT_MODEL,
            auth_header=("Authorization", f"Bearer {api_key}"),
            uses_local_gpu=False,
        )

    if name != "ollama":
        raise RuntimeError(
            f"Unknown LLM_PROVIDER={name!r}. Expected 'ollama', 'sarvam', or 'minimax'."
        )

    return LLMProvider(
        name="ollama",
        url=os.environ.get("OLLAMA_URL", OLLAMA_DEFAULT_URL),
        model=model_override or OLLAMA_DEFAULT_MODEL,
        auth_header=None,
        uses_local_gpu=True,
    )
