"""
Isolated TTS pitch lab — NO LLM, NO STT, NO WebSocket, NO Asterisk.

Loads OmniVoice against a reference voice, synthesizes Hindi phrases under a
sweep of diffusion configs, writes WAVs to out/, and (optionally) plays them.

Usage:
    python tts_lab/run_lab.py                                   # default sweep
    python tts_lab/run_lab.py --voice ravi_sir_8s --phrase greet --num-step 12 --play
    python tts_lab/run_lab.py --voice ravi_sir_8s --sweep 8,12,16,24,32 --phrase greet,short_1,medium_1
    python tts_lab/run_lab.py --list-voices
    python tts_lab/run_lab.py --list-phrases

Output layout:
    tts_lab/out/{voice}/{phrase}_ns{num_step}.wav
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import wave
from pathlib import Path

import numpy as np
import torch

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

REPO_ROOT = Path(__file__).resolve().parents[1]
VOICES_DIR = REPO_ROOT / "assets" / "voices"
OUT_DIR = Path(__file__).resolve().parent / "out"

sys.path.insert(0, str(REPO_ROOT))

from omnivoice.models.omnivoice import OmniVoice, OmniVoiceGenerationConfig  # noqa: E402

from tts_lab.phrases_hindi import ALL as PHRASES  # noqa: E402


def list_voices() -> list[Path]:
    if not VOICES_DIR.exists():
        return []
    return sorted(
        p for p in VOICES_DIR.iterdir()
        if p.suffix.lower() in {".mp3", ".wav", ".flac", ".ogg"}
    )


def resolve_voice(name: str) -> Path:
    """Accepts full filename, stem, or substring match."""
    voices = list_voices()
    for p in voices:
        if p.name == name or p.stem == name:
            return p
    matches = [p for p in voices if name.lower() in p.stem.lower()]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise SystemExit(f"No voice matches '{name}'. Use --list-voices.")
    raise SystemExit(
        f"Ambiguous voice '{name}'. Candidates: {[p.name for p in matches]}"
    )


def save_wav(path: Path, audio: np.ndarray, sr: int = 24000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(audio, -1.0, 1.0)
    pcm_int16 = (pcm * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm_int16.tobytes())


def play_wav(path: Path) -> None:
    """Blocking playback via winsound (Windows). Silent fallback otherwise."""
    try:
        if sys.platform == "win32":
            import winsound
            winsound.PlaySound(str(path), winsound.SND_FILENAME)
        else:
            print(f"  [play skipped: non-Windows platform, open {path} manually]")
    except Exception as e:
        print(f"  [play error: {e}]")


def load_model(device: str = "cuda:0", dtype: torch.dtype = torch.float16) -> OmniVoice:
    print(f"[lab] Loading OmniVoice on {device} dtype={dtype}...")
    t0 = time.perf_counter()
    model = OmniVoice.from_pretrained(
        "k2-fsa/OmniVoice",
        device_map=device,
        dtype=dtype,
        load_asr=False,
    )
    print(f"[lab] Model loaded in {time.perf_counter() - t0:.1f}s")
    return model


def synth(
    model: OmniVoice,
    voice_prompt,
    text: str,
    num_step: int,
    guidance_scale: float,
    t_shift: float,
) -> tuple[np.ndarray, float]:
    cfg = OmniVoiceGenerationConfig(
        num_step=num_step,
        guidance_scale=guidance_scale,
        t_shift=t_shift,
    )
    t0 = time.perf_counter()
    out = model.generate(
        text=text,
        voice_clone_prompt=voice_prompt,
        language="hi",
        generation_config=cfg,
    )
    elapsed = time.perf_counter() - t0
    audio = out[0]
    if hasattr(audio, "cpu"):
        audio = audio.cpu().numpy()
    return np.asarray(audio, dtype=np.float32), elapsed


def main() -> None:
    ap = argparse.ArgumentParser(description="TTS pitch/quality lab (OmniVoice only)")
    ap.add_argument("--voice", default="ravi_sir_8s",
                    help="Voice reference (filename, stem, or substring)")
    ap.add_argument("--phrase", default="greet,short_1,medium_1",
                    help="Comma-separated phrase keys (see --list-phrases)")
    ap.add_argument("--num-step", default=None, type=int,
                    help="Single num_step run (overrides --sweep)")
    ap.add_argument("--sweep", default="8,12,16,24,32",
                    help="Comma-separated num_step values")
    ap.add_argument("--guidance", type=float, default=2.0)
    ap.add_argument("--t-shift", type=float, default=0.1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--fp32", action="store_true", help="Use float32 instead of float16")
    ap.add_argument("--play", action="store_true",
                    help="Play each generated clip immediately after synthesis")
    ap.add_argument("--list-voices", action="store_true")
    ap.add_argument("--list-phrases", action="store_true")
    args = ap.parse_args()

    if args.list_voices:
        for p in list_voices():
            print(p.name)
        return
    if args.list_phrases:
        for k, v in PHRASES.items():
            preview = v[:60] + ("..." if len(v) > 60 else "")
            print(f"  {k:10s}  {preview}")
        return

    voice_path = resolve_voice(args.voice)
    phrase_keys = [k.strip() for k in args.phrase.split(",") if k.strip()]
    for k in phrase_keys:
        if k not in PHRASES:
            raise SystemExit(f"Unknown phrase key '{k}'. Use --list-phrases.")

    if args.num_step is not None:
        num_steps = [args.num_step]
    else:
        num_steps = [int(x) for x in args.sweep.split(",") if x.strip()]

    dtype = torch.float32 if args.fp32 else torch.float16
    model = load_model(device=args.device, dtype=dtype)

    print(f"[lab] Cloning voice from {voice_path.name}")
    voice_prompt = model.create_voice_clone_prompt(
        ref_audio=str(voice_path), preprocess_prompt=True
    )

    voice_out_dir = OUT_DIR / voice_path.stem
    rows = []
    for key in phrase_keys:
        text = PHRASES[key]
        for ns in num_steps:
            audio, elapsed = synth(
                model, voice_prompt, text, ns, args.guidance, args.t_shift
            )
            peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
            out_path = voice_out_dir / f"{key}_ns{ns}.wav"
            save_wav(out_path, audio, sr=24000)
            dur = len(audio) / 24000.0
            rtf = elapsed / max(dur, 1e-6)
            print(
                f"[synth] {key:10s} ns={ns:2d} "
                f"synth={elapsed*1000:6.0f}ms audio={dur:5.2f}s "
                f"RTF={rtf:4.2f} peak={peak:.3f} -> {out_path.relative_to(REPO_ROOT)}"
            )
            rows.append((key, ns, elapsed, dur, peak, out_path))
            if args.play:
                play_wav(out_path)

    print(f"\n[lab] Done. {len(rows)} clips written under {voice_out_dir.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
