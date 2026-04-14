import torch
import time
import numpy as np
import sys
from omnivoice import OmniVoice

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

def benchmark_asr():
    print("--- OmniVoice Pure ASR (Whisper) Benchmark ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("\nLoading ASR Model (Whisper Large v3 Turbo)...")
    model = OmniVoice.from_pretrained(
        "k2-fsa/OmniVoice",
        device_map=device,
        dtype=torch.float16,
        load_asr=True
    )

    # Test cases: 5s and 10s audio
    durations = [5.0, 10.0]
    sample_rate = 16000

    print("\nStarting ASR Speed Test...")
    print(f"{'Audio Dur (s)':<15} | {'ASR Time (s)':<15} | {'RTF':<10}")
    print("-" * 45)

    for duration in durations:
        t_arr = np.linspace(0, duration, int(sample_rate * duration))
        dummy_audio = 0.5 * np.sin(2 * np.pi * 440 * t_arr)
        
        # Warmup for first duration
        if duration == durations[0]:
            _ = model.transcribe((dummy_audio, sample_rate))

        t0 = time.time()
        text = model.transcribe((dummy_audio, sample_rate))
        t_asr = time.time() - t0
        
        rtf = t_asr / duration
        print(f"{duration:<15.2f} | {t_asr:<15.4f} | {rtf:<10.4f}")

    print("-" * 45)
    print(f"\nFinal Verdict: Your ASR is running at {1/rtf:.1f}x speed.")
    if rtf > 0.5:
        print("💡 Suggestion: ASR is relatively slow. Consider 'whisper-base' for < 1s latency.")

if __name__ == "__main__":
    benchmark_asr()
