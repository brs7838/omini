import torch
import time
import numpy as np
import sys
from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from audio_utils import to_8k

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

def benchmark_rtf():
    print("--- OmniVoice Telephony (8k) Benchmark ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("\nLoading model...")
    model = OmniVoice.from_pretrained(
        "k2-fsa/OmniVoice",
        device_map=device,
        dtype=torch.float16
    )

    test_text = "यह एक परीक्षण वाक्य है जिसे हम विशेष रूप से आपके जीपीयू की आठ किलोहर्ट्ज़ पर स्पीड मापने के लिए उपयोग कर रहे हैं।"

    print(f"\nBenchmarking Text: {test_text[:40]}...")
    print(f"{'Mode':<15} | {'Gen Time (s)':<15} | {'Resample (s)':<15} | {'RTF':<10}")
    print("-" * 65)

    # Warmup
    model.generate(text="warmup", generation_config=OmniVoiceGenerationConfig(num_step=8))

    # --- 24k Benchmark ---
    t0 = time.time()
    audio_list = model.generate(text=test_text, generation_config=OmniVoiceGenerationConfig(num_step=16))
    t1 = time.time()
    audio_24k = audio_list[0]
    dur = len(audio_24k) / 24000
    rtf_24k = (t1 - t0) / dur
    print(f"{'Default (24k)':<15} | {t1-t0:<15.4f} | {'N/A':<15} | {rtf_24k:<10.4f}")

    # --- 8k Benchmark ---
    t0 = time.time()
    audio_list = model.generate(text=test_text, generation_config=OmniVoiceGenerationConfig(num_step=16))
    t_gen_done = time.time()
    audio_resampled = to_8k(audio_list[0], orig_sr=24000)
    t_end = time.time()
    
    gen_time = t_gen_done - t0
    resample_time = t_end - t_gen_done
    total_time = t_end - t0
    rtf_8k = total_time / dur
    
    print(f"{'Telephony (8k)':<15} | {gen_time:<15.4f} | {resample_time:<15.4f} | {rtf_8k:<10.4f}")

    print("-" * 65)
    overhead = ((rtf_8k / rtf_24k) - 1) * 100
    print(f"\n8kHz OverHead: {overhead:.2f}% additional time spent in resampling.")
    print(f"Final 8k RTF: {rtf_8k:.4f}")

if __name__ == "__main__":
    benchmark_rtf()
