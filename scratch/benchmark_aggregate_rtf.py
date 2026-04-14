import torch
import time
import threading
import numpy as np
import sys
from omnivoice import OmniVoice, OmniVoiceGenerationConfig

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

def worker(device, results):
    """Worker to load model and generate audio on a specific GPU."""
    try:
        print(f"[{device}] Loading model...")
        model = OmniVoice.from_pretrained(
            "k2-fsa/OmniVoice",
            device_map=device,
            dtype=torch.float16
        )
        
        test_text = "यह एक परीक्षण वाक्य है जिसे हम विशेष रूप से आपके जीपीयू की स्पीड मापने के लिए उपयोग कर रहे हैं।"
        
        # Warmup
        model.generate(text="warmup", generation_config=OmniVoiceGenerationConfig(num_step=8))
        
        print(f"[{device}] Starting generation...")
        t0 = time.time()
        audio_list = model.generate(text=test_text, generation_config=OmniVoiceGenerationConfig(num_step=16))
        t1 = time.time()
        
        gen_time = t1 - t0
        audio_duration = len(audio_list[0]) / 24000
        rtf = gen_time / audio_duration
        
        results[device] = {
            "gen_time": gen_time,
            "audio_duration": audio_duration,
            "rtf": rtf
        }
        print(f"[{device}] Done. Audio: {audio_duration:.2f}s, Time: {gen_time:.2f}s, RTF: {rtf:.4f}")
        
    except Exception as e:
        print(f"[{device}] Error: {e}")

def benchmark_aggregate():
    print("--- OmniVoice AGGREGATE DUAL-GPU THROUGHPUT TEST ---")
    devices = ["cuda:0", "cuda:1"]
    results = {}
    threads = []
    
    start_time = time.time()
    
    for dev in devices:
        t = threading.Thread(target=worker, args=(dev, results))
        threads.append(t)
        t.start()
        
    for t in threads:
        t.join()
        
    end_time = time.time()
    total_time = end_time - start_time
    
    print("\n" + "="*40)
    print("FINAL COMBINED POWER RESULTS")
    print("="*40)
    
    total_audio = 0
    for dev, res in results.items():
        print(f"GPU {dev[-1]} ({'RTX 3060' if dev=='cuda:0' else 'GTX 1650'}):")
        print(f"  - RTF: {res['rtf']:.4f}")
        total_audio += res['audio_duration']
        
    aggregate_rtf = total_time / total_audio
    streams_per_sec = total_audio / total_time
    
    print("-" * 40)
    print(f"Total Audio Generated: {total_audio:.2f} seconds")
    print(f"Total Wall Clock Time: {total_time:.2f} seconds")
    print(f"Aggregate System RTF:  {aggregate_rtf:.4f}")
    print(f"Combined Speed:        {streams_per_sec:.2f}x real-time")
    print("="*40)
    
    improvement = (1 / aggregate_rtf) / (1 / results["cuda:0"]["rtf"]) - 1
    print(f"\nPower Boost: Your system is now {improvement*100:.2f}% more powerful than with GPU 0 alone.")

if __name__ == "__main__":
    benchmark_aggregate()
