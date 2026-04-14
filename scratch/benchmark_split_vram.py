import torch
import time
import sys
import os
from omnivoice import OmniVoice, OmniVoiceGenerationConfig

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

def print_gpu_mem(label):
    print(f"\n[{label}] Memory Usage:")
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024**2
        print(f"  GPU {i}: {allocated:.2f} MiB")

def benchmark_split_vram():
    print("--- OmniVoice SPLIT-GPU (RTX 3060 + GTX 1650) RTF Benchmark ---")
    
    # We will force the model to split across both GPUs
    # Since OmniVoice is ~2GB, we'll give each GPU 1GB limit to force a split
    max_memory = {0: "1GiB", 1: "4GiB"} 
    
    print("\nLoading model with forced split across GPUs...")
    try:
        model = OmniVoice.from_pretrained(
            "k2-fsa/OmniVoice",
            device_map="auto",
            max_memory=max_memory,
            dtype=torch.float16
        )
        
        # Check where the weights ended up
        print("\nModel Layer Distribution:")
        print(f"Main Model Device: {model.device}")
        # Note: In device_map="auto", model.hf_device_map shows the per-layer mapping
        if hasattr(model, "hf_device_map"):
            counts = {0: 0, 1: 0}
            for layer, dev in model.hf_device_map.items():
                counts[dev] = counts.get(dev, 0) + 1
            print(f"Layers on GPU 0: {counts[0]}")
            print(f"Layers on GPU 1: {counts[1]}")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print_gpu_mem("After Loading")

    test_text = "यह एक परीक्षण वाक्य है जिसे हम विशेष रूप से आपके जीपीयू की स्पीड मापने के लिए उपयोग कर रहे हैं।"
    
    print(f"\nBenchmarking Text: {test_text[:40]}...")
    print(f"{'Mode':<15} | {'Gen Time (s)':<15} | {'RTF':<10}")
    print("-" * 45)

    # Warmup
    model.generate(text="warmup", generation_config=OmniVoiceGenerationConfig(num_step=8))

    # --- Benchmark ---
    t0 = time.time()
    audio_list = model.generate(text=test_text, generation_config=OmniVoiceGenerationConfig(num_step=16))
    t1 = time.time()
    
    audio_duration = len(audio_list[0]) / 24000
    gen_time = t1 - t0
    rtf = gen_time / audio_duration
    
    print(f"{'Split GPU':<15} | {gen_time:<15.4f} | {rtf:<10.4f}")
    print("-" * 45)
    print(f"Final Split-VRAM RTF: {rtf:.4f}")
    print("\nNOTE: Because data has to travel over the PCIe bus between GPUs, this is likely SLOWER than single-GPU.")

if __name__ == "__main__":
    benchmark_split_vram()
