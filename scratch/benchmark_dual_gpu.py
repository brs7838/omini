import torch
import time
import numpy as np
import sys
from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from transformers import pipeline as hf_pipeline

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

def benchmark_dual_gpu():
    print("--- OmniVoice DUAL-GPU (RTX 3060 + GTX 1650) Benchmark ---")
    
    # Device Assignment
    device_tts = "cuda:0" # RTX 3060 (12GB)
    device_asr = "cuda:1" # GTX 1650 (4GB)
    
    print(f"\nLoading TTS (OmniVoice) on {device_tts}...")
    model_tts = OmniVoice.from_pretrained(
        "k2-fsa/OmniVoice",
        device_map=device_tts,
        dtype=torch.float16
    )
    
    print(f"Loading ASR (Whisper) on {device_asr}...")
    asr_pipe = hf_pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-large-v3-turbo",
        dtype=torch.float16,
        device=device_asr,
    )

    # 1. Prepare sample oral input (3 seconds)
    sample_rate = 16000
    duration = 3.0
    audio_input = np.random.randn(int(sample_rate * duration)).astype(np.float32)

    print("\nStarting Parallel Execution Test...")
    print("-" * 65)

    # Warmup
    _ = asr_pipe({"raw": audio_input, "sampling_rate": sample_rate})
    _ = model_tts.generate(text="warmup", generation_config=OmniVoiceGenerationConfig(num_step=8))

    # --- Full Cycle Simulation ---
    t_start = time.time()
    
    # Step 1: ASR on GPU 1
    t0 = time.time()
    asr_result = asr_pipe({"raw": audio_input, "sampling_rate": sample_rate})
    text = asr_result["text"].strip()
    if not text: text = "नमस्ते, आपका स्वागत है।"
    t_asr = time.time() - t0
    
    # Step 2: TTS on GPU 0
    t1 = time.time()
    _ = model_tts.generate(
        text=text, 
        generation_config=OmniVoiceGenerationConfig(num_step=16)
    )
    t_tts = time.time() - t1
    
    t_total = time.time() - t_start
    
    print(f"1. ASR Time (GTX 1650):  {t_asr:.4f} s")
    print(f"2. TTS Time (RTX 3060):  {t_tts:.4f} s")
    print("-" * 65)
    print(f"Total Latency:           {t_total:.4f} s")
    
    # Capacity Estimation
    # Now that they are on separate GPUs, GPU 0 is 100% free for TTS compute.
    print(f"\nCapacity Breakdown:")
    print(f"- GPU 0 (3060): Dedicated to TTS. Can handle multiple parallel streams via batching.")
    print(f"- GPU 1 (1650): Dedicated to ASR. Offloads the heavy transcription load.")

if __name__ == "__main__":
    benchmark_dual_gpu()
