import torch
import time
import os
import sys
from omnivoice import OmniVoice, OmniVoiceGenerationConfig

def print_memory(label):
    print(f"\n--- {label} ---")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**2
            reserved = torch.cuda.memory_reserved(i) / 1024**2
            print(f"GPU {i}: Allocated: {allocated:.2f} MiB, Reserved: {reserved:.2f} MiB")
    else:
        print("CUDA not available")

def monitor_vram():
    print_memory("Initial State")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load OmniVoice (TTS only)
    print("\nLoading OmniVoice (TTS only)...")
    model = OmniVoice.from_pretrained(
        "k2-fsa/OmniVoice",
        device_map=device,
        dtype=torch.float16
    )
    print_memory("After OmniVoice Loading")
    
    # 2. Perform one TTS generation
    print("\nGenerating TTS...")
    model.generate(text="नमस्ते, आप कैसे हैं?", generation_config=OmniVoiceGenerationConfig(num_step=8))
    print_memory("After 1 TTS Generation")
    
    # 3. Load ASR
    print("\nLoading ASR (Whisper)...")
    model.load_asr_model(model_name="openai/whisper-large-v3-turbo")
    print_memory("After ASR Loading")
    
    # 4. Perform ASR + TTS
    print("\nGenerating Pipeline (ASR+TTS)...")
    # Dummy audio 2s
    sample_rate = 16000
    duration = 2.0
    dummy_audio = torch.randn(int(sample_rate * duration))
    _ = model.transcribe((dummy_audio, sample_rate))
    _ = model.generate(text="नमस्ते", generation_config=OmniVoiceGenerationConfig(num_step=8))
    print_memory("After Full Pipeline Run")

if __name__ == "__main__":
    monitor_vram()
