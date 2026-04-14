import torch
import time
import numpy as np
import sys
from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from audio_utils import to_8k

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

def benchmark_full_pipeline():
    print("--- OmniVoice FULL PIPELINE (ASR + TTS) Benchmark ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("\nLoading Full Model (ASR + TTS)...")
    model = OmniVoice.from_pretrained(
        "k2-fsa/OmniVoice",
        device_map=device,
        dtype=torch.float16,
        load_asr=True
    )

    # 1. Create a dummy 3-second audio for ASR testing (16kHz)
    print("Preparing sample input audio...")
    sample_rate = 16000
    duration = 3.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    # Synthetic "humming" or noise to simulate audio input
    dummy_audio = 0.5 * np.sin(2 * np.pi * 440 * t) 
    audio_input = {"raw": dummy_audio.astype(np.float32), "sampling_rate": sample_rate}

    print("\nStarting Pipeline Test (ASR -> TTS)...")
    print("-" * 65)
    
    # Warmup
    _ = model._asr_pipe(audio_input, generate_kwargs={"language": "hindi"})
    _ = model.generate(text="warmup", generation_config=OmniVoiceGenerationConfig(num_step=8))

    # --- Step 1: ASR (Hearing) ---
    t0 = time.time()
    transcribed_text = model.transcribe((dummy_audio, sample_rate))
    t_asr = time.time() - t0
    if not transcribed_text:
        transcribed_text = "नमस्ते कैसे हो आप"
    
    # --- Step 2: TTS (Speaking) ---
    t1 = time.time()
    audio_list = model.generate(
        text=transcribed_text, 
        generation_config=OmniVoiceGenerationConfig(num_step=16)
    )
    t_gen_done = time.time()
    audio_8k = to_8k(audio_list[0])
    t_tts = time.time() - t1
    
    total_time = t_asr + t_tts
    audio_duration = len(audio_list[0]) / 24000
    
    print(f"1. ASR Time (Hearing):   {t_asr:.4f} s")
    print(f"2. TTS Time (Speaking):  {t_tts:.4f} s")
    print("-" * 65)
    print(f"Total Pipeline Latency:  {total_time:.4f} s")
    print(f"Generated Audio length: {audio_duration:.2f} s")
    
    rtf_pipeline = total_time / audio_duration
    print(f"Pipeline RTF:           {rtf_pipeline:.4f}")
    
    if total_time < audio_duration:
        print("\n✅ Result: PIPELINE IS FASTER THAN REAL-TIME.")
        print(f"Your system can listen and reply while the user is still thinking!")
    else:
        print("\n⚠️ Result: PIPELINE IS SLOWER THAN REAL-TIME.")
        print("You might need to use a smaller ASR model or fewer TTS steps.")

if __name__ == "__main__":
    benchmark_full_pipeline()
