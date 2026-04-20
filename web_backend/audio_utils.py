import os
from pydub import AudioSegment

def trim_audio_file(file_path, duration_sec=10):
    """Trims an audio file using pydub for maximum stability on Windows."""
    try:
        print(f"[Audio] Trimming {file_path} to {duration_sec}s using Pydub...")
        
        # Load audio (pydub handles mp3/wav if ffmpeg is present, otherwise simple wav)
        audio = AudioSegment.from_file(file_path)
        
        # Trim to duration (in milliseconds)
        trimmed = audio[:duration_sec * 1000]
        
        # Determine output path (.wav for best TTS compatibility)
        output_path = file_path.rsplit(".", 1)[0] + "_trimmed.wav"
        
        # Export with clean parameters
        trimmed.export(output_path, format="wav")
        
        print(f"[Audio] Trimmed file saved to {output_path}")
        return output_path
    except Exception as e:
        print(f"[Audio] Trimming failed: {e}")
        # If pydub fails (maybe missing ffmpeg), just return original and hope for the best
        return file_path

def to_8k(audio, orig_sr=24000, device="cpu"):
    """Fast resampling to 8kHz using scipy. Keep it simple and stable."""
    import numpy as np
    from scipy import signal
    
    if orig_sr == 8000:
        return audio
        
    # Standard 24k -> 8k is a simple 3:1 decimation
    if orig_sr == 24000:
        return signal.decimate(audio, 3).astype(np.float32)
    
    # Generic resample for other rates
    ratio = 8000 / orig_sr
    num_samples = int(len(audio) * ratio)
    return signal.resample(audio, num_samples).astype(np.float32)
