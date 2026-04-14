import numpy as np
from scipy.signal import resample_poly

def resample_audio(audio_np, orig_sr, target_sr):
    """
    Resamples a numpy audio array from orig_sr to target_sr.
    Using resample_poly for better quality/speed than basic resample.
    """
    if orig_sr == target_sr:
        return audio_np
    
    # Calculate GCD for polyphase resampling
    gcd = np.gcd(orig_sr, target_sr)
    up = target_sr // gcd
    down = orig_sr // gcd
    
    resampled = resample_poly(audio_np, up, down)
    return resampled.astype(np.float32)

def to_8k(audio_np, orig_sr=24000):
    """Fixed helper for telephony 8kHz conversion."""
    return resample_audio(audio_np, orig_sr, 8000)
