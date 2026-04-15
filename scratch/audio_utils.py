import torch
import torchaudio

class CachedResampler:
    """Caches torchaudio resamplers to avoid initialization overhead."""
    _resamplers = {}

    @classmethod
    def get_resampler(cls, orig_sr, target_sr, device="cpu"):
        key = (orig_sr, target_sr, device)
        if key not in cls._resamplers:
            import torchaudio
            cls._resamplers[key] = torchaudio.transforms.Resample(orig_sr, target_sr).to(device)
        return cls._resamplers[key]

def resample_audio(audio_np, orig_sr, target_sr, device="cpu"):
    if orig_sr == target_sr:
        return audio_np
    
    waveform = torch.from_numpy(audio_np).float().to(device)
    resampler = CachedResampler.get_resampler(orig_sr, target_sr, device)
    resampled = resampler(waveform)
    return resampled.cpu().numpy()

def to_8k(audio_np, orig_sr=24000, device="cpu"):
    return resample_audio(audio_np, orig_sr, 8000, device=device)
