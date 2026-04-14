import os
import torch
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import uvicorn
import io
import wave
from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice.utils.text import chunk_text_punctuation
from audio_utils import to_8k  # Local utility

app = FastAPI(title="OmniVoice Telephony API")

print("Loading OmniVoice Model...")
model = OmniVoice.from_pretrained(
    "k2-fsa/OmniVoice",
    device_map="cuda",
    dtype=torch.float16
)

# Voice clone prompt path (included in repo)
REF_AUDIO_PATH = os.path.join(os.path.dirname(__file__), "..", "assets", "voices", "ravi_sir.mp3")
print(f"Loading reference voice: {REF_AUDIO_PATH}")
voice_prompt = model.create_voice_clone_prompt(ref_audio=REF_AUDIO_PATH)

@app.post("/v1/audio/speech")
async def text_to_speech_streaming(request: Request):
    """
    OpenAI-Compatible TTS Endpoint with 8kHz Telephony support.
    """
    data = await request.json()
    text = data.get("input", "")
    target_sr = data.get("sample_rate", 8000) # Default to 8k for telephony
    
    if not text:
        return {"error": "No input text provided"}

    # Split text into sentence chunks for streaming
    sentences = chunk_text_punctuation(text, chunk_len=150)

    def generate():
        for sentence in sentences:
            if not sentence.strip():
                continue
            
            # Generate at 24kHz
            audio_list = model.generate(
                text=sentence,
                voice_clone_prompt=voice_prompt,
                generation_config=OmniVoiceGenerationConfig(num_step=16)
            )
            audio_24k = audio_list[0]
            
            # Resample IF requested (e.g. to 8k)
            if target_sr == 8000:
                audio_out = to_8k(audio_24k, orig_sr=24000)
                sr_out = 8000
            else:
                audio_out = audio_24k
                sr_out = 24000
                
            # Convert to PCM16
            pcm_data = (audio_out * 32767).astype(np.int16).tobytes()
            yield pcm_data

    return StreamingResponse(generate(), media_type="audio/pcm")

if __name__ == "__main__":
    print("Starting OmniVoice 8kHz Telephony Server on port 8002...")
    uvicorn.run(app, host="0.0.0.0", port=8002)
