from omnivoice import OmniVoice
import torch
import os

def download_models():
    print("Downloading/Loading OmniVoice model...")
    try:
        model = OmniVoice.from_pretrained(
            "k2-fsa/OmniVoice",
            device_map="cuda:0",
            dtype=torch.float16
        )
        print("Model downloaded and loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")

if __name__ == "__main__":
    download_models()
