from omnivoice import OmniVoice
import torch

def count_parameters():
    print("Loading model to count parameters...")
    model = OmniVoice.from_pretrained(
        "k2-fsa/OmniVoice",
        device_map="cpu",  # Use CPU to avoid GPU OOM if model is large
        dtype=torch.float16
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e9:.2f} Billion")

if __name__ == "__main__":
    count_parameters()
