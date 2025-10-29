"""Load pre-trained CLIP models from Hugging Face."""
import torch
from transformers import CLIPProcessor, CLIPModel


def load_clip(model_name="openai/clip-vit-base-patch16"):
    """Load CLIP vision and text encoders (frozen)."""
    print(f"Loading CLIP model: {model_name}")
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    
    # Freeze both encoders
    for param in model.parameters():
        param.requires_grad = False
    
    print("CLIP encoders loaded and frozen")
    return model, processor


if __name__ == "__main__":
    model, processor = load_clip()
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

