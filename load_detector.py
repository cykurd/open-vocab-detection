"""Load pre-trained GroundingDINO or OWL-ViT for open-vocabulary detection."""
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection


def load_detector(model_name="IDEA-Research/grounding-dino-tiny"):
    """Load GroundingDINO decoder and identify tunable components."""
    print(f"Loading detector model: {model_name}")
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name)
    processor = AutoProcessor.from_pretrained(model_name)
    
    # Print model structure to identify tunable components
    print("\nModel structure:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  {name}: {param.shape}")
    
    return model, processor


if __name__ == "__main__":
    model, processor = load_detector()

