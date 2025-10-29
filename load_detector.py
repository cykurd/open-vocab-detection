"""Load pre-trained GroundingDINO or OWL-ViT for open-vocabulary detection."""
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection


def load_detector(model_name="IDEA-Research/grounding-dino-tiny", verbose=False):
    """Load GroundingDINO decoder and (optionally) print tunable components."""
    if verbose:
        print(f"Loading detector model: {model_name}")
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name)
    processor = AutoProcessor.from_pretrained(model_name)

    if verbose:
        print("\nModel structure:")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"  {name}: {param.shape}")

    return model, processor


if __name__ == "__main__":
    model, processor = load_detector()

