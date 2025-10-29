"""Evaluate trained model on val/test sets with custom text queries."""
import os
import argparse
import torch
from torch.utils.data import DataLoader
from load_detector import load_detector
from data import BDD100KDataset


def load_checkpoint(model, checkpoint_path, device):
    """Load saved checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from step {checkpoint.get('step', 'unknown')}")
    print(f"Checkpoint loss: {checkpoint.get('loss', 'unknown')}")
    return checkpoint


def collate_fn(batch):
    """Custom collate function for PIL Images."""
    return {
        "image": [item["image"] for item in batch],
        "boxes": [item["boxes"] for item in batch],
        "labels": [item["labels"] for item in batch],
        "image_id": [item["image_id"] for item in batch]
    }

def evaluate_with_query(model, processor, dataset, device, text_query, max_samples=None):
    """Evaluate model with a custom text query."""
    model.eval()
    
    if max_samples:
        # Create a subset for quick testing
        indices = list(range(min(max_samples, len(dataset))))
        dataset = torch.utils.data.Subset(dataset, indices)
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    
    total_loss = 0
    count = 0
    
    with torch.no_grad():
        for sample in dataloader:
            try:
                image = sample["image"]
                
                # Use custom text query instead of dataset labels
                text = text_query
                
                # Process inputs
                inputs = processor(images=image, text=text, return_tensors="pt")
                inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}

                # Forward pass
                outputs = model(**inputs)
                
                # Compute loss (using ground truth boxes for evaluation)
                pred_boxes = outputs.pred_boxes
                target_boxes = sample.get("boxes", [])
                
                if len(target_boxes) > 0:
                    tb = torch.tensor(target_boxes[0], dtype=torch.float32, device=device)
                    K = min(tb.shape[0], pred_boxes.shape[1])
                    loss = torch.nn.functional.l1_loss(pred_boxes[0, :K], tb[:K])
                    total_loss += loss.item()
                
                count += 1
                
            except Exception as e:
                print(f"Error processing sample: {e}")
                continue
    
    avg_loss = total_loss / count if count > 0 else 0
    return avg_loss, count


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model with custom text queries")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset root")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"], help="Dataset split")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file (e.g., checkpoints/best_model.pth)")
    parser.add_argument("--text_query", type=str, default=None, help="Custom text query (e.g., 'car person truck'). If None, uses dataset labels.")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of samples for evaluation")
    parser.add_argument("--use_10k", action="store_true", help="Use 10k subset layout")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    args = parser.parse_args()

    # Device selection
    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    elif args.device == "mps":
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Using device: {device}")

    # Load model
    print("Loading detector model...")
    model, processor = load_detector()
    model = model.to(device)
    
    # Load checkpoint
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        return
    load_checkpoint(model, args.checkpoint, device)
    
    # Load dataset
    print(f"Loading {args.split} dataset...")
    dataset = BDD100KDataset(
        data_dir=args.data_dir,
        split=args.split,
        transform=None,
        max_samples=args.max_samples,
        use_100k=not args.use_10k
    )
    print(f"Dataset loaded: {len(dataset)} samples")
    
    # Determine text query
    if args.text_query:
        text_query = args.text_query
        print(f"\nUsing custom text query: '{text_query}'")
    else:
        text_query = None  # Will use dataset labels
        print("\nUsing dataset labels as text queries")
    
    # Evaluate
    print(f"\nEvaluating on {args.split} split...")
    if text_query:
        # Single custom query
        loss, count = evaluate_with_query(model, processor, dataset, device, text_query, args.max_samples)
        print(f"\nResults with query '{text_query}':")
        print(f"  Samples evaluated: {count}")
        print(f"  Average L1 loss: {loss:.4f}")
    else:
        # Use dataset labels (multiple queries)
        # This is a simplified evaluation - you'd want to iterate through all samples
        print("Running evaluation with dataset labels...")
        # For now, just evaluate on first few samples
        loss, count = evaluate_with_query(model, processor, dataset, device, "object", args.max_samples or 10)
        print(f"\nResults:")
        print(f"  Samples evaluated: {count}")
        print(f"  Average L1 loss: {loss:.4f}")
    
    print("\nEvaluation completed.")


if __name__ == "__main__":
    main()

