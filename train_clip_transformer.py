"""Training script for the custom CLIP-Transformer model."""
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from clip_transformer_model import load_clip_transformer_detector
from data import BDD100KDataset
from losses import compute_detection_loss


def collate_fn(batch):
    """Custom collate function for PIL Images."""
    return {
        "image": [item["image"] for item in batch],
        "boxes": [item["boxes"] for item in batch],
        "labels": [item["labels"] for item in batch],
        "image_id": [item["image_id"] for item in batch]
    }


def main():
    parser = argparse.ArgumentParser(description="Train CLIP-Transformer model for zero-shot detection")
    parser.add_argument("--data_dir", type=str, default="data", help="Path to BDD100K root")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"], help="Dataset split")
    parser.add_argument("--max_samples", type=int, default=16, help="Limit number of samples for quick test")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--steps", type=int, default=4, help="Number of optimization steps")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--use_10k", action="store_true", help="Use 10k subset instead of full 100k dataset")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"], help="Force device selection")
    parser.add_argument("--text_query", type=str, default=None, help="Optional fixed text query. If omitted, uses sample labels as the query")
    parser.add_argument("--use_labels_query", action="store_true", help="If set, build text query from each sample's GT labels")
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

    # Dataset
    try:
        dataset = BDD100KDataset(
            data_dir=args.data_dir, 
            split=args.split, 
            transform=None, 
            max_samples=args.max_samples, 
            use_100k=False  # Use 10k format for 10k_clean dataset
        )
        if len(dataset) == 0:
            print("No samples found. Check your data directory and file structure.")
            return
        
        # Test loading a single sample
        print("Testing data loading...")
        try:
            test_sample = dataset[0]
            print(f"✓ Sample loaded: image shape {test_sample['image'].size}, {len(test_sample.get('boxes', []))} boxes, {len(test_sample.get('labels', []))} labels")
        except Exception as e:
            print(f"Error loading test sample: {e}")
            import traceback
            traceback.print_exc()
            return
        
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Model
    try:
        print("Loading CLIP-Transformer model...")
        model, processor = load_clip_transformer_detector()
        model = model.to(device)
        print("✓ Model loaded successfully")

        # Count parameters
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"Trainable params: {trainable:,} / {total:,} ({trainable/total:.2%})")

        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
        model.train()
        print("✓ Model ready for training")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return

    # Create output directory
    output_dir = "outputs"
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    step = 0
    best_loss = float('inf')
    
    print(f"\nStarting training with text query: '{args.text_query}'")
    print("=" * 50)
    
    for sample in dataloader:
        if step >= args.steps:
            break
        try:
            images = sample["image"]
            all_labels = [str(l).strip().lower() for l in sample.get("labels", [])]
            all_boxes = sample.get("boxes", [])

            # Build per-label targets: label -> list of boxes for that label
            label_to_boxes = {}
            for lbl, box in zip(all_labels, all_boxes):
                if not lbl:
                    continue
                label_to_boxes.setdefault(lbl, []).append(box)

            # If not using per-label queries, fall back to provided query
            per_label_queries = list(label_to_boxes.keys()) if (args.use_labels_query or not args.text_query) else [args.text_query]
            if len(per_label_queries) == 0:
                per_label_queries = ["object"]

            # Train once per label/query for this image
            for q in per_label_queries:
                if step >= args.steps:
                    break

                target_boxes = label_to_boxes.get(q, []) if (args.use_labels_query or not args.text_query) else all_boxes
                if len(target_boxes) == 0 and (args.use_labels_query or not args.text_query):
                    continue  # skip queries without targets during supervised training

                raw_outputs = model(images, q)

                # Convert predicted boxes to pixel space to match targets
                w, h = images[0].size
                pred_boxes_norm = torch.sigmoid(raw_outputs.pred_boxes)
                scale = torch.tensor([w, h, w, h], dtype=pred_boxes_norm.dtype, device=pred_boxes_norm.device).view(1, 1, 4)
                pred_boxes_px = pred_boxes_norm * scale

                class Outputs:
                    def __init__(self, pred_boxes, logits):
                        self.pred_boxes = pred_boxes
                        self.logits = raw_outputs.logits

                outputs = Outputs(pred_boxes_px, raw_outputs.logits)
                loss = compute_detection_loss(outputs, target_boxes, [q] * len(target_boxes))

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=1.0)
                optimizer.step()

                step += 1
                current_loss = loss.item()
                print(f"Step {step}/{args.steps} - query='{q}' - targets={len(target_boxes)} - loss: {current_loss:.4f}")
            
            # Save checkpoint
            if step % 10 == 0 or current_loss < best_loss:
                checkpoint_path = os.path.join(checkpoint_dir, f"clip_transformer_step_{step}.pth")
                torch.save({
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': current_loss,
                    'args': args
                }, checkpoint_path)
                print(f"  Saved checkpoint: {checkpoint_path}")
                
                if current_loss < best_loss:
                    best_loss = current_loss
                    best_path = os.path.join(checkpoint_dir, "best_clip_transformer.pth")
                    torch.save({
                        'step': step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': current_loss,
                        'args': args
                    }, best_path)
                    print(f"  New best model saved: {best_path}")
            
        except Exception as e:
            print(f"Error on step {step + 1}: {e}")
            import traceback
            traceback.print_exc()
            break

    # Save final model
    final_path = os.path.join(checkpoint_dir, "final_clip_transformer.pth")
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': current_loss if 'current_loss' in locals() else float('inf'),
        'args': args
    }, final_path)
    print(f"Final model saved: {final_path}")
    print("Training completed.")


if __name__ == "__main__":
    main()
