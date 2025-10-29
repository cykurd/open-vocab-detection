"""Training script for the custom CLIP-Transformer model."""
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from clip_transformer_model import load_clip_transformer_detector
from data import BDD100KDataset
from losses import compute_detection_loss, compute_detection_loss_v2
from losses_simple import compute_detection_loss_simple


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
    # Loss options
    parser.add_argument("--loss_type", type=str, default="simple", choices=["simple", "v2"], help="Which loss function to use")
    parser.add_argument("--matching_strategy", type=str, default="greedy", choices=["greedy", "hungarian"], help="Assignment strategy")
    parser.add_argument("--use_soft_obj_targets", action="store_true", help="Use IoU as soft positives instead of 1.0")
    parser.add_argument("--lambda_l1", type=float, default=5.0)
    parser.add_argument("--lambda_giou", type=float, default=2.0)
    parser.add_argument("--lambda_obj", type=float, default=1.0)
    parser.add_argument("--lambda_div", type=float, default=0.5)
    parser.add_argument("--lambda_contrastive", type=float, default=0.0, help="Set >0 to enable contrastive if region embeddings are available")
    parser.add_argument("--focal_alpha", type=float, default=0.25)
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--diversity_iou_thresh", type=float, default=0.5)
    parser.add_argument("--contrastive_temperature", type=float, default=0.07)
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
            # all_labels is a list of lists (one per image in batch)
            # all_boxes is a list of lists (one per image in batch)
            all_labels_lists = sample.get("labels", [])
            all_boxes_lists = sample.get("boxes", [])
            
            # Flatten and process labels and boxes
            all_labels_flat = []
            all_boxes_flat = []
            for labels, boxes in zip(all_labels_lists, all_boxes_lists):
                for lbl, box in zip(labels, boxes):
                    all_labels_flat.append(str(lbl).strip().lower())
                    all_boxes_flat.append(box)

            # Build per-label targets: label -> list of boxes for that label
            label_to_boxes = {}
            for lbl, box in zip(all_labels_flat, all_boxes_flat):
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
                # DEBUG: Track when we skip
                if len(target_boxes) == 0 and (args.use_labels_query or not args.text_query):
                    print(f"  SKIPPED '{q}' - no targets in image")
                    continue  # skip queries without targets during supervised training

                raw_outputs = model(images, q)

                # Convert predicted boxes to pixel space to match targets
                # Model now outputs boxes in [0,1] range with sigmoid already applied
                w, h = images[0].size
                pred_boxes_norm = raw_outputs.pred_boxes  # Already in [0,1]
                scale = torch.tensor([w, h, w, h], dtype=pred_boxes_norm.dtype, device=pred_boxes_norm.device).view(1, 1, 4)
                pred_boxes_px = pred_boxes_norm * scale

                # DEBUG: Print box ranges and IoU on first step (commented out for cleaner output)
                if False and step == 1:
                    print(f"DEBUG: pred_boxes_norm range: [{pred_boxes_norm.min():.3f}, {pred_boxes_norm.max():.3f}]")
                    print(f"DEBUG: pred_boxes_px range: [{pred_boxes_px.min():.1f}, {pred_boxes_px.max():.1f}]")
                    print(f"DEBUG: target_boxes count: {len(target_boxes)}")
                    print(f"DEBUG: target_boxes (first 2): {target_boxes[:2] if len(target_boxes) >= 2 else target_boxes}")
                    print(f"DEBUG: Image size: {w}x{h}")
                    # Compute IoU manually for debugging
                    import numpy as np
                    if len(target_boxes) > 0:
                        # Check if boxes overlap at all
                        pred_np = pred_boxes_px[0, 0, :].detach().cpu().numpy()  # First pred box
                        tgt_np = np.array(target_boxes[0])  # First target box
                        print(f"DEBUG: First pred box: {pred_np}")
                        print(f"DEBUG: First target box: {tgt_np}")
                        # Manual IoU
                        inter_x1 = max(pred_np[0], tgt_np[0])
                        inter_y1 = max(pred_np[1], tgt_np[1])
                        inter_x2 = min(pred_np[2], tgt_np[2])
                        inter_y2 = min(pred_np[3], tgt_np[3])
                        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
                        pred_area = (pred_np[2] - pred_np[0]) * (pred_np[3] - pred_np[1])
                        tgt_area = (tgt_np[2] - tgt_np[0]) * (tgt_np[3] - tgt_np[1])
                        union_area = pred_area + tgt_area - inter_area
                        iou = inter_area / (union_area + 1e-7)
                        print(f"DEBUG: First pred-target IoU: {iou:.3f}")

                # Package outputs with region embeddings for contrastive loss
                class Outputs:
                    def __init__(self, pred_boxes, logits, region_embeddings=None):
                        self.pred_boxes = pred_boxes
                        self.logits = logits
                        self.region_embeddings = region_embeddings

                outputs = Outputs(pred_boxes_px, raw_outputs.logits, region_embeddings=raw_outputs.region_embeddings)
                
                # Get text embeddings for contrastive loss
                # Extract from model's text encoder (already computed)
                with torch.no_grad():
                    text_embeds = model.encode_text([q])  # [batch_size, 1, d_model]
                
                # Choose loss function
                if args.loss_type == "simple":
                    loss_out = compute_detection_loss_simple(
                        outputs,
                        targets=[{'boxes': target_boxes}],
                        lambda_l1=args.lambda_l1,
                        lambda_giou=args.lambda_giou,
                        lambda_obj=args.lambda_obj,
                        focal_alpha=args.focal_alpha,
                        focal_gamma=args.focal_gamma,
                    )
                    if isinstance(loss_out, tuple):
                        loss, metrics = loss_out
                    else:
                        loss, metrics = loss_out, {'loss_l1': 0.0, 'loss_giou': 0.0, 'loss_obj': 0.0, 'num_matches': 0}
                else:
                    # Use new v2 loss (batch-aware). Our batch is typically 1, so wrap targets in a list
                    loss_out = compute_detection_loss_v2(
                        outputs,
                        targets=[{'boxes': target_boxes}],
                        text_embeddings=text_embeds,  # Now available for contrastive loss
                        matching_strategy=args.matching_strategy,
                        lambda_l1=args.lambda_l1,
                        lambda_giou=args.lambda_giou,
                        lambda_obj=args.lambda_obj,
                        lambda_div=args.lambda_div,
                        lambda_contrastive=args.lambda_contrastive,
                        focal_alpha=args.focal_alpha,
                        focal_gamma=args.focal_gamma,
                        diversity_iou_thresh=args.diversity_iou_thresh,
                        use_soft_obj_targets=args.use_soft_obj_targets,
                        contrastive_temperature=args.contrastive_temperature,
                        return_metrics=True,
                    )
                    if isinstance(loss_out, tuple):
                        loss, metrics = loss_out
                    else:
                        loss, metrics = loss_out, {
                            'loss_total': loss_out.item(),
                            'loss_l1': 0.0, 'loss_giou': 0.0, 'loss_obj': 0.0, 'loss_div': 0.0, 'loss_contrastive': 0.0
                        }

                # DEBUG: Check why loss is zero
                if abs(loss.item()) < 1e-8 and len(target_boxes) > 0 and step < 50:
                    print(f"  ⚠️ ZERO LOSS for '{q}' with {len(target_boxes)} targets at step {step}")
                    print(f"     Components: l1={metrics.get('loss_l1',0):.6f}, giou={metrics.get('loss_giou',0):.6f}, obj={metrics.get('loss_obj',0):.6f}")
                    # Check if predictions match targets exactly (IoU = 1.0)
                    import numpy as np
                    try:
                        pred_first = pred_boxes_px[0, 0, :].detach().cpu().numpy()
                        tgt_first = np.array(target_boxes[0])
                        print(f"     First pred: {pred_first}")
                        print(f"     First target: {tgt_first}")
                        # Quick IoU check
                        inter_x1 = max(pred_first[0], tgt_first[0])
                        inter_y1 = max(pred_first[1], tgt_first[1])
                        inter_x2 = min(pred_first[2], tgt_first[2])
                        inter_y2 = min(pred_first[3], tgt_first[3])
                        inter = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
                        pred_a = (pred_first[2] - pred_first[0]) * (pred_first[3] - pred_first[1])
                        tgt_a = (tgt_first[2] - tgt_first[0]) * (tgt_first[3] - tgt_first[1])
                        iou = inter / (pred_a + tgt_a - inter + 1e-7)
                        print(f"     IoU: {iou:.4f}")
                    except:
                        pass

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=1.0)
                optimizer.step()

                step += 1
                current_loss = loss.item()
                num_matched = metrics.get('num_matches', '?')
                penalty_str = f"pen {metrics.get('penalty',0):.3f}" if 'penalty' in metrics else ""
                print(
                    f"Step {step}/{args.steps} - query='{q}' - targets={len(target_boxes)} - matches={num_matched} - "
                    f"loss: {current_loss:.4f} | l1 {metrics.get('loss_l1',0):.3f} "
                    f"giou {metrics.get('loss_giou',0):.3f} obj {metrics.get('loss_obj',0):.3f} {penalty_str}"
                )
            
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
