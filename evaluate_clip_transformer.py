"""Evaluate CLIP-Transformer model with custom text queries and visualize results."""
import os
import argparse
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import re
import random
from torch.utils.data import DataLoader
from clip_transformer_model import load_clip_transformer_detector
from data import BDD100KDataset


def nms(boxes, scores, iou_threshold=0.5):
    """Non-Maximum Suppression to remove overlapping boxes."""
    if len(boxes) == 0:
        return np.array([], dtype=np.int32)
    
    # Convert to corners
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    # Calculate areas
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    
    # Sort by score (descending)
    order = scores.argsort()[::-1]
    
    keep = []
    while len(order) > 0:
        # Take the highest scoring box
        i = order[0]
        keep.append(i)
        
        if len(order) == 1:
            break
        
        # Calculate IoU with remaining boxes
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        intersection = w * h
        
        iou = intersection / (areas[i] + areas[order[1:]] - intersection)
        
        # Keep boxes with IoU below threshold
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    
    return np.array(keep, dtype=np.int32)


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


def draw_boxes_on_image(image, boxes, labels, color='red', thickness=2):
    """Draw bounding boxes on PIL image."""
    draw = ImageDraw.Draw(image)
    
    # Try to load a font, fallback to default if not available
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    for i, (box, label) in enumerate(zip(boxes, labels)):
        x1, y1, x2, y2 = box
        
        # Validate and fix bounding box coordinates
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        
        # Skip invalid boxes (zero area or negative coordinates)
        if x1 >= x2 or y1 >= y2 or x1 < 0 or y1 < 0:
            continue
            
        # Ensure coordinates are within image bounds
        img_width, img_height = image.size
        x1 = max(0, min(x1, img_width))
        y1 = max(0, min(y1, img_height))
        x2 = max(0, min(x2, img_width))
        y2 = max(0, min(y2, img_height))
        
        # Skip if box becomes invalid after clipping
        if x1 >= x2 or y1 >= y2:
            continue
        
        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline=color, width=thickness)
        # Draw label in top-right corner of the box
        try:
            # PIL >= 8
            text_bbox = draw.textbbox((0, 0), f"{label}", font=font)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]
        except Exception:
            try:
                text_w, text_h = font.getsize(f"{label}")
            except Exception:
                text_w, text_h = (len(str(label)) * 7, 12)
        tx = max(0, min(x2 - text_w, image.size[0] - text_w))
        ty = max(0, y1 - text_h - 2)
        draw.text((tx, ty), f"{label}", fill=color, font=font)
    
    return image


def visualize_predictions(model, dataset, device, text_query, max_samples=5, save_dir=None):
    """Create visualizations showing original image, ground truth boxes, and model predictions."""
    model.eval()
    
    # Set random seed for reproducibility but vary with time
    random.seed()
    np.random.seed()
    
    if max_samples:
        # Randomize sample selection
        all_indices = np.arange(len(dataset))
        np.random.shuffle(all_indices)
        indices = all_indices[:min(max_samples, len(dataset))].tolist()
        dataset = torch.utils.data.Subset(dataset, indices)
    
    # Default save directory
    if save_dir is None:
        query_slug = re.sub(r"[^a-z0-9_\-]+", "_", text_query.strip().lower())[:50] or "query"
        save_dir = os.path.join("outputs", "visualizations", f"clip_transformer_{query_slug}")
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    
    # Create output directory
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Creating visualizations for {len(dataset)} samples...")
    print(f"Text query: '{text_query}'")
    print(f"Saving results to: {save_dir}")
    
    with torch.no_grad():
        for i, sample in enumerate(dataloader):
            try:
                # Get original image and ground truth
                image = sample["image"][0]  # PIL Image
                gt_boxes = sample["boxes"][0]  # List of [x1, y1, x2, y2]
                gt_labels = sample["labels"][0]  # List of labels
                image_id = sample["image_id"][0]
                
                # Process with model
                outputs = model([image], text_query)
                pred_boxes = outputs.pred_boxes[0]  # [num_queries, 4]
                pred_logits = outputs.logits[0]  # [num_queries, 1]

                # Normalize boxes to [0,1] then convert to pixel coordinates
                # Our head outputs unconstrained values; sigmoid brings them into [0,1]
                w, h = image.size
                pred_boxes_norm = torch.sigmoid(pred_boxes)
                pred_boxes_np = pred_boxes_norm.cpu().numpy()
                pred_scores = torch.sigmoid(pred_logits).squeeze(-1).cpu().numpy()  # [num_queries]

                # Convert to pixel coordinates (assume [x1,y1,x2,y2] in [0,1])
                pred_boxes_pixel = pred_boxes_np.copy()
                pred_boxes_pixel[:, [0, 2]] *= w
                pred_boxes_pixel[:, [1, 3]] *= h

                # Ensure x1<x2, y1<y2
                x1 = np.minimum(pred_boxes_pixel[:, 0], pred_boxes_pixel[:, 2])
                y1 = np.minimum(pred_boxes_pixel[:, 1], pred_boxes_pixel[:, 3])
                x2 = np.maximum(pred_boxes_pixel[:, 0], pred_boxes_pixel[:, 2])
                y2 = np.maximum(pred_boxes_pixel[:, 1], pred_boxes_pixel[:, 3])
                pred_boxes_pixel = np.stack([x1, y1, x2, y2], axis=1)
                
                # Filter by confidence threshold
                conf_threshold = 0.2
                valid_preds = pred_scores >= conf_threshold
                
                if valid_preds.sum() > 0:
                    filtered_boxes = pred_boxes_pixel[valid_preds]
                    filtered_scores = pred_scores[valid_preds]
                    
                    # Apply Non-Maximum Suppression (stricter to reduce overlaps)
                    keep_indices = nms(filtered_boxes, filtered_scores, iou_threshold=0.3)
                    filtered_boxes = filtered_boxes[keep_indices]
                    filtered_scores = filtered_scores[keep_indices]
                    
                    # Take top predictions after NMS (max 3)
                    top_k = min(3, len(filtered_boxes))
                    top_indices = np.argsort(filtered_scores)[-top_k:][::-1]
                    top_boxes = filtered_boxes[top_indices]
                    top_scores = filtered_scores[top_indices]
                else:
                    # If no high-confidence predictions, lower threshold and take top 3
                    conf_threshold = 0.3
                    valid_preds = pred_scores >= conf_threshold
                    if valid_preds.sum() > 0:
                        filtered_boxes = pred_boxes_pixel[valid_preds]
                        filtered_scores = pred_scores[valid_preds]
                        # Apply NMS
                        keep_indices = nms(filtered_boxes, filtered_scores, iou_threshold=0.3)
                        filtered_boxes = filtered_boxes[keep_indices]
                        filtered_scores = filtered_scores[keep_indices]
                        top_k = min(3, len(filtered_boxes))
                        top_indices = np.argsort(filtered_scores)[-top_k:][::-1]
                        top_boxes = filtered_boxes[top_indices]
                        top_scores = filtered_scores[top_indices]
                    else:
                        # Last resort: just take top 3 by score
                        top_k = min(3, len(pred_boxes_pixel))
                        top_indices = np.argsort(pred_scores)[-top_k:][::-1]
                        top_boxes = pred_boxes_pixel[top_indices]
                        top_scores = pred_scores[top_indices]
                
                # Create visualization
                fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                fig.suptitle(f"Image: {image_id}\nQuery: '{text_query}'", fontsize=14)
                
                # Original image
                axes[0].imshow(image)
                axes[0].set_title("Original Image")
                axes[0].axis('off')
                
                # Ground truth boxes
                gt_image = image.copy()
                if gt_boxes:
                    gt_image = draw_boxes_on_image(gt_image, gt_boxes, gt_labels, color='green', thickness=3)
                axes[1].imshow(gt_image)
                axes[1].set_title(f"Ground Truth ({len(gt_boxes)} boxes)")
                axes[1].axis('off')
                
                # Model predictions
                pred_image = image.copy()
                if len(top_boxes) > 0:
                    # Create labels with query name and confidence
                    pred_labels = [f"{text_query} {score:.2f}" for score in top_scores]
                    pred_image = draw_boxes_on_image(pred_image, top_boxes.tolist(), pred_labels, color='red', thickness=2)
                axes[2].imshow(pred_image)
                axes[2].set_title(f"CLIP-Transformer Predictions ({len(top_boxes)} boxes)")
                axes[2].axis('off')
                
                # Save individual visualization
                plt.tight_layout()
                save_path = os.path.join(save_dir, f"eval_{i:03d}.png")
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                print(f"  Sample {i+1}: {len(gt_boxes)} GT boxes, {len(top_boxes)} predictions")
                
            except Exception as e:
                print(f"Error processing sample {i+1}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    print(f"\nVisualizations saved to: {save_dir}")
    print(f"Check the generated images to see ground truth vs predictions!")


def main():
    parser = argparse.ArgumentParser(description="Evaluate CLIP-Transformer model with custom text queries")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset root")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"], help="Dataset split")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/best_clip_transformer.pth", help="Path to checkpoint file")
    parser.add_argument("--text_query", type=str, default=None, help="Optional fixed text query. If omitted and --use_labels_query set, uses each sample's labels")
    parser.add_argument("--use_labels_query", action="store_true", help="If set, build the text query from each sample's GT labels during eval")
    parser.add_argument("--auto_top_k_labels", type=int, default=0, help="If >0, auto-compute top-K frequent labels and evaluate each label as its own query")
    parser.add_argument("--max_samples", type=int, default=5, help="Number of samples to visualize")
    parser.add_argument("--use_10k", action="store_true", help="Use 10k subset layout")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save visualization results")
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
    print("Loading CLIP-Transformer model...")
    model, processor = load_clip_transformer_detector()
    model = model.to(device)
    
    # Load checkpoint if provided
    if args.checkpoint and os.path.exists(args.checkpoint):
        load_checkpoint(model, args.checkpoint, device)
    else:
        print(f"Warning: Checkpoint not found at {args.checkpoint}, using untrained model")
    
    # Load dataset
    print(f"Loading {args.split} dataset...")
    dataset = BDD100KDataset(
        data_dir=args.data_dir,
        split=args.split,
        transform=None,
        max_samples=args.max_samples,
        use_100k=False  # Use 10k format for 10k_clean dataset
    )
    print(f"Dataset loaded: {len(dataset)} samples")
    
    # Create visualizations
    print(f"\nCreating visualizations...")
    # If using labels as queries, loop per-sample (visualize_predictions expects a single text_query)
    if args.use_labels_query and args.text_query is None:
        # Evaluate with queries derived per-sample by wrapping a tiny adapter around visualize loop
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
        os.makedirs(args.save_dir or os.path.join("outputs", "visualizations", "labels_query"), exist_ok=True)
        shown = 0
        with torch.no_grad():
            for sample in dataloader:
                labels = sample.get("labels", [[]])[0]
                unique_labels = sorted({str(l).strip().lower() for l in labels if l})
                text_query = " ".join(unique_labels[:10]) if unique_labels else "object"
                visualize_predictions(model, torch.utils.data.Subset(dataset, [shown]), device, text_query, 1, args.save_dir)
                shown += 1
                if args.max_samples and shown >= args.max_samples:
                    break
    elif args.auto_top_k_labels and args.auto_top_k_labels > 0:
        # Compute top-K labels by frequency across the dataset
        print(f"\nComputing top-{args.auto_top_k_labels} frequent labels...")
        from collections import Counter
        label_counter = Counter()
        for idx in range(len(dataset)):
            ann = dataset[idx]
            for l in ann.get("labels", []):
                if l:
                    label_counter[str(l).strip().lower()] += 1
        top_labels = [l for l, _ in label_counter.most_common(args.auto_top_k_labels)]
        print(f"Top labels: {top_labels}")

        # For each label, build a subset of images that contain it and visualize
        from torch.utils.data import Subset
        import itertools
        for label in top_labels:
            indices_with_label = [i for i in range(len(dataset)) if label in set(str(x).strip().lower() for x in dataset[i].get("labels", []))]
            if len(indices_with_label) == 0:
                continue
            sub_indices = indices_with_label[: (args.max_samples or 10)]
            print(f"\nEvaluating label '{label}' on {len(sub_indices)} samples...")
            visualize_predictions(
                model,
                Subset(dataset, sub_indices),
                device,
                label,
                len(sub_indices),
                args.save_dir,
            )
    else:
        visualize_predictions(
            model, dataset, device, 
            args.text_query or "object", args.max_samples, args.save_dir
        )
    
    print("\nEvaluation completed!")


if __name__ == "__main__":
    main()
