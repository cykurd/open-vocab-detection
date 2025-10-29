"""Evaluate trained model on val/test sets with custom text queries and visualize results."""
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
from load_detector import load_detector
from data import BDD100KDataset


def nms(boxes, scores, iou_threshold=0.5):
    """
    Non-Maximum Suppression to remove overlapping boxes.
    
    Args:
        boxes: numpy array of shape [N, 4] in format [x1, y1, x2, y2]
        scores: numpy array of shape [N] with confidence scores
        iou_threshold: IoU threshold for suppression
        
    Returns:
        keep: indices of boxes to keep
    """
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
        # Draw label
        draw.text((x1, y1-20), f"{label}", fill=color, font=font)
    
    return image


def visualize_predictions(model, processor, dataset, device, text_query, max_samples=5, save_dir=None):
    """Create visualizations showing original image, ground truth boxes, and model predictions."""
    model.eval()
    
    # Set random seed for reproducibility but vary with time
    random.seed()
    np.random.seed()
    
    if max_samples:
        # Randomize sample selection - use numpy for better randomness
        all_indices = np.arange(len(dataset))
        np.random.shuffle(all_indices)
        indices = all_indices[:min(max_samples, len(dataset))].tolist()
        dataset = torch.utils.data.Subset(dataset, indices)
    
    # Default save directory; bucket by query to avoid overwrites across queries
    if save_dir is None:
        query_slug = re.sub(r"[^a-z0-9_\-]+", "_", text_query.strip().lower())[:50] or "query"
        save_dir = os.path.join("outputs", "visualizations", query_slug)
    
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
                inputs = processor(images=[image], text=text_query, return_tensors="pt")
                inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
                
                # Get model predictions - use raw outputs directly
                outputs = model(**inputs)
                pred_boxes = outputs.pred_boxes[0]  # [900, 4] in normalized coordinates
                pred_logits = outputs.logits[0]  # [900, 256]
                
                # Convert to pixel coordinates
                w, h = image.size
                pred_boxes_np = pred_boxes.cpu().numpy()
                pred_scores = torch.softmax(pred_logits, dim=-1).max(dim=-1)[0].cpu().numpy()
                
                # Convert normalized coordinates to pixel coordinates
                # GroundingDINO outputs are in [0,1] range
                pred_boxes_pixel = pred_boxes_np.copy()
                pred_boxes_pixel[:, [0, 2]] *= w  # x coordinates
                pred_boxes_pixel[:, [1, 3]] *= h  # y coordinates
                
                # Filter by confidence threshold
                conf_threshold = 0.3  # Higher threshold for better quality
                valid_preds = pred_scores >= conf_threshold
                
                if valid_preds.sum() > 0:
                    filtered_boxes = pred_boxes_pixel[valid_preds]
                    filtered_scores = pred_scores[valid_preds]
                    
                    # Apply Non-Maximum Suppression
                    keep_indices = nms(filtered_boxes, filtered_scores, iou_threshold=0.5)
                    filtered_boxes = filtered_boxes[keep_indices]
                    filtered_scores = filtered_scores[keep_indices]
                    
                    # Take top predictions after NMS (max 5)
                    top_k = min(5, len(filtered_boxes))
                    top_indices = np.argsort(filtered_scores)[-top_k:][::-1]
                    top_boxes = filtered_boxes[top_indices]
                    top_scores = filtered_scores[top_indices]
                else:
                    # If no high-confidence predictions, lower threshold and take top 5
                    conf_threshold = 0.2
                    valid_preds = pred_scores >= conf_threshold
                    if valid_preds.sum() > 0:
                        filtered_boxes = pred_boxes_pixel[valid_preds]
                        filtered_scores = pred_scores[valid_preds]
                        # Apply NMS
                        keep_indices = nms(filtered_boxes, filtered_scores, iou_threshold=0.5)
                        filtered_boxes = filtered_boxes[keep_indices]
                        filtered_scores = filtered_scores[keep_indices]
                        top_k = min(5, len(filtered_boxes))
                        top_indices = np.argsort(filtered_scores)[-top_k:][::-1]
                        top_boxes = filtered_boxes[top_indices]
                        top_scores = filtered_scores[top_indices]
                    else:
                        # Last resort: just take top 5 by score
                        top_k = min(5, len(pred_boxes_pixel))
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
                    # Create labels with confidence scores
                    pred_labels = [f"{score:.2f}" for score in top_scores]
                    pred_image = draw_boxes_on_image(pred_image, top_boxes.tolist(), pred_labels, color='red', thickness=2)
                axes[2].imshow(pred_image)
                axes[2].set_title(f"Model Predictions ({len(top_boxes)} boxes)")
                axes[2].axis('off')
                
                # Save individual visualization (overwrite existing)
                plt.tight_layout()
                # Use simple naming that overwrites
                save_path = os.path.join(save_dir, f"eval_{i:03d}.png")
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                print(f"  Sample {i+1}: {len(gt_boxes)} GT boxes, {len(top_boxes)} predictions")
                
            except Exception as e:
                print(f"Error processing sample {i+1}: {e}")
                continue
    
    print(f"\nVisualizations saved to: {save_dir}")
    print(f"Check the generated images to see ground truth vs predictions!")


def evaluate_with_query(model, processor, dataset, device, text_query, max_samples=None):
    """Evaluate model with a custom text query (legacy function for loss computation)."""
    model.eval()
    
    if max_samples:
        indices = list(range(min(max_samples, len(dataset))))
        dataset = torch.utils.data.Subset(dataset, indices)
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    
    total_loss = 0
    count = 0
    
    with torch.no_grad():
        for i, sample in enumerate(dataloader):
            try:
                image = sample["image"]
                text = text_query
                
                inputs = processor(images=image, text=text, return_tensors="pt")
                inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}

                outputs = model(**inputs)
                
                # Compute loss
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
    parser = argparse.ArgumentParser(description="Evaluate trained model with custom text queries and visualize results")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset root")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"], help="Dataset split")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/best_model.pth", help="Path to checkpoint file (default: outputs/checkpoints/best_model.pth)")
    parser.add_argument("--text_query", type=str, default="car person truck", help="Custom text query for zero-shot detection")
    parser.add_argument("--max_samples", type=int, default=5, help="Number of samples to visualize")
    parser.add_argument("--use_10k", action="store_true", help="Use 10k subset layout")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save visualization results (default: outputs/visualizations)")
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
    model, processor = load_detector(verbose=False)
    model = model.to(device)
    
    # Load checkpoint - try multiple possible paths
    checkpoint_path = args.checkpoint
    if not os.path.exists(checkpoint_path):
        # Try in outputs/checkpoints if not found
        if checkpoint_path.startswith("checkpoints/"):
            checkpoint_path = checkpoint_path.replace("checkpoints/", "outputs/checkpoints/")
        if not os.path.exists(checkpoint_path):
            print(f"Error: Checkpoint not found at {args.checkpoint}")
            print(f"Also tried: {checkpoint_path}")
            return
    load_checkpoint(model, checkpoint_path, device)
    
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
    
    # Create visualizations
    print(f"\nCreating visualizations...")
    visualize_predictions(
        model, processor, dataset, device, 
        args.text_query, args.max_samples, args.save_dir
    )
    
    # Also compute average loss for comparison
    print(f"\nComputing average loss...")
    loss, count = evaluate_with_query(model, processor, dataset, device, args.text_query, args.max_samples)
    print(f"Average L1 loss: {loss:.4f} (over {count} samples)")
    
    print("\nEvaluation completed!")


if __name__ == "__main__":
    main()