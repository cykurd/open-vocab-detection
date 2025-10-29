"""Proper object detection loss functions."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def giou_loss(pred_boxes, target_boxes):
    """
    Compute Generalized IoU (GIoU) loss.
    Boxes format: [x1, y1, x2, y2]
    """
    # Calculate intersection
    inter_x1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
    inter_y1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
    inter_x2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
    inter_y2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])
    
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    
    # Calculate union
    pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
    target_area = (target_boxes[:, 2] - target_boxes[:, 0]) * (target_boxes[:, 3] - target_boxes[:, 1])
    union_area = pred_area + target_area - inter_area
    
    # IoU
    iou = inter_area / (union_area + 1e-7)
    
    # Calculate enclosing box
    enclose_x1 = torch.min(pred_boxes[:, 0], target_boxes[:, 0])
    enclose_y1 = torch.min(pred_boxes[:, 1], target_boxes[:, 1])
    enclose_x2 = torch.max(pred_boxes[:, 2], target_boxes[:, 2])
    enclose_y2 = torch.max(pred_boxes[:, 3], target_boxes[:, 3])
    
    enclose_area = (enclose_x2 - enclose_x1) * (enclose_y2 - enclose_y1)
    
    # GIoU
    giou = iou - (enclose_area - union_area) / (enclose_area + 1e-7)
    
    # GIoU loss (1 - GIoU)
    loss_giou = 1 - giou
    return loss_giou.mean()


def focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    """
    Focal loss for classification.
    logits: [N, num_classes] raw logits
    targets: [N] class indices (0 for background, 1 for positive)
    """
    ce_loss = F.cross_entropy(logits, targets, reduction='none')
    pt = torch.exp(-ce_loss)
    focal_loss = alpha * (1 - pt) ** gamma * ce_loss
    return focal_loss.mean()


def compute_detection_loss(outputs, target_boxes, target_labels=None):
    """
    Compute proper object detection loss combining:
    - GIoU loss for bounding box regression
    - Focal loss for classification
    - L1 loss for box coordinates
    
    Args:
        outputs: Model outputs with pred_boxes [B, N, 4] and logits [B, N, num_classes]
        target_boxes: List of target boxes per image [[x1,y1,x2,y2], ...]
        target_labels: Optional list of labels for classification loss
    """
    pred_boxes = outputs.pred_boxes  # [B, N, 4]
    pred_logits = outputs.logits  # [B, N, num_classes]
    
    B, N, _ = pred_boxes.shape
    
    if len(target_boxes) == 0 or target_boxes is None:
        # If no targets, just penalize high confidence predictions
        pred_scores = F.softmax(pred_logits, dim=-1).max(dim=-1)[0]
        loss_no_target = pred_scores.mean() * 0.1  # Small penalty
        return loss_no_target
    
    # Convert target boxes to tensor
    target_boxes_tensor = torch.tensor(target_boxes, dtype=torch.float32, device=pred_boxes.device)
    num_targets = target_boxes_tensor.shape[0]
    
    # For now, match first K predictions to first K targets (simple matching)
    # In production, you'd use Hungarian matching
    K = min(num_targets, N)
    
    # Bounding box losses
    pred_boxes_selected = pred_boxes[0, :K]  # [K, 4]
    target_boxes_selected = target_boxes_tensor[:K]  # [K, 4]
    
    # L1 loss
    loss_l1 = F.l1_loss(pred_boxes_selected, target_boxes_selected)
    
    # GIoU loss
    loss_giou = giou_loss(pred_boxes_selected, target_boxes_selected)
    
    # Classification loss (simplified - assume positive class for matched boxes)
    if target_labels is not None and len(target_labels) > 0:
        # For simplicity, create binary targets (1 for matched, 0 for rest)
        class_targets = torch.zeros(N, dtype=torch.long, device=pred_logits.device)
        class_targets[:K] = 1
        loss_cls = focal_loss(pred_logits[0], class_targets)
    else:
        # If no labels, use softmax entropy as proxy
        pred_probs = F.softmax(pred_logits[0], dim=-1)
        loss_cls = -torch.log(pred_probs.max(dim=-1)[0] + 1e-7).mean() * 0.1
    
    # Penalty for too many high-confidence predictions (encourage sparsity)
    # Count predictions above a threshold
    high_conf_threshold = 0.3
    high_conf_count = (pred_scores > high_conf_threshold).sum()
    sparsity_penalty = torch.clamp(high_conf_count.float() - 5.0, min=0.0) * 0.1  # Penalty if > 5 boxes
    
    # Combined loss
    total_loss = loss_l1 + loss_giou + loss_cls + sparsity_penalty
    
    # Safety check
    if torch.isnan(total_loss) or torch.isinf(total_loss):
        print(f"Warning: Invalid loss detected (l1: {loss_l1.item():.4f}, giou: {loss_giou.item():.4f}, cls: {loss_cls.item():.4f}, sparsity: {sparsity_penalty.item():.4f})")
        total_loss = torch.tensor(0.0, device=pred_boxes.device, requires_grad=True)
    
    return total_loss
