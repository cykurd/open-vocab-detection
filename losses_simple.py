"""Simple object detection loss function - keeps only what matters."""
import torch
import torch.nn.functional as F


def giou_loss(pred_boxes, target_boxes):
    """Compute Generalized IoU (GIoU) loss."""
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


def focal_loss_with_logits(logits, targets, alpha=0.25, gamma=2.0):
    """Focal loss for objectness."""
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    pt = torch.exp(-bce)
    focal = alpha * (1 - pt) ** gamma * bce
    return focal.mean()


def compute_detection_loss_simple(
    outputs,
    targets,
    lambda_l1=5.0,
    lambda_giou=2.0,
    lambda_obj=1.0,
    focal_alpha=0.25,
    focal_gamma=2.0,
    penalty_no_match=10.0,
):
    """
    Simple detection loss:
    1. Match predictions to targets (greedy IoU)
    2. L1 + GIoU loss on matched pairs
    3. Focal loss for objectness (matched=1, unmatched=0)
    
    Args:
        outputs: object with pred_boxes [B, N, 4] and logits [B, N, 1]
        targets: list of B dicts with 'boxes' key, or list of B tensors [T_i, 4]
    """
    pred_boxes = outputs.pred_boxes  # [B, N, 4]
    obj_logits = outputs.logits.squeeze(-1)  # [B, N]
    device = pred_boxes.device
    
    B, N = pred_boxes.shape[:2]
    
    # Normalize targets
    target_tensors = []
    for b in range(B):
        tgt = targets[b] if isinstance(targets, list) else targets
        if isinstance(tgt, dict):
            boxes = tgt['boxes']
        else:
            boxes = tgt
        
        # Convert to tensor
        if isinstance(boxes, list):
            boxes = torch.tensor(boxes, dtype=torch.float32, device=device)
        else:
            boxes = boxes.to(device)
        
        # Ensure valid boxes (x1 < x2, y1 < y2)
        if boxes.numel() > 0:
            x1y1 = torch.min(boxes[:, 0:2], boxes[:, 2:4])
            x2y2 = torch.max(boxes[:, 0:2], boxes[:, 2:4])
            boxes = torch.cat([x1y1, x2y2], dim=1)
        
        target_tensors.append(boxes)
    
    # Compute losses per image
    loss_l1 = torch.zeros((), device=device)
    loss_giou = torch.zeros((), device=device)
    num_matches = 0
    
    # Objectness targets: 0 for all predictions initially
    obj_targets = torch.zeros((B, N), dtype=torch.float32, device=device)
    
    for b in range(B):
        boxes_b = pred_boxes[b]  # [N, 4]
        tgt_b = target_tensors[b]  # [T, 4]
        
        if tgt_b.numel() == 0:
            # No targets: all predictions should have 0 objectness
            continue
        
        # Greedy IoU matching
        T = tgt_b.shape[0]
        
        # Compute IoU matrix: [N, T]
        x11, y11, x12, y12 = boxes_b[:, 0], boxes_b[:, 1], boxes_b[:, 2], boxes_b[:, 3]
        x21, y21, x22, y22 = tgt_b[:, 0], tgt_b[:, 1], tgt_b[:, 2], tgt_b[:, 3]
        
        inter_x1 = torch.max(x11[:, None], x21[None, :])
        inter_y1 = torch.max(y11[:, None], y21[None, :])
        inter_x2 = torch.min(x12[:, None], x22[None, :])
        inter_y2 = torch.min(y12[:, None], y22[None, :])
        
        inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
        area1 = (x12 - x11).clamp(min=0) * (y12 - y11).clamp(min=0)
        area2 = (x22 - x21).clamp(min=0) * (y22 - y21).clamp(min=0)
        union = area1[:, None] + area2[None, :] - inter + 1e-7
        iou_mat = inter / union  # [N, T]
        
        # Greedy matching: assign each target to best unmatched prediction
        matched_pred_idx = []
        matched_gt_idx = []
        available = torch.ones(N, dtype=torch.bool, device=device)
        
        for gt_idx in range(T):
            if not available.any():
                break
            ious = iou_mat[:, gt_idx].clone()
            ious[~available] = -1.0
            best_pred_idx = int(torch.argmax(ious).item())
            best_iou = float(ious[best_pred_idx].item())
            
            if best_iou > 0:  # Only match if IoU > 0
                matched_pred_idx.append(best_pred_idx)
                matched_gt_idx.append(gt_idx)
                available[best_pred_idx] = False
                obj_targets[b, best_pred_idx] = 1.0  # Positive target
        
        # Compute regression losses on matched pairs
        if matched_pred_idx:
            pred_matched = boxes_b[matched_pred_idx]
            tgt_matched = tgt_b[matched_gt_idx]
            loss_l1 = loss_l1 + F.l1_loss(pred_matched, tgt_matched)
            loss_giou = loss_giou + giou_loss(pred_matched, tgt_matched)
            num_matches += len(matched_pred_idx)
    
    # Normalize regression losses by number of matches
    if num_matches > 0:
        loss_l1 = loss_l1 / num_matches
        loss_giou = loss_giou / num_matches
        penalty = torch.tensor(0.0, device=device)
    else:
        # No matches: penalize high objectness predictions (they should be low if there are no objects)
        # This encourages the model to predict low confidence when there are no targets
        loss_l1 = torch.tensor(0.0, device=device)
        loss_giou = torch.tensor(0.0, device=device)
        # Penalize predictions with high objectness when there are no matches
        penalty = torch.sigmoid(obj_logits).mean() * penalty_no_match
    
    # Objectness focal loss
    loss_obj = focal_loss_with_logits(obj_logits, obj_targets, alpha=focal_alpha, gamma=focal_gamma)
    
    # Total loss
    total_loss = lambda_l1 * loss_l1 + lambda_giou * loss_giou + lambda_obj * loss_obj + penalty
    
    # Return loss and metrics
    metrics = {
        'loss_l1': float(loss_l1.item()) if isinstance(loss_l1, torch.Tensor) else loss_l1,
        'loss_giou': float(loss_giou.item()) if isinstance(loss_giou, torch.Tensor) else loss_giou,
        'loss_obj': float(loss_obj.item()) if isinstance(loss_obj, torch.Tensor) else loss_obj,
        'penalty': float(penalty.item()) if isinstance(penalty, torch.Tensor) else penalty,
        'num_matches': num_matches,
    }
    
    return total_loss, metrics

