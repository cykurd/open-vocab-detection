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
    pred_boxes = outputs.pred_boxes  # [B, N, 4] (pixel space)
    pred_obj_logits = outputs.logits[..., 0]  # [B, N] objectness logit

    B, N, _ = pred_boxes.shape

    # Convert targets
    target_boxes_tensor = torch.tensor(target_boxes or [], dtype=torch.float32, device=pred_boxes.device)
    num_targets = target_boxes_tensor.shape[0]

    # Default objectness targets are zeros (negatives)
    obj_targets = torch.zeros(N, dtype=torch.float32, device=pred_boxes.device)

    # If no targets, just suppress objectness
    if num_targets == 0:
        bce = F.binary_cross_entropy_with_logits(pred_obj_logits[0], obj_targets)
        return bce * 0.5

    # Greedy IoU matching between predictions and GT
    with torch.no_grad():
        # Compute IoU matrix [N, T]
        x11, y11, x12, y12 = pred_boxes[0, :, 0], pred_boxes[0, :, 1], pred_boxes[0, :, 2], pred_boxes[0, :, 3]
        x21, y21, x22, y22 = target_boxes_tensor[:, 0], target_boxes_tensor[:, 1], target_boxes_tensor[:, 2], target_boxes_tensor[:, 3]
        inter_x1 = torch.max(x11[:, None], x21[None, :])
        inter_y1 = torch.max(y11[:, None], y21[None, :])
        inter_x2 = torch.min(x12[:, None], x22[None, :])
        inter_y2 = torch.min(y12[:, None], y22[None, :])
        inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
        area1 = (x12 - x11).clamp(min=0) * (y12 - y11).clamp(min=0)
        area2 = (x22 - x21).clamp(min=0) * (y22 - y21).clamp(min=0)
        union = area1[:, None] + area2[None, :] - inter + 1e-7
        iou_mat = inter / union

    matched_pred_indices = []
    matched_gt_indices = []
    available_mask = torch.ones(N, dtype=torch.bool, device=pred_boxes.device)
    for gi in range(num_targets):
        if not available_mask.any():
            break
        ious_col = iou_mat[:, gi]  # [N]
        ious_col = ious_col.masked_fill(~available_mask, -1.0)
        best_pi = int(torch.argmax(ious_col).item())
        best_val = float(ious_col[best_pi].item())
        if best_val > 0:
            matched_pred_indices.append(best_pi)
            matched_gt_indices.append(gi)
            available_mask[best_pi] = False

    # Regression losses on matched pairs
    if matched_pred_indices:
        pred_boxes_selected = pred_boxes[0, matched_pred_indices]
        target_boxes_selected = target_boxes_tensor[matched_gt_indices]
        loss_l1 = F.l1_loss(pred_boxes_selected, target_boxes_selected)
        loss_giou = giou_loss(pred_boxes_selected, target_boxes_selected)
        # Objectness targets use IoU for calibration
        obj_targets[matched_pred_indices] = iou_mat[matched_pred_indices, matched_gt_indices].clamp(0, 1)
    else:
        loss_l1 = torch.tensor(0.0, device=pred_boxes.device)
        loss_giou = torch.tensor(0.0, device=pred_boxes.device)

    # Overlap penalty among matched preds (diversity)
    if len(matched_pred_indices) > 1:
        pb = pred_boxes[0, matched_pred_indices]
        x1 = torch.max(pb[:, None, 0], pb[None, :, 0])
        y1 = torch.max(pb[:, None, 1], pb[None, :, 1])
        x2 = torch.min(pb[:, None, 2], pb[None, :, 2])
        y2 = torch.min(pb[:, None, 3], pb[None, :, 3])
        inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
        area = (pb[:, 2] - pb[:, 0]).clamp(min=0) * (pb[:, 3] - pb[:, 1]).clamp(min=0)
        union = area[:, None] + area[None, :] - inter + 1e-7
        iou_mat = inter / union
        # exclude diagonal
        overlap_penalty = (iou_mat - torch.diag(torch.diag(iou_mat))).mean() * 0.5
    else:
        overlap_penalty = torch.tensor(0.0, device=pred_boxes.device)

    # Binary objectness loss for all predictions
    bce = F.binary_cross_entropy_with_logits(pred_obj_logits[0], obj_targets, reduction='mean')

    # Combined loss
    total_loss = loss_l1 + loss_giou + bce + overlap_penalty
    
    # Safety check
    if torch.isnan(total_loss) or torch.isinf(total_loss):
        print(f"Warning: Invalid loss detected (l1: {loss_l1.item():.4f}, giou: {loss_giou.item():.4f}, bce: {bce.item():.4f}, overlap: {overlap_penalty.item():.4f})")
        total_loss = torch.tensor(0.0, device=pred_boxes.device, requires_grad=True)
    
    return total_loss


def binary_focal_loss_with_logits(logits, targets, alpha=0.25, gamma=2.0, reduction='mean'):
    """
    Binary focal loss for logits.
    logits: [*,] raw logits
    targets: [*,] in {0,1}
    """
    prob = torch.sigmoid(logits)
    pt = torch.where(targets == 1, prob, 1 - prob).clamp(min=1e-7, max=1 - 1e-7)
    loss = -alpha * (1 - pt) ** gamma * torch.log(pt)
    if reduction == 'mean':
        return loss.mean()
    if reduction == 'sum':
        return loss.sum()
    return loss


def _ensure_boxes_tensor_per_image(boxes, device):
    """
    Normalize different target formats to a tensor of shape [T,4] on device.
    Accepts: Tensor, list[list[float]], list[Tensor]
    """
    if boxes is None:
        return torch.zeros((0, 4), dtype=torch.float32, device=device)
    if isinstance(boxes, torch.Tensor):
        t = boxes.to(device=device, dtype=torch.float32)
    else:
        # list-like
        t = torch.tensor(boxes, dtype=torch.float32, device=device)
    if t.numel() == 0:
        return t.reshape(0, 4)
    return t.view(-1, 4)


def _greedy_match_iou(pred_boxes_img, target_boxes_img):
    """
    Greedy IoU matching for a single image.
    pred_boxes_img: [N,4], target_boxes_img: [T,4]
    Returns (matched_pred_indices, matched_gt_indices, iou_matrix)
    """
    N = pred_boxes_img.shape[0]
    T = target_boxes_img.shape[0]
    if T == 0 or N == 0:
        return [], [], None

    x11, y11, x12, y12 = pred_boxes_img[:, 0], pred_boxes_img[:, 1], pred_boxes_img[:, 2], pred_boxes_img[:, 3]
    x21, y21, x22, y22 = target_boxes_img[:, 0], target_boxes_img[:, 1], target_boxes_img[:, 2], target_boxes_img[:, 3]

    inter_x1 = torch.max(x11[:, None], x21[None, :])
    inter_y1 = torch.max(y11[:, None], y21[None, :])
    inter_x2 = torch.min(x12[:, None], x22[None, :])
    inter_y2 = torch.min(y12[:, None], y22[None, :])
    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    area1 = (x12 - x11).clamp(min=0) * (y12 - y11).clamp(min=0)
    area2 = (x22 - x21).clamp(min=0) * (y22 - y21).clamp(min=0)
    union = area1[:, None] + area2[None, :] - inter + 1e-7
    iou_mat = inter / union

    matched_pred_indices = []
    matched_gt_indices = []
    available_mask = torch.ones(N, dtype=torch.bool, device=pred_boxes_img.device)
    for gi in range(T):
        if not available_mask.any():
            break
        masked_ious = iou_mat[:, gi].clone()
        masked_ious[~available_mask] = -1.0
        best_pi = int(torch.argmax(masked_ious).item())
        best_val = float(masked_ious[best_pi].item())
        if best_val > 0:
            matched_pred_indices.append(best_pi)
            matched_gt_indices.append(gi)
            available_mask[best_pi] = False

    return matched_pred_indices, matched_gt_indices, iou_mat


def compute_detection_loss_v2(
    outputs,
    targets,
    text_embeddings=None,
    matching_strategy='greedy',
    lambda_l1=5.0,
    lambda_giou=2.0,
    lambda_obj=1.0,
    lambda_div=0.5,
    lambda_contrastive=1.0,
    focal_alpha=0.25,
    focal_gamma=2.0,
    diversity_iou_thresh=0.5,
    use_soft_obj_targets=False,
    contrastive_temperature=0.07,
    return_metrics=False,
):
    """
    Batch-aware detection loss with configurable weights, focal objectness, and optional
    contrastive alignment between region and CLIP text embeddings.

    Args:
        outputs: object with fields
            - pred_boxes: [B, N, 4]
            - logits: [..., 0] used as objectness logits => [B, N]
            - region_embeddings (optional): [B, N, D]
        targets: list length B of dicts or tensors
            - if dict: must include key 'boxes' => [T_i, 4]
            - if tensor/list: treated as boxes for that image
        text_embeddings: Tensor [T_total, D] or list length B of [T_i, D]; if provided, enables contrastive loss
        lambda_*: weights for each loss component
        focal_alpha, focal_gamma: focal loss hyperparameters
        diversity_iou_thresh: only penalize overlap above this IoU
        use_soft_obj_targets: if True, positives get IoU as target; else 1.0
    """
    pred_boxes = outputs.pred_boxes  # [B, N, 4]
    obj_logits = outputs.logits[..., 0]  # [B, N]
    device = pred_boxes.device

    B, N, _ = pred_boxes.shape

    # Normalize targets per image
    norm_targets = []
    if isinstance(targets, (list, tuple)) and len(targets) == B:
        for b in range(B):
            tb = targets[b]
            if isinstance(tb, dict) and 'boxes' in tb:
                norm_targets.append(_ensure_boxes_tensor_per_image(tb['boxes'], device))
            else:
                norm_targets.append(_ensure_boxes_tensor_per_image(tb, device))
    else:
        # Backward-compat: single set of boxes for B==1
        norm_targets.append(_ensure_boxes_tensor_per_image(targets, device))
        if B > 1:
            for _ in range(B - 1):
                norm_targets.append(torch.zeros((0, 4), dtype=torch.float32, device=device))

    total_l1 = torch.zeros((), device=device)
    total_giou = torch.zeros((), device=device)
    total_div = torch.zeros((), device=device)
    num_matched_total = 0

    all_obj_targets = []

    # Per-image loop
    for b in range(B):
        boxes_b = pred_boxes[b]
        tgt_b = norm_targets[b]

        # Simple validity clamp for targets (avoid negative widths/heights)
        if tgt_b.numel() > 0:
            x1y1 = torch.min(tgt_b[:, 0:2], tgt_b[:, 2:4])
            x2y2 = torch.max(tgt_b[:, 0:2], tgt_b[:, 2:4])
            tgt_b = torch.cat([x1y1, x2y2], dim=1)

        # Compute matches per strategy
        if matching_strategy == 'greedy':
            matched_pred_idx, matched_gt_idx, iou_mat = _greedy_match_iou(boxes_b, tgt_b)
        elif matching_strategy == 'hungarian':
            # Use IoU-based cost for assignment (maximize IoU => minimize -IoU)
            mp, mg, iou_mat = _greedy_match_iou(boxes_b, tgt_b) if tgt_b.numel() == 0 or boxes_b.numel() == 0 else (None, None, None)
            if iou_mat is None:
                matched_pred_idx, matched_gt_idx = [], []
            else:
                try:
                    from scipy.optimize import linear_sum_assignment
                    cost = (-iou_mat).detach().cpu().numpy()
                    row_ind, col_ind = linear_sum_assignment(cost)
                    # Filter out non-positive IoUs
                    valid = iou_mat[row_ind, col_ind] > 0
                    matched_pred_idx = [int(r) for r in row_ind[valid].tolist()]
                    matched_gt_idx = [int(c) for c in col_ind[valid].tolist()]
                except Exception:
                    # Fallback to greedy if scipy not available
                    matched_pred_idx, matched_gt_idx, iou_mat = _greedy_match_iou(boxes_b, tgt_b)
        else:
            matched_pred_idx, matched_gt_idx, iou_mat = _greedy_match_iou(boxes_b, tgt_b)

        # Objectness targets
        obj_t = torch.zeros((N,), dtype=torch.float32, device=device)
        if matched_pred_idx:
            pred_sel = boxes_b[matched_pred_idx]
            tgt_sel = tgt_b[matched_gt_idx]
            total_l1 = total_l1 + F.l1_loss(pred_sel, tgt_sel)
            total_giou = total_giou + giou_loss(pred_sel, tgt_sel)

            if use_soft_obj_targets and iou_mat is not None:
                obj_t[matched_pred_idx] = iou_mat[matched_pred_idx, matched_gt_idx].clamp(0, 1)
            else:
                obj_t[matched_pred_idx] = 1.0

            # Diversity penalty among matched predictions above threshold
            if len(matched_pred_idx) > 1:
                pb = pred_sel
                x1 = torch.max(pb[:, None, 0], pb[None, :, 0])
                y1 = torch.max(pb[:, None, 1], pb[None, :, 1])
                x2 = torch.min(pb[:, None, 2], pb[None, :, 2])
                y2 = torch.min(pb[:, None, 3], pb[None, :, 3])
                inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
                area = (pb[:, 2] - pb[:, 0]).clamp(min=0) * (pb[:, 3] - pb[:, 1]).clamp(min=0)
                union = area[:, None] + area[None, :] - inter + 1e-7
                pair_iou = inter / union
                # Exclude diagonal and threshold
                mask = ~torch.eye(pair_iou.size(0), dtype=torch.bool, device=device)
                pair_vals = pair_iou[mask]
                if pair_vals.numel() > 0:
                    high_overlap = pair_vals[pair_vals > diversity_iou_thresh]
                    if high_overlap.numel() > 0:
                        total_div = total_div + high_overlap.mean()

            num_matched_total += len(matched_pred_idx)

        all_obj_targets.append(obj_t)

    # Objectness focal loss across all predictions
    obj_targets = torch.stack(all_obj_targets, dim=0)  # [B, N]
    loss_obj = binary_focal_loss_with_logits(obj_logits, obj_targets, alpha=focal_alpha, gamma=focal_gamma, reduction='mean')

    # Normalize L1/GIoU/Div by number of matched pairs to keep scales stable
    denom = max(num_matched_total, 1)
    total_l1 = total_l1 / denom
    total_giou = total_giou / denom
    total_div = total_div / denom

    # Optional contrastive alignment loss (zero-shot alignment)
    loss_contrastive = torch.zeros((), device=device)
    if text_embeddings is not None and hasattr(outputs, 'region_embeddings') and outputs.region_embeddings is not None:
        region_embeddings = outputs.region_embeddings  # [B, N, D]
        # Build matched pairs across batch and align with text embeddings selection strategy
        # Expect text_embeddings as list per image or a single tensor per image mapping 1:1 with GT order
        if isinstance(text_embeddings, (list, tuple)) and len(text_embeddings) == B:
            for b in range(B):
                tgt_b = norm_targets[b]
                if tgt_b.numel() == 0:
                    continue
                boxes_b = pred_boxes[b]
                matched_pred_idx, matched_gt_idx, _ = _greedy_match_iou(boxes_b, tgt_b)
                if not matched_pred_idx:
                    continue
                reg_b = region_embeddings[b][matched_pred_idx]  # [M, D]
                txt_b = text_embeddings[b]
                if isinstance(txt_b, torch.Tensor):
                    txt_b = txt_b.to(device=device, dtype=reg_b.dtype)
                else:
                    txt_b = torch.tensor(txt_b, dtype=reg_b.dtype, device=device)
                # Align using gt indices
                txt_sel = txt_b[matched_gt_idx]
                reg_b = F.normalize(reg_b, p=2, dim=-1)
                txt_sel = F.normalize(txt_sel, p=2, dim=-1)
                sim = F.cosine_similarity(reg_b, txt_sel, dim=1) / contrastive_temperature
                loss_contrastive = loss_contrastive + (-sim.mean())
        else:
            # Single shared text bank: pick by matched GT index (assumes consistent ordering)
            txt = text_embeddings
            if not isinstance(txt, torch.Tensor):
                txt = torch.tensor(txt, dtype=region_embeddings.dtype, device=device)
            else:
                txt = txt.to(device=device, dtype=region_embeddings.dtype)
            for b in range(B):
                tgt_b = norm_targets[b]
                if tgt_b.numel() == 0:
                    continue
                boxes_b = pred_boxes[b]
                matched_pred_idx, matched_gt_idx, _ = _greedy_match_iou(boxes_b, tgt_b)
                if not matched_pred_idx:
                    continue
                reg_b = outputs.region_embeddings[b][matched_pred_idx]
                txt_sel = txt[matched_gt_idx]
                reg_b = F.normalize(reg_b, p=2, dim=-1)
                txt_sel = F.normalize(txt_sel, p=2, dim=-1)
                sim = F.cosine_similarity(reg_b, txt_sel, dim=1) / contrastive_temperature
                loss_contrastive = loss_contrastive + (-sim.mean())

        # Average contrastive over images with matches for stability
        loss_contrastive = loss_contrastive / max(B, 1)

    total_loss = (
        lambda_l1 * total_l1
        + lambda_giou * total_giou
        + lambda_obj * loss_obj
        + lambda_div * total_div
        + lambda_contrastive * loss_contrastive
    )

    if torch.isnan(total_loss) or torch.isinf(total_loss):
        print(
            f"Warning: Invalid loss (l1:{total_l1.item():.4f}, giou:{total_giou.item():.4f}, "
            f"obj:{loss_obj.item():.4f}, div:{total_div.item():.4f}, contr:{loss_contrastive.item():.4f})"
        )
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)

    if return_metrics:
        metrics = {
            'loss_total': float(total_loss.detach().item()),
            'loss_l1': float(total_l1.detach().item()),
            'loss_giou': float(total_giou.detach().item()),
            'loss_obj': float(loss_obj.detach().item()),
            'loss_div': float(total_div.detach().item()),
            'loss_contrastive': float(loss_contrastive.detach().item()) if lambda_contrastive > 0 else 0.0,
        }
        return total_loss, metrics
    return total_loss
