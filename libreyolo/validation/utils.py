"""Validation utilities: prediction-to-GT matching and box conversions."""

from typing import Tuple

import numpy as np
import torch

from ..utils.box_ops import box_iou


def match_predictions_to_gt(
    pred_boxes: torch.Tensor,
    pred_classes: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_classes: torch.Tensor,
    iou_thresholds: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Match predictions to ground truth using IoU-based greedy assignment.

    Algorithm:
    1. Compute IoU matrix between all predictions and ground truths
    2. Zero out different-class pairs
    3. For each GT, assign the highest-confidence prediction that wants it
    4. Determine correctness at all thresholds

    Args:
        pred_boxes: (N, 4) xyxy format.
        pred_classes: (N,) predicted class indices.
        gt_boxes: (M, 4) xyxy format.
        gt_classes: (M,) ground truth class indices.
        iou_thresholds: (T,) IoU thresholds.

    Returns:
        correct: (N, T) boolean tensor indicating TP at each threshold.
        iou_values: (N,) max IoU per prediction with matching class.
    """
    device = pred_boxes.device
    n_pred = len(pred_boxes)
    n_gt = len(gt_boxes)
    n_thresh = len(iou_thresholds)

    correct = torch.zeros(n_pred, n_thresh, dtype=torch.bool, device=device)
    iou_values = torch.zeros(n_pred, device=device)

    if n_pred == 0 or n_gt == 0:
        return correct, iou_values

    iou_matrix = box_iou(pred_boxes, gt_boxes)  # (N_pred, N_gt)

    # Zero out different-class pairs
    class_match = pred_classes.unsqueeze(1) == gt_classes.unsqueeze(0)  # (N_pred, N_gt)
    iou_matrix = iou_matrix * class_match.float()

    iou_values, _ = iou_matrix.max(dim=1)

    min_thresh = iou_thresholds.min().item()

    max_iou_per_pred, best_gt_per_pred = iou_matrix.max(dim=1)  # (N_pred,)
    valid_preds = max_iou_per_pred >= min_thresh

    # Greedy assignment: each GT goes to the highest-confidence pred that wants it
    # Preds are sorted by confidence (highest first), so lower indices = higher priority
    pred_matched_iou = torch.zeros(n_pred, device=device)

    if valid_preds.any():
        valid_pred_indices = torch.where(valid_preds)[0]  # sorted ascending
        target_gts = best_gt_per_pred[valid_pred_indices]

        gt_taken = torch.zeros(n_gt, dtype=torch.bool, device=device)

        for i, pred_idx in enumerate(valid_pred_indices):
            gt_idx = target_gts[i]
            if not gt_taken[gt_idx]:
                gt_taken[gt_idx] = True
                pred_matched_iou[pred_idx] = max_iou_per_pred[pred_idx]

    # correct[i, j] = True if pred i matched with IoU >= threshold j
    matched_mask = pred_matched_iou > 0
    correct = (
        pred_matched_iou.unsqueeze(1) >= iou_thresholds.unsqueeze(0)
    ) & matched_mask.unsqueeze(1)

    return correct, iou_values


def process_batch(
    pred_boxes: torch.Tensor,
    pred_scores: torch.Tensor,
    pred_classes: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_classes: torch.Tensor,
    iou_thresholds: torch.Tensor,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Process single image predictions for metrics computation.

    Returns:
        (correct, conf, pred_cls, target_cls) as numpy arrays.
    """
    sorted_idx = torch.argsort(pred_scores, descending=True)
    pred_boxes = pred_boxes[sorted_idx]
    pred_scores = pred_scores[sorted_idx]
    pred_classes = pred_classes[sorted_idx]

    correct, _ = match_predictions_to_gt(
        pred_boxes, pred_classes, gt_boxes, gt_classes, iou_thresholds
    )

    return (
        correct.cpu().numpy(),
        pred_scores.cpu().numpy(),
        pred_classes.cpu().numpy(),
        gt_classes.cpu().numpy(),
    )


# =============================================================================
# Box format conversions
# =============================================================================


def xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert (cx, cy, w, h) → (x1, y1, x2, y2)."""
    xyxy = boxes.clone()
    xyxy[..., 0] = boxes[..., 0] - boxes[..., 2] / 2
    xyxy[..., 1] = boxes[..., 1] - boxes[..., 3] / 2
    xyxy[..., 2] = boxes[..., 0] + boxes[..., 2] / 2
    xyxy[..., 3] = boxes[..., 1] + boxes[..., 3] / 2
    return xyxy


def xyxy_to_xywh(boxes: torch.Tensor) -> torch.Tensor:
    """Convert (x1, y1, x2, y2) → (cx, cy, w, h)."""
    xywh = boxes.clone()
    xywh[..., 0] = (boxes[..., 0] + boxes[..., 2]) / 2
    xywh[..., 1] = (boxes[..., 1] + boxes[..., 3]) / 2
    xywh[..., 2] = boxes[..., 2] - boxes[..., 0]
    xywh[..., 3] = boxes[..., 3] - boxes[..., 1]
    return xywh


def scale_boxes(
    boxes: torch.Tensor,
    from_shape: Tuple[int, int],
    to_shape: Tuple[int, int],
    clip: bool = True,
) -> torch.Tensor:
    """Scale boxes from one image size (h, w) to another, optionally clipping."""
    gain_h = to_shape[0] / from_shape[0]
    gain_w = to_shape[1] / from_shape[1]

    scaled = boxes.clone().float()
    scaled[..., [0, 2]] *= gain_w
    scaled[..., [1, 3]] *= gain_h

    if clip:
        scaled[..., [0, 2]] = scaled[..., [0, 2]].clamp(0, to_shape[1])
        scaled[..., [1, 3]] = scaled[..., [1, 3]].clamp(0, to_shape[0])

    return scaled


def clip_boxes(boxes: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
    """Clip boxes to image boundaries (h, w)."""
    boxes[..., [0, 2]] = boxes[..., [0, 2]].clamp(0, shape[1])
    boxes[..., [1, 3]] = boxes[..., [1, 3]].clamp(0, shape[0])
    return boxes
