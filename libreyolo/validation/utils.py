"""
Utility functions for validation.

Provides prediction-to-ground-truth matching and helper functions.
"""

from typing import Dict, List, Tuple

import numpy as np
import torch


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
    1. Compute IoU matrix between all predictions and all ground truths
    2. For each class, find the best matching GT for each prediction
    3. Mark prediction as TP if IoU >= threshold and GT not already matched
    4. Apply this for each IoU threshold

    Args:
        pred_boxes: (N, 4) predicted boxes in xyxy format.
        pred_classes: (N,) predicted class indices.
        gt_boxes: (M, 4) ground truth boxes in xyxy format.
        gt_classes: (M,) ground truth class indices.
        iou_thresholds: (T,) IoU thresholds to evaluate.

    Returns:
        correct: (N, T) boolean tensor indicating TP at each threshold.
        iou_values: (N,) maximum IoU value for each prediction with matching class.
    """
    from libreyolo.common.postprocessing import box_iou

    device = pred_boxes.device
    n_pred = len(pred_boxes)
    n_gt = len(gt_boxes)
    n_thresh = len(iou_thresholds)

    correct = torch.zeros(n_pred, n_thresh, dtype=torch.bool, device=device)
    iou_values = torch.zeros(n_pred, device=device)

    if n_pred == 0 or n_gt == 0:
        return correct, iou_values

    # Compute IoU matrix
    iou_matrix = box_iou(pred_boxes, gt_boxes)  # (N_pred, N_gt)

    # Match by class (greedy matching)
    for iou_idx, thresh in enumerate(iou_thresholds):
        gt_matched = torch.zeros(n_gt, dtype=torch.bool, device=device)

        # Process predictions (they should be sorted by confidence beforehand)
        for pred_idx in range(n_pred):
            pred_cls = pred_classes[pred_idx]

            # Find GTs with same class
            cls_mask = gt_classes == pred_cls
            if not cls_mask.any():
                continue

            # Get IoU with same-class GTs
            ious = iou_matrix[pred_idx].clone()
            ious[~cls_mask] = 0  # Mask different classes
            ious[gt_matched] = 0  # Mask already matched

            max_iou, gt_idx = ious.max(0)

            if max_iou >= thresh:
                correct[pred_idx, iou_idx] = True
                gt_matched[gt_idx] = True

                # Store max IoU (only once, at first threshold)
                if iou_idx == 0:
                    iou_values[pred_idx] = max_iou

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

    Args:
        pred_boxes: (N, 4) predicted boxes in xyxy format.
        pred_scores: (N,) confidence scores.
        pred_classes: (N,) predicted class indices.
        gt_boxes: (M, 4) ground truth boxes in xyxy format.
        gt_classes: (M,) ground truth class indices.
        iou_thresholds: IoU thresholds for evaluation.

    Returns:
        Tuple of numpy arrays:
            - correct: (N, T) boolean array indicating TP
            - conf: (N,) confidence scores
            - pred_cls: (N,) predicted class indices
            - target_cls: (M,) ground truth class indices
    """
    # Sort predictions by confidence (descending)
    sorted_idx = torch.argsort(pred_scores, descending=True)
    pred_boxes = pred_boxes[sorted_idx]
    pred_scores = pred_scores[sorted_idx]
    pred_classes = pred_classes[sorted_idx]

    # Match predictions to ground truth
    correct, _ = match_predictions_to_gt(
        pred_boxes, pred_classes, gt_boxes, gt_classes, iou_thresholds
    )

    return (
        correct.cpu().numpy(),
        pred_scores.cpu().numpy(),
        pred_classes.cpu().numpy(),
        gt_classes.cpu().numpy(),
    )


def xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """
    Convert boxes from xywh (center) to xyxy format.

    Args:
        boxes: (..., 4) boxes in xywh format (cx, cy, w, h).

    Returns:
        Boxes in xyxy format (x1, y1, x2, y2).
    """
    xyxy = boxes.clone()
    xyxy[..., 0] = boxes[..., 0] - boxes[..., 2] / 2  # x1
    xyxy[..., 1] = boxes[..., 1] - boxes[..., 3] / 2  # y1
    xyxy[..., 2] = boxes[..., 0] + boxes[..., 2] / 2  # x2
    xyxy[..., 3] = boxes[..., 1] + boxes[..., 3] / 2  # y2
    return xyxy


def xyxy_to_xywh(boxes: torch.Tensor) -> torch.Tensor:
    """
    Convert boxes from xyxy to xywh (center) format.

    Args:
        boxes: (..., 4) boxes in xyxy format (x1, y1, x2, y2).

    Returns:
        Boxes in xywh format (cx, cy, w, h).
    """
    xywh = boxes.clone()
    xywh[..., 0] = (boxes[..., 0] + boxes[..., 2]) / 2  # cx
    xywh[..., 1] = (boxes[..., 1] + boxes[..., 3]) / 2  # cy
    xywh[..., 2] = boxes[..., 2] - boxes[..., 0]  # w
    xywh[..., 3] = boxes[..., 3] - boxes[..., 1]  # h
    return xywh


def scale_boxes(
    boxes: torch.Tensor,
    from_shape: Tuple[int, int],
    to_shape: Tuple[int, int],
    clip: bool = True,
) -> torch.Tensor:
    """
    Scale boxes from one image size to another.

    Args:
        boxes: (N, 4) boxes in xyxy format.
        from_shape: Source image shape (height, width).
        to_shape: Target image shape (height, width).
        clip: Whether to clip boxes to image boundaries.

    Returns:
        Scaled boxes in xyxy format.
    """
    gain_h = to_shape[0] / from_shape[0]
    gain_w = to_shape[1] / from_shape[1]

    scaled = boxes.clone().float()
    scaled[..., [0, 2]] *= gain_w  # x coordinates
    scaled[..., [1, 3]] *= gain_h  # y coordinates

    if clip:
        scaled[..., [0, 2]] = scaled[..., [0, 2]].clamp(0, to_shape[1])
        scaled[..., [1, 3]] = scaled[..., [1, 3]].clamp(0, to_shape[0])

    return scaled


def clip_boxes(boxes: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
    """
    Clip boxes to image boundaries.

    Args:
        boxes: (N, 4) boxes in xyxy format.
        shape: Image shape (height, width).

    Returns:
        Clipped boxes.
    """
    boxes[..., [0, 2]] = boxes[..., [0, 2]].clamp(0, shape[1])  # x
    boxes[..., [1, 3]] = boxes[..., [1, 3]].clamp(0, shape[0])  # y
    return boxes
