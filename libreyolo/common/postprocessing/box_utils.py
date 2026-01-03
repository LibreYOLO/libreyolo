"""
Shared box utility functions for post-processing.

This module provides common operations used across different post-processing methods:
- IoU computation (standard, DIoU, CIoU, GIoU)
- Box format conversions
- Box rescaling
"""

import torch
from typing import Tuple


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute IoU between two sets of boxes.

    Args:
        boxes1: Tensor of shape (N, 4) in xyxy format
        boxes2: Tensor of shape (M, 4) in xyxy format

    Returns:
        IoU matrix of shape (N, M)
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Intersection
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # (N, M, 2)
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # (N, M, 2)
    wh = (rb - lt).clamp(min=0)  # (N, M, 2)
    inter = wh[:, :, 0] * wh[:, :, 1]  # (N, M)

    # Union
    union = area1[:, None] + area2 - inter

    return inter / (union + 1e-7)


def box_iou_pairwise(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise IoU between corresponding boxes.

    Args:
        boxes1: Tensor of shape (N, 4) in xyxy format
        boxes2: Tensor of shape (N, 4) in xyxy format

    Returns:
        IoU tensor of shape (N,)
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]

    union = area1 + area2 - inter
    return inter / (union + 1e-7)


def box_diou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute Distance-IoU (DIoU) between two sets of boxes.

    DIoU adds a penalty term based on the normalized distance between
    box centers, which helps handle cases where boxes don't overlap.

    Reference: https://arxiv.org/abs/1911.08287

    Args:
        boxes1: Tensor of shape (N, 4) in xyxy format
        boxes2: Tensor of shape (M, 4) in xyxy format

    Returns:
        DIoU matrix of shape (N, M)
    """
    # Standard IoU
    iou = box_iou(boxes1, boxes2)

    # Box centers
    center1 = (boxes1[:, :2] + boxes1[:, 2:]) / 2  # (N, 2)
    center2 = (boxes2[:, :2] + boxes2[:, 2:]) / 2  # (M, 2)

    # Squared distance between centers
    center_dist_sq = ((center1[:, None, :] - center2[None, :, :]) ** 2).sum(dim=-1)  # (N, M)

    # Enclosing box (smallest box containing both boxes)
    enclose_lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])  # (N, M, 2)
    enclose_rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])  # (N, M, 2)

    # Diagonal squared of enclosing box
    enclose_diag_sq = ((enclose_rb - enclose_lt) ** 2).sum(dim=-1)  # (N, M)

    # DIoU = IoU - (center_distance^2 / diagonal^2)
    diou = iou - center_dist_sq / (enclose_diag_sq + 1e-7)

    return diou


def box_ciou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute Complete-IoU (CIoU) between two sets of boxes.

    CIoU extends DIoU by adding an aspect ratio consistency term,
    which helps the model learn better box regression.

    Reference: https://arxiv.org/abs/1911.08287

    Args:
        boxes1: Tensor of shape (N, 4) in xyxy format
        boxes2: Tensor of shape (M, 4) in xyxy format

    Returns:
        CIoU matrix of shape (N, M)
    """
    import math

    # Standard IoU components
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    iou = inter / (union + 1e-7)

    # Box centers
    center1 = (boxes1[:, :2] + boxes1[:, 2:]) / 2
    center2 = (boxes2[:, :2] + boxes2[:, 2:]) / 2
    center_dist_sq = ((center1[:, None, :] - center2[None, :, :]) ** 2).sum(dim=-1)

    # Enclosing box diagonal
    enclose_lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    enclose_rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    enclose_diag_sq = ((enclose_rb - enclose_lt) ** 2).sum(dim=-1)

    # Aspect ratio term
    w1, h1 = boxes1[:, 2] - boxes1[:, 0], boxes1[:, 3] - boxes1[:, 1]
    w2, h2 = boxes2[:, 2] - boxes2[:, 0], boxes2[:, 3] - boxes2[:, 1]

    # v measures aspect ratio consistency
    v = (4 / (math.pi ** 2)) * (
        torch.atan(w1[:, None] / (h1[:, None] + 1e-7)) -
        torch.atan(w2 / (h2 + 1e-7))
    ) ** 2

    # alpha is the trade-off parameter
    alpha = v / (1 - iou + v + 1e-7)

    # CIoU = IoU - distance_term - aspect_ratio_term
    ciou = iou - center_dist_sq / (enclose_diag_sq + 1e-7) - alpha * v

    return ciou


def box_giou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute Generalized IoU (GIoU) between two sets of boxes.

    GIoU extends IoU by considering the area of the smallest enclosing box,
    providing a better gradient signal for non-overlapping boxes.

    Reference: https://arxiv.org/abs/1902.09630

    Args:
        boxes1: Tensor of shape (N, 4) in xyxy format
        boxes2: Tensor of shape (M, 4) in xyxy format

    Returns:
        GIoU matrix of shape (N, M)
    """
    # Standard IoU components
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    iou = inter / (union + 1e-7)

    # Enclosing box area
    enclose_lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    enclose_rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    enclose_wh = enclose_rb - enclose_lt
    enclose_area = enclose_wh[:, :, 0] * enclose_wh[:, :, 1]

    # GIoU = IoU - (enclose_area - union) / enclose_area
    giou = iou - (enclose_area - union) / (enclose_area + 1e-7)

    return giou


def rescale_boxes(
    boxes: torch.Tensor,
    input_size: int,
    original_size: Tuple[int, int],
    clip: bool = True
) -> torch.Tensor:
    """
    Rescale boxes from model input size to original image size.

    Args:
        boxes: Tensor of shape (N, 4) in xyxy format
        input_size: Model input size (assumes square input)
        original_size: Original image size as (width, height)
        clip: Whether to clip boxes to image boundaries

    Returns:
        Rescaled boxes of shape (N, 4)
    """
    if len(boxes) == 0:
        return boxes

    scale_x = original_size[0] / input_size
    scale_y = original_size[1] / input_size

    boxes = boxes.clone()
    boxes[:, [0, 2]] *= scale_x
    boxes[:, [1, 3]] *= scale_y

    if clip:
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, original_size[0])
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, original_size[1])

    return boxes


def filter_valid_boxes(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    class_ids: torch.Tensor,
    min_size: float = 0.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Filter out boxes with invalid dimensions.

    Args:
        boxes: Tensor of shape (N, 4) in xyxy format
        scores: Tensor of shape (N,)
        class_ids: Tensor of shape (N,)
        min_size: Minimum box dimension (default: 0)

    Returns:
        Tuple of filtered (boxes, scores, class_ids)
    """
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    valid_mask = (widths > min_size) & (heights > min_size)

    return boxes[valid_mask], scores[valid_mask], class_ids[valid_mask]


def xyxy_to_xywh(boxes: torch.Tensor) -> torch.Tensor:
    """
    Convert boxes from xyxy to xywh format.

    Args:
        boxes: Tensor of shape (..., 4) in xyxy format

    Returns:
        Boxes in xywh format (center_x, center_y, width, height)
    """
    x1, y1, x2, y2 = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2
    cy = y1 + h / 2
    return torch.stack([cx, cy, w, h], dim=-1)


def xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """
    Convert boxes from xywh to xyxy format.

    Args:
        boxes: Tensor of shape (..., 4) in xywh format (center_x, center_y, width, height)

    Returns:
        Boxes in xyxy format
    """
    cx, cy, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)
