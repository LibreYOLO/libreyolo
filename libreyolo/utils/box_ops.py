"""Unified bounding box IoU operations for LibreYOLO."""

import math

import torch
from torch import Tensor


def box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """
    Compute pairwise IoU between two sets of boxes.

    Args:
        boxes1: (N, 4) boxes in xyxy format.
        boxes2: (M, 4) boxes in xyxy format.

    Returns:
        IoU matrix of shape (N, M).
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2 - inter
    return inter / (union + 1e-7)


def compute_iou(
    bbox1: Tensor,
    bbox2: Tensor,
    mode: str = "iou",
    *,
    pairwise: bool = True,
) -> Tensor:
    """
    Calculate IoU, DIoU, or CIoU between two sets of bounding boxes.

    Args:
        bbox1: Bounding boxes in xyxy format. Shape: (A, 4) or (B, A, 4)
        bbox2: Bounding boxes in xyxy format. Shape: (B, 4) or (B, B, 4)
        mode: IoU variant - "iou", "diou", or "ciou"
        pairwise: Whether to compute all pairwise IoUs for 2D inputs.

    Returns:
        IoU tensor. Shape depends on input dimensions.
    """
    mode = mode.lower()
    EPS = 1e-7
    dtype = bbox1.dtype
    bbox1 = bbox1.to(torch.float32)
    bbox2 = bbox2.to(torch.float32)

    if bbox1.ndim == 2 and bbox2.ndim == 2:
        if pairwise:
            bbox1 = bbox1.unsqueeze(1)  # (A, 4) -> (A, 1, 4)
            bbox2 = bbox2.unsqueeze(0)  # (B, 4) -> (1, B, 4)
        else:
            if bbox1.shape != bbox2.shape:
                raise ValueError(
                    "bbox1 and bbox2 must have the same shape for elementwise IoU"
                )
    elif bbox1.ndim == 3 and bbox2.ndim == 3:
        bbox1 = bbox1.unsqueeze(2)  # (BZ, A, 4) -> (BZ, A, 1, 4)
        bbox2 = bbox2.unsqueeze(1)  # (BZ, B, 4) -> (BZ, 1, B, 4)

    xmin_inter = torch.max(bbox1[..., 0], bbox2[..., 0])
    ymin_inter = torch.max(bbox1[..., 1], bbox2[..., 1])
    xmax_inter = torch.min(bbox1[..., 2], bbox2[..., 2])
    ymax_inter = torch.min(bbox1[..., 3], bbox2[..., 3])

    intersection_area = torch.clamp(xmax_inter - xmin_inter, min=0) * torch.clamp(
        ymax_inter - ymin_inter, min=0
    )

    area_bbox1 = (bbox1[..., 2] - bbox1[..., 0]) * (bbox1[..., 3] - bbox1[..., 1])
    area_bbox2 = (bbox2[..., 2] - bbox2[..., 0]) * (bbox2[..., 3] - bbox2[..., 1])

    union_area = area_bbox1 + area_bbox2 - intersection_area

    iou = intersection_area / (union_area + EPS)
    if mode == "iou":
        return iou.to(dtype)

    # Centroid distance
    cx1 = (bbox1[..., 2] + bbox1[..., 0]) / 2
    cy1 = (bbox1[..., 3] + bbox1[..., 1]) / 2
    cx2 = (bbox2[..., 2] + bbox2[..., 0]) / 2
    cy2 = (bbox2[..., 3] + bbox2[..., 1]) / 2
    cent_dis = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2

    # Diagonal of smallest enclosing box
    c_x = torch.max(bbox1[..., 2], bbox2[..., 2]) - torch.min(
        bbox1[..., 0], bbox2[..., 0]
    )
    c_y = torch.max(bbox1[..., 3], bbox2[..., 3]) - torch.min(
        bbox1[..., 1], bbox2[..., 1]
    )
    diag_dis = c_x**2 + c_y**2 + EPS

    diou = iou - (cent_dis / diag_dis)
    if mode == "diou":
        return diou.to(dtype)

    # Aspect ratio penalty (CIoU)
    arctan = torch.atan(
        (bbox1[..., 2] - bbox1[..., 0]) / (bbox1[..., 3] - bbox1[..., 1] + EPS)
    ) - torch.atan(
        (bbox2[..., 2] - bbox2[..., 0]) / (bbox2[..., 3] - bbox2[..., 1] + EPS)
    )
    v = (4 / (math.pi**2)) * (arctan**2)
    with torch.no_grad():
        alpha = v / (v - iou + 1 + EPS)
    ciou = diou - alpha * v
    return ciou.to(dtype)
