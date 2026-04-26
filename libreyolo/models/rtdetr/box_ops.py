"""
Box operations for RT-DETR.

Provides box format conversions and IoU computations used by the
matcher, criterion, and denoising modules.
"""

import torch


def box_cxcywh_to_xyxy(x: torch.Tensor) -> torch.Tensor:
    """Convert boxes from (cx, cy, w, h) to (x1, y1, x2, y2) format."""
    cx, cy, w, h = x.unbind(-1)
    return torch.stack([cx - 0.5 * w, cy - 0.5 * h,
                        cx + 0.5 * w, cy + 0.5 * h], dim=-1)


def box_xyxy_to_cxcywh(x: torch.Tensor) -> torch.Tensor:
    """Convert boxes from (x1, y1, x2, y2) to (cx, cy, w, h) format."""
    x0, y0, x1, y1 = x.unbind(-1)
    return torch.stack([(x0 + x1) / 2, (y0 + y1) / 2,
                        x1 - x0, y1 - y0], dim=-1)


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor):
    """Compute pairwise IoU between two sets of boxes in xyxy format.

    Returns:
        iou: [N, M] IoU matrix
        union: [N, M] union area matrix
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2[None, :] - inter

    iou = inter / (union + 1e-6)
    return iou, union


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Compute pairwise Generalized IoU between two sets of boxes in xyxy format.

    Returns:
        giou: [N, M] GIoU matrix
    """
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all(), "boxes1 must be in xyxy format"
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all(), "boxes2 must be in xyxy format"

    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[None, :, 2:])

    wh = (rb - lt).clamp(min=0)
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / (area + 1e-6)