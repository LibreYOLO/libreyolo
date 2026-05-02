"""LibreVocab1 Phase 2 — box format conversions and IoU/GIoU.

Self-contained on purpose. Phase 2 should not depend on which specific box-ops
implementation a future LibreYOLO refactor settles on, and the math is small.

Conventions:
    - cxcywh: (cx, cy, w, h), normalized to [0, 1] (decoder output convention).
    - xyxy:   (x1, y1, x2, y2), arbitrary range.
"""

from __future__ import annotations

from typing import Tuple

import torch


def box_cxcywh_to_xyxy(b: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = b.unbind(-1)
    return torch.stack(
        [cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dim=-1
    )


def box_xyxy_to_cxcywh(b: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = b.unbind(-1)
    return torch.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], dim=-1)


def box_area(b: torch.Tensor) -> torch.Tensor:
    return (b[..., 2] - b[..., 0]).clamp(min=0) * (b[..., 3] - b[..., 1]).clamp(min=0)


def _box_iou_pair(a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """IoU and union between two box sets in xyxy.

    Args:
        a: (Na, 4)
        b: (Nb, 4)
    Returns:
        iou:   (Na, Nb)
        union: (Na, Nb)
    """
    area_a = box_area(a)  # (Na,)
    area_b = box_area(b)  # (Nb,)
    lt = torch.max(a[:, None, :2], b[None, :, :2])  # (Na, Nb, 2)
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]  # (Na, Nb)
    union = area_a[:, None] + area_b[None, :] - inter
    iou = inter / union.clamp(min=1e-7)
    return iou, union


def generalized_box_iou_pair(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """GIoU between two box sets in xyxy.

    Args:
        a: (Na, 4)
        b: (Nb, 4)
    Returns:
        giou: (Na, Nb), values in [-1, 1].
    """
    if a.numel() == 0 or b.numel() == 0:
        return a.new_zeros((a.shape[0], b.shape[0]))
    iou, union = _box_iou_pair(a, b)
    lt = torch.min(a[:, None, :2], b[None, :, :2])
    rb = torch.max(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    enclosing = wh[..., 0] * wh[..., 1]
    return iou - (enclosing - union) / enclosing.clamp(min=1e-7)
