"""Shared rotated-box (OBB) postprocessing primitives.

Task-level helpers, not family-level: every OBB family decodes to the same
``xywhr`` contract (centre, long side first, angle in ``[-pi/2, pi/2)``) and
therefore needs the same corner conversion, axis-aligned proxy, and exact
rotated NMS. Keeping one implementation here avoids a second, subtly
different rotated NMS drifting out of sync with this one.

Originally extracted from ``libreyolo/postprocess/yolo9.py``, which now
imports from this module.
"""

from __future__ import annotations

import numpy as np

from ..utils.lazy import lazy_module


# torch is resolved on first use so this module stays importable in a
# torch-free ONNX deployment (discussions/711).
torch = lazy_module("torch")


def xywhr_to_corners(xywhr: "torch.Tensor") -> "torch.Tensor":
    """Convert ``(N, 5)`` rotated boxes to ``(N, 4, 2)`` corners."""
    xy = xywhr[:, :2]
    w = xywhr[:, 2] / 2
    h = xywhr[:, 3] / 2
    angle = xywhr[:, 4]
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    corners = torch.stack(
        [
            torch.stack([-w, -h], dim=1),
            torch.stack([w, -h], dim=1),
            torch.stack([w, h], dim=1),
            torch.stack([-w, h], dim=1),
        ],
        dim=1,
    )
    rot = torch.stack(
        [
            torch.stack([cos, -sin], dim=1),
            torch.stack([sin, cos], dim=1),
        ],
        dim=1,
    )
    return torch.matmul(corners, rot.transpose(1, 2)) + xy[:, None, :]


def xywhr_to_xyxy(xywhr: "torch.Tensor") -> "torch.Tensor":
    """Axis-aligned bounding box enclosing each rotated box."""
    if xywhr.numel() == 0:
        return torch.zeros((0, 4), dtype=xywhr.dtype, device=xywhr.device)
    corners = xywhr_to_corners(xywhr)
    x = corners[..., 0]
    y = corners[..., 1]
    return torch.stack(
        [
            x.min(dim=1).values,
            y.min(dim=1).values,
            x.max(dim=1).values,
            y.max(dim=1).values,
        ],
        dim=1,
    )


def rotated_nms_keep_indices(
    xywhr: "torch.Tensor",
    scores: "torch.Tensor",
    class_ids: "torch.Tensor",
    iou_thres: float,
    max_det: int,
) -> "torch.Tensor":
    """Greedy per-class NMS on exact rotated-rectangle IoU.

    Returns indices into the input tensors, highest score first. Non-finite
    rows are dropped before suppression rather than poisoning the ordering.
    """
    # Imported here rather than at module scope: ``libreyolo.data`` pulls the
    # dataset stack (and torch) at import time, and this module sits on the
    # torch-free ONNX import path.
    from ..data.obb import xywhr_iou

    if xywhr.numel() == 0:
        return torch.zeros(0, dtype=torch.long, device=xywhr.device)

    finite_mask = torch.isfinite(xywhr).all(dim=1) & torch.isfinite(scores)
    if not finite_mask.all():
        valid_indices = torch.where(finite_mask)[0]
        if len(valid_indices) == 0:
            return torch.zeros(0, dtype=torch.long, device=xywhr.device)
        xywhr = xywhr[finite_mask]
        scores = scores[finite_mask]
        class_ids = class_ids[finite_mask]
    else:
        valid_indices = None

    order = torch.argsort(scores, descending=True)
    rects = xywhr.detach().cpu().numpy().astype(np.float32)
    classes = class_ids.detach().cpu().numpy().astype(np.int64)
    ordered = order.detach().cpu().numpy().astype(np.int64).tolist()

    keep_local: list[int] = []
    while ordered and len(keep_local) < max_det:
        current = ordered.pop(0)
        keep_local.append(current)

        remaining = []
        for candidate in ordered:
            if classes[candidate] != classes[current]:
                remaining.append(candidate)
                continue
            if xywhr_iou(rects[current], rects[candidate]) <= iou_thres:
                remaining.append(candidate)
        ordered = remaining

    keep = torch.as_tensor(keep_local, dtype=torch.long, device=xywhr.device)
    if valid_indices is not None:
        keep = valid_indices[keep]
    return keep


def canonicalize_xywhr_tensor(xywhr: "torch.Tensor") -> "torch.Tensor":
    """Put the long side first and normalise the angle to ``[-pi/2, pi/2)``.

    The public LibreYOLO OBB contract, applied at the postprocessing boundary.
    This changes only the *representation*: the polygon each row describes is
    unchanged, which ``tests/unit/test_yolonas_obb_geometry.py`` asserts.
    """
    import math

    if xywhr.numel() == 0:
        return xywhr
    out = xywhr.clone()
    w = out[:, 2]
    h = out[:, 3]
    angle = out[:, 4]
    swap = h > w
    new_w = torch.where(swap, h, w)
    new_h = torch.where(swap, w, h)
    angle = torch.where(swap, angle + math.pi / 2, angle)
    # ``(a + pi/2) % pi - pi/2`` -> [-pi/2, pi/2)
    angle = torch.remainder(angle + math.pi / 2, math.pi) - math.pi / 2
    out[:, 2] = new_w
    out[:, 3] = new_h
    out[:, 4] = angle
    return out


__all__ = [
    "canonicalize_xywhr_tensor",
    "rotated_nms_keep_indices",
    "xywhr_to_corners",
    "xywhr_to_xyxy",
]
