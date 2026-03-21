"""
Pluggable post-processing functions for object detection.

All functions share the same signature as torchvision.ops.nms:
    (boxes: Tensor[N,4], scores: Tensor[N], iou_threshold: float) -> Tensor[K]
returning indices of boxes to keep.
"""

from typing import Callable, Union

import torch


def soft_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float = 0.45,
    *,
    sigma: float = 0.1,
    score_threshold: float = 0.001,
) -> torch.Tensor:
    """
    Soft-NMS with Gaussian score decay.

    Instead of hard-suppressing overlapping boxes, decays their confidence
    scores using a Gaussian penalty: score *= exp(-(iou^2) / sigma).
    Returns indices of boxes whose decayed score exceeds score_threshold.

    Reference: Bodla et al., "Improving Object Detection With One Line of Code", ICCV 2017.

    Args:
        boxes: Boxes in xyxy format (N, 4).
        scores: Confidence scores (N,).
        iou_threshold: Not used for hard suppression — kept for interface compatibility.
        sigma: Gaussian decay parameter. Lower = more aggressive suppression.
            Default 0.1 is tuned for YOLO-family models. The original paper
            uses 0.5 (for Faster R-CNN), which is too lenient for YOLO.
        score_threshold: Minimum decayed score to keep a box.

    Returns:
        Indices of boxes to keep (on the same device as input).
    """
    if len(boxes) == 0:
        return torch.tensor([], dtype=torch.long, device=boxes.device)

    # Filter NaN/Inf
    valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores)
    if not valid_mask.any():
        return torch.tensor([], dtype=torch.long, device=boxes.device)

    if not valid_mask.all():
        valid_indices = torch.where(valid_mask)[0]
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]
    else:
        valid_indices = None

    N = len(boxes)
    # Work on clones to avoid mutating caller data
    scores = scores.clone()
    original_indices = torch.arange(N, device=boxes.device)
    keep = []

    for _ in range(N):
        # Find current max score
        max_idx = scores.argmax()
        if scores[max_idx] < score_threshold:
            break

        keep.append(original_indices[max_idx].item())

        if len(scores) == 1:
            break

        # Compute IoU of selected box vs all remaining
        box_i = boxes[max_idx]
        x1 = torch.max(box_i[0], boxes[:, 0])
        y1 = torch.max(box_i[1], boxes[:, 1])
        x2 = torch.min(box_i[2], boxes[:, 2])
        y2 = torch.min(box_i[3], boxes[:, 3])

        inter = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        area_i = (box_i[2] - box_i[0]) * (box_i[3] - box_i[1])
        area_all = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        iou = inter / (area_i + area_all - inter + 1e-7)

        # Gaussian decay
        decay = torch.exp(-(iou**2) / sigma)
        scores = scores * decay

        # Remove the selected box from consideration
        remaining = torch.ones(len(scores), dtype=torch.bool, device=boxes.device)
        remaining[max_idx] = False
        boxes = boxes[remaining]
        scores = scores[remaining]
        original_indices = original_indices[remaining]

    keep_tensor = torch.tensor(keep, dtype=torch.long, device=boxes.device)

    if valid_indices is not None:
        keep_tensor = valid_indices[keep_tensor]

    return keep_tensor


def diou_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float = 0.45,
) -> torch.Tensor:
    """
    NMS using Distance-IoU (DIoU) as the suppression criterion.

    DIoU adds a center-point distance penalty to IoU, helping in crowded
    scenes where boxes overlap but belong to different objects.
    Suppression criterion: DIoU = IoU - (center_distance^2 / diagonal^2).

    Reference: Zheng et al., "Distance-IoU Loss", AAAI 2020.

    Args:
        boxes: Boxes in xyxy format (N, 4).
        scores: Confidence scores (N,).
        iou_threshold: DIoU threshold for suppression.

    Returns:
        Indices of boxes to keep (on the same device as input).
    """
    if len(boxes) == 0:
        return torch.tensor([], dtype=torch.long, device=boxes.device)

    # Filter NaN/Inf
    valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores)
    if not valid_mask.any():
        return torch.tensor([], dtype=torch.long, device=boxes.device)

    if not valid_mask.all():
        valid_indices = torch.where(valid_mask)[0]
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]
    else:
        valid_indices = None

    _, order = scores.sort(0, descending=True)
    keep = []

    while len(order) > 0:
        i = order[0]
        keep.append(i.item())

        if len(order) == 1:
            break

        box_i = boxes[i]
        boxes_remaining = boxes[order[1:]]

        # Standard IoU
        x1_inter = torch.max(box_i[0], boxes_remaining[:, 0])
        y1_inter = torch.max(box_i[1], boxes_remaining[:, 1])
        x2_inter = torch.min(box_i[2], boxes_remaining[:, 2])
        y2_inter = torch.min(box_i[3], boxes_remaining[:, 3])

        inter = torch.clamp(x2_inter - x1_inter, min=0) * torch.clamp(
            y2_inter - y1_inter, min=0
        )
        area_i = (box_i[2] - box_i[0]) * (box_i[3] - box_i[1])
        area_r = (boxes_remaining[:, 2] - boxes_remaining[:, 0]) * (
            boxes_remaining[:, 3] - boxes_remaining[:, 1]
        )
        iou = inter / (area_i + area_r - inter + 1e-7)

        # Center distance penalty
        cx_i = (box_i[0] + box_i[2]) / 2
        cy_i = (box_i[1] + box_i[3]) / 2
        cx_r = (boxes_remaining[:, 0] + boxes_remaining[:, 2]) / 2
        cy_r = (boxes_remaining[:, 1] + boxes_remaining[:, 3]) / 2
        center_dist_sq = (cx_i - cx_r) ** 2 + (cy_i - cy_r) ** 2

        # Enclosing box diagonal
        enclose_x1 = torch.min(box_i[0], boxes_remaining[:, 0])
        enclose_y1 = torch.min(box_i[1], boxes_remaining[:, 1])
        enclose_x2 = torch.max(box_i[2], boxes_remaining[:, 2])
        enclose_y2 = torch.max(box_i[3], boxes_remaining[:, 3])
        diag_sq = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2 + 1e-7

        diou = iou - center_dist_sq / diag_sq

        order = order[1:][diou < iou_threshold]

    keep_tensor = torch.tensor(keep, dtype=torch.long, device=boxes.device)

    if valid_indices is not None:
        keep_tensor = valid_indices[keep_tensor]

    return keep_tensor


# ---------------------------------------------------------------------------
# Registry & resolution
# ---------------------------------------------------------------------------

# Maps user-facing string names to callables.
# None means "use the default NMS path" (torchvision or fallback).
POSTPROCESS_METHODS = {
    "nms": None,
    "soft_nms": soft_nms,
    "diou_nms": diou_nms,
}


def resolve_nms_fn(
    method: Union[str, Callable, None],
) -> Union[Callable, None]:
    """
    Resolve a postprocess method specification to a callable (or None for default).

    Args:
        method: One of:
            - None or "nms": use default NMS (returns None)
            - A string key from POSTPROCESS_METHODS
            - A callable matching (boxes, scores, iou_threshold) -> indices

    Returns:
        A callable, or None for the default NMS path.

    Raises:
        ValueError: If the string is not a recognized method name.
        TypeError: If the argument is neither a string, callable, nor None.
    """
    if method is None:
        return None

    if callable(method):
        return method

    if isinstance(method, str):
        if method not in POSTPROCESS_METHODS:
            available = ", ".join(f"'{k}'" for k in sorted(POSTPROCESS_METHODS))
            raise ValueError(
                f"Unknown postprocess method '{method}'. "
                f"Available: {available}. Or pass a callable."
            )
        return POSTPROCESS_METHODS[method]

    raise TypeError(
        f"postprocess must be a string, callable, or None, got {type(method).__name__}"
    )
