"""Class-agnostic non-maximum suppression (``agnostic_nms=True``).

One step shared by ``predict()`` and ``val()`` for every detection family. It
runs on a family's finished detections, after its own postprocess, so it means
the same thing whether the family suppresses per class, emits a set
prediction without NMS, or ships its NMS inside an exported graph: among
boxes that overlap above the IoU threshold only the highest-scoring one
survives, whatever their classes.
"""

from __future__ import annotations

from typing import Any, Dict

import torch
from torchvision.ops import nms

#: Per-detection arrays of a postprocess detection dict that a keep index
#: must be applied to, so they stay aligned with the boxes.
DETECTION_KEYS = ("boxes", "scores", "classes", "masks", "keypoints", "obb")


def agnostic_nms_keep(
    boxes: torch.Tensor, scores: torch.Tensor, iou_thres: float
) -> torch.Tensor:
    """Indices that survive class-agnostic NMS, in their input order.

    Args:
        boxes: ``(N, 4)`` xyxy boxes.
        scores: ``(N,)`` confidences.
        iou_thres: A box is suppressed by a higher-scoring kept box when
            their IoU exceeds this.

    Rows with a non-finite box or score take no part in suppression and are
    kept as they are.
    """
    count = int(boxes.shape[0])
    device = boxes.device
    if count < 2:
        return torch.arange(count, dtype=torch.long, device=device)
    # NMS runs on CPU in fp32: at most max_det boxes, and the kernel is not
    # available for every device and dtype a model runs on.
    cpu_boxes = boxes.detach().reshape(count, 4).float().cpu()
    cpu_scores = scores.detach().reshape(count).float().cpu()
    finite = torch.isfinite(cpu_boxes).all(dim=1) & torch.isfinite(cpu_scores)
    candidates = torch.nonzero(finite).flatten()
    kept = candidates[nms(cpu_boxes[candidates], cpu_scores[candidates], float(iou_thres))]
    keep = torch.cat([kept, torch.nonzero(~finite).flatten()]).sort().values
    return keep.to(device)


def agnostic_rotated_nms_keep(
    xywhr: torch.Tensor, scores: torch.Tensor, iou_thres: float
) -> torch.Tensor:
    """Class-agnostic NMS on rotated ``(N, 5)`` xywhr boxes, input order kept."""
    from ..postprocess.obb_ops import rotated_nms_keep_indices  # noqa: PLC0415

    count = int(xywhr.shape[0])
    if count < 2:
        return torch.arange(count, dtype=torch.long, device=xywhr.device)
    keep = rotated_nms_keep_indices(
        xywhr,
        scores,
        torch.zeros(count, dtype=torch.long, device=xywhr.device),
        float(iou_thres),
        count,
    )
    return keep.sort().values


def top_detections(detections: Dict[str, Any], max_det: int) -> Dict[str, Any]:
    """Keep the ``max_det`` highest-scoring detections, in their input order."""
    scores = detections["scores"]
    if max_det is None or max_det < 0 or len(scores) <= max_det:
        return detections
    top = torch.topk(scores.detach().float().cpu(), int(max_det)).indices.sort().values
    filtered = dict(detections)
    for key in DETECTION_KEYS:
        value = detections.get(key)
        if value is not None:
            filtered[key] = value[top.to(value.device)]
    if "num_detections" in filtered:
        filtered["num_detections"] = len(top)
    return filtered


def agnostic_nms_detections(detections: Dict[str, Any], iou_thres: float) -> Dict[str, Any]:
    """Apply class-agnostic NMS to a detection dict of aligned tensors.

    ``detections`` holds ``boxes`` (xyxy), ``scores`` and ``classes``, with
    optional ``masks``, ``keypoints`` and ``obb``. Returns a new dict with
    every per-detection entry filtered alike; other entries are untouched.
    """
    boxes = detections["boxes"]
    if len(boxes) < 2:
        return detections
    keep = agnostic_nms_keep(boxes, detections["scores"], iou_thres)
    if len(keep) == len(boxes):
        return detections
    filtered = dict(detections)
    for key in DETECTION_KEYS:
        value = detections.get(key)
        if value is not None:
            filtered[key] = value[keep.to(value.device)]
    if "num_detections" in filtered:
        filtered["num_detections"] = len(keep)
    return filtered
