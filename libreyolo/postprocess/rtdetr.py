"""RT-DETR postprocessing (flat top-K decode, no NMS).

Extracted verbatim from ``LibreRTDETR._postprocess``
(``libreyolo/models/rtdetr/model.py``), which now delegates here.
RT-DETRv2 shares this path via inheritance.
"""

import math
from typing import Any, Dict, Tuple

import torch


def postprocess(
    output: Any,
    conf_thres: float,
    iou_thres: float,
    original_size: Tuple[int, int],
    max_det: int = 300,
    ratio: float = 1.0,
    **kwargs,
) -> Dict:
    """Convert RTDETR outputs to detection results.

    Args:
        output: dict with pred_logits [1, Q, C] and pred_boxes [1, Q, 4] (cxcywh normalized)
        conf_thres: confidence threshold
        iou_thres: IoU threshold (not used for RTDETR - NMS-free)
        original_size: (width, height)
        max_det: maximum detections
        ratio: aspect ratio (1.0 for RTDETR)

    Returns:
        Dict with boxes, scores, classes, num_detections
    """
    pred_logits = output["pred_logits"]  # [1, Q, C]
    pred_boxes = output["pred_boxes"]  # [1, Q, 4] cxcywh normalized

    # Match upstream RTDETRPostProcessor: top-K across the flattened (Q*C)
    # score matrix, allowing multiple classes per query. The previous
    # per-query ``scores.max(dim=-1)`` cost ~0.7–0.9 mAP on COCO val2017
    # because non-argmax classes that would still rank in the top-300
    # globally were silently discarded before COCO eval saw them.
    scores_per_class = torch.sigmoid(pred_logits[0])  # [Q, C]
    num_classes = scores_per_class.shape[-1]
    flat = scores_per_class.flatten()
    k = min(max_det, flat.numel())
    topk_scores, topk_indices = torch.topk(flat, k)
    query_idx = topk_indices // num_classes
    class_idx = topk_indices % num_classes

    boxes = pred_boxes[0][query_idx]  # [k, 4] cxcywh normalized
    scores = topk_scores
    labels = class_idx

    # Convert cxcywh normalized to xyxy pixel coords
    orig_w, orig_h = original_size
    cx, cy, w, h = boxes.unbind(-1)
    x1 = (cx - w / 2) * orig_w
    y1 = (cy - h / 2) * orig_h
    x2 = (cx + w / 2) * orig_w
    y2 = (cy + h / 2) * orig_h
    boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)

    # Filter by confidence after top-K (matches upstream + D-FINE).
    mask = scores > conf_thres
    scores = scores[mask]
    labels = labels[mask]
    boxes_xyxy = boxes_xyxy[mask]

    return {
        "boxes": boxes_xyxy.cpu(),
        "scores": scores.cpu(),
        "classes": labels.cpu(),
        "num_detections": len(boxes_xyxy),
    }


def postprocess_obb(
    output: Any,
    conf_thres: float,
    iou_thres: float,
    original_size: Tuple[int, int],
    max_det: int = 300,
    input_size: int | Tuple[int, int] = 1024,
    float32_scale: bool = False,
    **kwargs,
) -> Dict:
    """Decode RT-DETRv2 OBB outputs with global flat top-k and no NMS.

    The model emits normalized ``(cx, cy, w, h, angle/pi)`` coordinates on a
    top-left-aligned, aspect-preserving padded canvas.  Returned OBB rows are
    ``(cx, cy, w, h, angle_radians, confidence, class)`` in original-image
    pixels.  Horizontal boxes enclose the rotated corners.

    ``float32_scale`` computes the letterbox scale as a float32 tensor, the
    way RiO-DETRv2's ``PostProcessorOBB`` does (int / int64 tensor), instead
    of a Python float; the two can differ by one float32 ULP in the returned
    pixels.
    """
    del iou_thres, kwargs  # RT-DETR is NMS-free.

    pred_logits = output["pred_logits"]
    pred_boxes = output["pred_boxes"]
    if pred_boxes.shape[-1] != 5:
        raise ValueError(
            "RT-DETRv2 OBB pred_boxes must have five coordinates, "
            f"got shape {tuple(pred_boxes.shape)}"
        )

    scores_per_class = pred_logits[0].sigmoid()
    num_classes = scores_per_class.shape[-1]
    flat = scores_per_class.flatten()
    k = min(max_det, flat.numel())
    scores, flat_indices = torch.topk(flat, k)
    query_indices = flat_indices // num_classes
    labels = flat_indices % num_classes

    selected = pred_boxes[0][query_indices]
    if isinstance(input_size, int):
        target_h = target_w = int(input_size)
    else:
        target_h, target_w = int(input_size[0]), int(input_size[1])
    orig_w, orig_h = original_size
    if float32_scale:
        # Same expression as upstream: Python int divided by an int64 tensor.
        # That kernel rounds differently from a float32 / float32 division.
        sizes = torch.tensor([int(orig_h), int(orig_w)], device=selected.device)
        scale = torch.minimum(target_w / sizes[1], target_h / sizes[0])
    else:
        scale = min(target_w / orig_w, target_h / orig_h)

    target_wh = selected.new_tensor([target_w, target_h, target_w, target_h])
    xywh = selected[:, :4] * target_wh / scale
    angles = selected[:, 4] * math.pi

    keep = scores > conf_thres
    xywh = xywh[keep]
    angles = angles[keep]
    scores = scores[keep]
    labels = labels[keep]

    if len(xywh):
        half_w = xywh[:, 2] / 2
        half_h = xywh[:, 3] / 2
        cos = angles.cos().abs()
        sin = angles.sin().abs()
        extent_x = cos * half_w + sin * half_h
        extent_y = sin * half_w + cos * half_h
        boxes_xyxy = torch.stack(
            [
                xywh[:, 0] - extent_x,
                xywh[:, 1] - extent_y,
                xywh[:, 0] + extent_x,
                xywh[:, 1] + extent_y,
            ],
            dim=-1,
        )
        obb = torch.cat(
            [
                xywh,
                angles.unsqueeze(-1),
                scores.unsqueeze(-1),
                labels.to(xywh.dtype).unsqueeze(-1),
            ],
            dim=-1,
        )
    else:
        boxes_xyxy = selected.new_zeros((0, 4))
        obb = selected.new_zeros((0, 7))

    return {
        "boxes": boxes_xyxy.cpu(),
        "scores": scores.cpu(),
        "classes": labels.cpu(),
        "obb": obb.cpu(),
        "num_detections": len(boxes_xyxy),
    }


__all__ = ["postprocess", "postprocess_obb"]
