"""RT-DETR postprocessing (flat top-K decode, no NMS).

Extracted verbatim from ``LibreRTDETR._postprocess``
(``libreyolo/models/rtdetr/model.py``), which now delegates here.
RT-DETRv2 shares this path via inheritance.
"""

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
