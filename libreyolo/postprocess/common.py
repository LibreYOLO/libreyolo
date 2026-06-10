"""Shared detection postprocessing steps.

``postprocess_detections`` is the common scale/clip/filter/NMS tail used by
the YOLOX and RTMDet decode paths. Moved verbatim from
``libreyolo/utils/general.py`` (which re-exports it for backward
compatibility).
"""

from typing import Tuple

import torch
import torchvision.ops


def postprocess_detections(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    class_ids: torch.Tensor,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    input_size: int = 640,
    original_size: Tuple[int, int] | None = None,
    max_det: int = 300,
    letterbox: bool = False,
) -> dict:
    """
    Shared post-processing pipeline for object detection outputs.

    This function handles the common post-processing steps:
    - Scale boxes to original image size
    - Clip boxes to image boundaries
    - Filter invalid boxes (zero/negative area)
    - Apply per-class NMS
    - Limit to max detections

    Args:
        boxes: Decoded boxes in xyxy format (N, 4)
        scores: Confidence scores after sigmoid (N,)
        class_ids: Class indices (N,)
        conf_thres: Confidence threshold (already applied before calling)
        iou_thres: IoU threshold for NMS
        input_size: Model input size for scaling
        original_size: Original image size (width, height)
        max_det: Maximum number of detections
        letterbox: If True, use letterbox-inverse scaling (aspect-preserving).
            If False, use independent x/y scaling (simple resize).

    Returns:
        Dictionary with boxes, scores, classes, num_detections
    """
    if len(boxes) == 0:
        return {"boxes": [], "scores": [], "classes": [], "num_detections": 0}

    # Drop NaN/Inf rows FIRST, before scaling/clamping can mask `inf` by
    # clipping it to the image bounds (which would otherwise let a bogus
    # row survive the guard later).
    finite_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores)
    if not finite_mask.all():
        boxes = boxes[finite_mask]
        scores = scores[finite_mask]
        class_ids = class_ids[finite_mask]

    if len(boxes) == 0:
        return {"boxes": [], "scores": [], "classes": [], "num_detections": 0}

    # Scale boxes to original image size
    if original_size is not None:
        if letterbox:
            # Letterbox inverse: r = min(input/orig_h, input/orig_w)
            orig_w, orig_h = original_size
            r = min(input_size / orig_h, input_size / orig_w)
            boxes[:, :4] = boxes[:, :4] / r
        else:
            # Simple resize: independent x/y scaling
            scale_x = original_size[0] / input_size
            scale_y = original_size[1] / input_size
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y

        boxes[:, [0, 2]] = torch.clamp(boxes[:, [0, 2]], 0, original_size[0])
        boxes[:, [1, 3]] = torch.clamp(boxes[:, [1, 3]], 0, original_size[1])

        # Filter zero/negative-area boxes
        box_widths = boxes[:, 2] - boxes[:, 0]
        box_heights = boxes[:, 3] - boxes[:, 1]
        valid_mask = (box_widths > 0) & (box_heights > 0)

        if not valid_mask.all():
            boxes = boxes[valid_mask]
            scores = scores[valid_mask]
            class_ids = class_ids[valid_mask]

    if len(boxes) == 0:
        return {"boxes": [], "scores": [], "classes": [], "num_detections": 0}

    # Cast to fp32 — batched_nms applies a per-class offset of (boxes.max() + 1)
    # which overflows fp16 (max 65504) for typical letterbox coords × num_classes
    # and silently merges classes that should stay separate. Detectron2 carries
    # this same wrapper for the same reason. scores is cast alongside because
    # torchvision.ops.nms requires matching dtypes; check both inputs.
    if boxes.dtype == torch.float16 or scores.dtype == torch.float16:
        boxes = boxes.float()
        scores = scores.float()

    # Per-class NMS — single batched dispatch instead of one kernel per class.
    # batched_nms's class-offset trick uses (boxes.max() + 1) and only
    # separates classes when all coords are non-negative. Callers may pass
    # boxes with negative coords (e.g. YOLOX with ratio==1.0 skips the
    # pre-clamp); shift to neutralise. NMS IoU is translation-invariant so
    # kept indices are unchanged.
    nms_boxes = boxes - boxes.min().clamp(max=0)
    keep_indices = torchvision.ops.batched_nms(nms_boxes, scores, class_ids, iou_thres)

    if len(keep_indices) == 0:
        return {"boxes": [], "scores": [], "classes": [], "num_detections": 0}

    # Always sort by descending score — guarantees the output contract
    # regardless of batched_nms's internal ordering across torchvision
    # versions, and also handles the max_det truncation in one step.
    k = min(len(keep_indices), max_det)
    _, top_indices = torch.topk(scores[keep_indices], k)
    keep_indices = keep_indices[top_indices]

    final_boxes = boxes[keep_indices].cpu().numpy()
    final_scores = scores[keep_indices].cpu().numpy()
    final_classes = class_ids[keep_indices].cpu().numpy()

    return {
        "boxes": final_boxes.tolist(),
        "scores": final_scores.tolist(),
        "classes": final_classes.tolist(),
        "num_detections": len(final_boxes),
    }
