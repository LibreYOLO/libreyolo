"""Utility functions for YOLOv9 E2E (NMS-free) inference."""

from typing import Dict, Tuple

import torch

from ..yolo9.utils import decode_boxes, preprocess_image, preprocess_numpy


def _scale_and_clip_boxes(
    boxes: torch.Tensor,
    input_size: int,
    original_size: Tuple[int, int] | None,
    letterbox: bool,
) -> torch.Tensor:
    if original_size is None or len(boxes) == 0:
        return boxes

    boxes = boxes.clone()
    orig_w, orig_h = original_size

    if letterbox:
        ratio = min(input_size / orig_h, input_size / orig_w)
        boxes[:, :4] = boxes[:, :4] / ratio
    else:
        scale_x = orig_w / input_size
        scale_y = orig_h / input_size
        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y

    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, orig_w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, orig_h)
    return boxes


def postprocess(
    output: Dict,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    input_size: int = 640,
    original_size: Tuple[int, int] | None = None,
    max_det: int = 300,
    letterbox: bool = False,
) -> Dict:
    """Postprocess YOLOv9 E2E outputs with top-K selection (no NMS).

    The one-to-one head produces at most one prediction per object, so NMS
    is not required. Detections are filtered by confidence and ranked by
    per-anchor max score before applying the user's max_det cap.
    """
    del iou_thres  # not used — no NMS

    predictions = output["predictions"]
    if predictions.dim() == 2:
        predictions = predictions.unsqueeze(0)

    preds = predictions.transpose(1, 2)  # (B, N, 4+nc)
    boxes = preds[..., :4]
    scores = preds[..., 4:]

    batch_size, num_anchors, num_classes = scores.shape
    topk_anchors = min(max_det, num_anchors)
    if topk_anchors == 0 or num_classes == 0:
        return {"boxes": [], "scores": [], "classes": [], "num_detections": 0}

    # Stage 1: select top-K anchors by their best class score
    anchor_scores = scores.amax(dim=-1)
    anchor_scores, anchor_indices = torch.topk(anchor_scores, topk_anchors, dim=-1)
    del anchor_scores

    gather_box_idx = anchor_indices.unsqueeze(-1).expand(-1, -1, boxes.shape[-1])
    gather_score_idx = anchor_indices.unsqueeze(-1).expand(-1, -1, num_classes)
    boxes = torch.gather(boxes, dim=1, index=gather_box_idx)
    scores = torch.gather(scores, dim=1, index=gather_score_idx)

    # Stage 2: rank by individual (anchor, class) score pairs up to max_det
    flat_scores = scores.flatten(1)
    topk_scores = min(max_det, flat_scores.shape[1])
    scores, flat_indices = torch.topk(flat_scores, topk_scores, dim=-1)
    class_ids = flat_indices % num_classes
    box_indices = flat_indices // num_classes
    boxes = boxes.gather(
        dim=1, index=box_indices.unsqueeze(-1).expand(-1, -1, boxes.shape[-1])
    )

    # Batch dim 0 only (single image inference)
    scores = scores[0]
    class_ids = class_ids[0]
    boxes = boxes[0]

    keep = scores > conf_thres
    if not keep.any():
        return {"boxes": [], "scores": [], "classes": [], "num_detections": 0}

    boxes = boxes[keep]
    scores = scores[keep]
    class_ids = class_ids[keep]
    boxes = _scale_and_clip_boxes(boxes, input_size, original_size, letterbox)

    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    valid = (widths > 0) & (heights > 0)
    if not valid.any():
        return {"boxes": [], "scores": [], "classes": [], "num_detections": 0}

    boxes = boxes[valid].cpu()
    scores = scores[valid].cpu()
    class_ids = class_ids[valid].cpu()
    return {
        "boxes": boxes,
        "scores": scores,
        "classes": class_ids,
        "num_detections": len(boxes),
    }


__all__ = ["preprocess_numpy", "preprocess_image", "decode_boxes", "postprocess"]
