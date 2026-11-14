"""RetinaNet anchor filtering, class-aware NMS, and coordinate restoration."""

from __future__ import annotations

import math
from typing import Any, Optional, Tuple

import numpy as np
import torch
from torchvision.ops import batched_nms
from torchvision.ops.boxes import clip_boxes_to_image


_STRIDES = (8, 16, 32, 64, 128)
_ANCHORS_PER_LOCATION = 9
_SIZE_DIVISIBLE = 32
_UPSTREAM_MAX_SIZE = 1333
_UPSTREAM_MIN_SIZE = 800


def resize_geometry(
    original_size: Tuple[int, int], input_size: int, fixed_canvas: bool = False
) -> tuple[int, int, float, float]:
    """Return upstream resized H/W and exact x/y inverse-scale factors.

    ``fixed_canvas``: the image was fitted into an ``input_size`` square.
    """
    original_width, original_height = original_size
    if fixed_canvas:
        scale = min(input_size / original_height, input_size / original_width)
    else:
        max_size = round(input_size * _UPSTREAM_MAX_SIZE / _UPSTREAM_MIN_SIZE)
        scale = min(
            input_size / min(original_height, original_width),
            max_size / max(original_height, original_width),
        )
    resized_height = int(original_height * scale)
    resized_width = int(original_width * scale)
    scale_x = resized_width / original_width
    scale_y = resized_height / original_height
    return resized_height, resized_width, scale_x, scale_y


def level_anchor_counts(resized_height: int, resized_width: int) -> list[int]:
    """Return P3-P7 anchor counts after the upstream 32-pixel padding."""
    canvas_height = math.ceil(resized_height / _SIZE_DIVISIBLE) * _SIZE_DIVISIBLE
    canvas_width = math.ceil(resized_width / _SIZE_DIVISIBLE) * _SIZE_DIVISIBLE
    return [
        math.ceil(canvas_height / stride)
        * math.ceil(canvas_width / stride)
        * _ANCHORS_PER_LOCATION
        for stride in _STRIDES
    ]


def _empty() -> dict:
    return {
        "num_detections": 0,
        "boxes": np.zeros((0, 4), dtype=np.float32),
        "scores": np.zeros((0,), dtype=np.float32),
        "classes": np.zeros((0,), dtype=np.int64),
    }


def _unpack(output: Any) -> torch.Tensor:
    tensor = torch.as_tensor(output)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 3 or tensor.shape[-1] < 5:
        raise ValueError(
            "RetinaNet output must have shape (B, anchors, 4 + classes), "
            f"got {tuple(tensor.shape)}"
        )
    if tensor.shape[0] != 1:
        raise ValueError(
            "RetinaNet native postprocessing currently requires batch=1, "
            f"got batch={tensor.shape[0]}"
        )
    return tensor[0]


def postprocess(
    output: Any,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    original_size: Tuple[int, int] = (800, 800),
    max_det: int = 300,
    ratio: float = 1.0,
    input_size: int = 800,
    topk_candidates: int = 1000,
    classes: Optional[list[int]] = None,
    **_unused,
) -> dict:
    """Decode one RetinaNet output into LibreYOLO's canonical dictionary.

    Candidate selection mirrors torchvision: threshold and top-K independently
    on every FPN level, then one class-aware NMS over the level union. The
    model graph already decodes anchor deltas and maps official sparse COCO-91
    scores into LibreYOLO's contiguous COCO-80 order.
    """
    del ratio  # exact rounded x/y scales are reconstructed below
    predictions = _unpack(output)
    resized_height, resized_width, scale_x, scale_y = resize_geometry(
        original_size, int(input_size)
    )
    counts = level_anchor_counts(resized_height, resized_width)
    if sum(counts) != predictions.shape[0]:
        raise ValueError(
            "RetinaNet anchor count does not match preprocessing geometry: "
            f"output has {predictions.shape[0]}, expected {sum(counts)} for "
            f"resized {resized_height}x{resized_width}."
        )

    per_level = predictions.split(counts, dim=0)
    selected_boxes: list[torch.Tensor] = []
    selected_scores: list[torch.Tensor] = []
    selected_classes: list[torch.Tensor] = []
    allowed_classes = None
    if classes is not None:
        allowed_classes = torch.as_tensor(
            classes, dtype=torch.int64, device=predictions.device
        )

    for level in per_level:
        boxes = level[:, :4]
        scores = level[:, 4:]
        valid = scores > conf_thres
        if allowed_classes is not None:
            class_mask = torch.zeros(
                scores.shape[1], dtype=torch.bool, device=scores.device
            )
            valid_ids = allowed_classes[
                (allowed_classes >= 0) & (allowed_classes < scores.shape[1])
            ]
            if valid_ids.numel():
                class_mask[valid_ids] = True
            valid &= class_mask.unsqueeze(0)
        pairs = torch.nonzero(valid, as_tuple=False)
        if pairs.numel() == 0:
            continue
        pair_scores = scores[pairs[:, 0], pairs[:, 1]]
        keep_count = min(int(topk_candidates), int(pair_scores.numel()))
        pair_scores, order = pair_scores.topk(keep_count)
        pairs = pairs[order]
        level_boxes = boxes[pairs[:, 0]]
        level_boxes = clip_boxes_to_image(level_boxes, (resized_height, resized_width))
        selected_boxes.append(level_boxes)
        selected_scores.append(pair_scores)
        selected_classes.append(pairs[:, 1].to(torch.int64))

    if not selected_boxes:
        return _empty()

    boxes = torch.cat(selected_boxes)
    scores = torch.cat(selected_scores)
    class_ids = torch.cat(selected_classes)
    finite = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores)
    boxes = boxes[finite]
    scores = scores[finite]
    class_ids = class_ids[finite]
    valid_boxes = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    boxes = boxes[valid_boxes]
    scores = scores[valid_boxes]
    class_ids = class_ids[valid_boxes]
    if boxes.numel() == 0:
        return _empty()

    keep = batched_nms(boxes.float(), scores.float(), class_ids, iou_thres)
    keep = keep[: max(0, int(max_det))]
    boxes = boxes[keep].clone().to(torch.float32)
    scores = scores[keep].to(torch.float32)
    class_ids = class_ids[keep]

    original_width, original_height = original_size
    boxes[:, [0, 2]] /= scale_x
    boxes[:, [1, 3]] /= scale_y
    boxes[:, [0, 2]].clamp_(0, original_width)
    boxes[:, [1, 3]].clamp_(0, original_height)

    boxes_array = boxes.detach().cpu().numpy()
    scores_array = scores.detach().cpu().numpy()
    classes_array = class_ids.detach().cpu().numpy().astype(np.int64, copy=False)
    return {
        "num_detections": int(boxes_array.shape[0]),
        "boxes": boxes_array,
        "scores": scores_array,
        "classes": classes_array,
    }


__all__ = ["level_anchor_counts", "postprocess", "resize_geometry"]
