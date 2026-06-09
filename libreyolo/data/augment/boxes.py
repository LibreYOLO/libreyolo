"""Bounding-box format conversions and adjustments.

The ``2``-named converters (``xyxy2cxcywh``, ``cxcywh2xyxy``) are YOLOX-legacy
and MUTATE their input via sequential column overwrites; the ``_to_``-named
converters return a new array and leave the input untouched. Prefer the
out-of-place forms in new code.
"""

import numpy as np


def cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """Out-of-place (cx, cy, w, h) -> xyxy conversion."""
    out = np.zeros_like(boxes, dtype=np.float32)
    out[:, 0] = boxes[:, 0] - boxes[:, 2] * 0.5
    out[:, 1] = boxes[:, 1] - boxes[:, 3] * 0.5
    out[:, 2] = boxes[:, 0] + boxes[:, 2] * 0.5
    out[:, 3] = boxes[:, 1] + boxes[:, 3] * 0.5
    return out


def xyxy_to_cxcywh(boxes: np.ndarray) -> np.ndarray:
    """Out-of-place xyxy -> (cx, cy, w, h) conversion."""
    out = np.zeros_like(boxes, dtype=np.float32)
    out[:, 0] = (boxes[:, 0] + boxes[:, 2]) * 0.5
    out[:, 1] = (boxes[:, 1] + boxes[:, 3]) * 0.5
    out[:, 2] = boxes[:, 2] - boxes[:, 0]
    out[:, 3] = boxes[:, 3] - boxes[:, 1]
    return out


def xyxy2cxcywh(bboxes):
    """Convert bboxes from xyxy to (cx, cy, w, h) in-place."""
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]  # w
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]  # h
    bboxes[:, 0] = bboxes[:, 0] + bboxes[:, 2] * 0.5  # cx
    bboxes[:, 1] = bboxes[:, 1] + bboxes[:, 3] * 0.5  # cy
    return bboxes


def cxcywh2xyxy(bboxes):
    """Convert bboxes from (cx, cy, w, h) to xyxy in-place."""
    bboxes[:, 0] = bboxes[:, 0] - bboxes[:, 2] * 0.5  # x1
    bboxes[:, 1] = bboxes[:, 1] - bboxes[:, 3] * 0.5  # y1
    bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2]  # x2
    bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3]  # y2
    return bboxes


def adjust_box_anns(bbox, scale_ratio, padw, padh, w_max, h_max):
    """Scale + offset boxes, then clip to (w_max, h_max)."""
    bbox[:, 0::2] = np.clip(bbox[:, 0::2] * scale_ratio + padw, 0, w_max)
    bbox[:, 1::2] = np.clip(bbox[:, 1::2] * scale_ratio + padh, 0, h_max)
    return bbox
