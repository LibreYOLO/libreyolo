"""Bounding-box format conversions and adjustments."""

import numpy as np


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
