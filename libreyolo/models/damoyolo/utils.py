"""DAMO-YOLO preprocessing + postprocessing helpers.

Preprocessing matches upstream's inference pipeline
(``damo/utils/demo_utils.py::transform_img`` + ``test_transform`` defaults):

- RGB, no normalization, float32 in [0, 255]
- resize to (640, 640) via ``cv2.INTER_LINEAR``, *no* keep-ratio (image stretched)

Postprocessing applies multiclass NMS to the head's ``(cls_scores, boxes)``
output and rescales boxes back to the original image frame.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
import torchvision


def preprocess_numpy(
    img_rgb_hwc: np.ndarray, input_size: int = 640
) -> Tuple[np.ndarray, float]:
    """Stretch-resize an RGB HWC uint8 array to ``(input_size, input_size)``.

    Returns ``(chw_float32_in_0_255, ratio)``. Since DAMO-YOLO uses non-keep-ratio
    resize, ``ratio`` is reported as ``1.0`` (caller scales x and y separately
    via the original_size hint).
    """
    img = cv2.resize(img_rgb_hwc, (input_size, input_size), interpolation=cv2.INTER_LINEAR).astype(np.uint8)
    chw = np.ascontiguousarray(img.transpose(2, 0, 1), dtype=np.float32)
    return chw, 1.0


def preprocess_image(
    image_path: str | Path,
    input_size: Tuple[int, int] = (640, 640),
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Load and preprocess one image.

    Returns ``(tensor, (orig_w, orig_h))``. ``tensor`` is float32 (3, H, W),
    range [0, 255], RGB.
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(image_path)
    orig_h, orig_w = img.shape[:2]
    # cv2 reads BGR; upstream loads via PIL.convert("RGB") so we mirror that.
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h_t, w_t = input_size
    img = cv2.resize(img, (w_t, h_t), interpolation=cv2.INTER_LINEAR).astype(np.uint8)
    img = img.transpose(2, 0, 1)  # HWC -> CHW
    tensor = torch.from_numpy(np.ascontiguousarray(img, dtype=np.float32))
    return tensor, (orig_w, orig_h)


def multiclass_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    score_thr: float,
    iou_thr: float,
    max_num: int = 100,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """NMS over (N, 4) boxes and (N, C) per-class scores, returning the top
    ``max_num`` detections.

    Mirrors upstream ``damo/utils/boxes.py::multiclass_nms``.
    """
    num_classes = scores.size(1)
    bboxes = boxes[:, None].expand(scores.size(0), num_classes, 4)
    valid = scores > score_thr
    if not valid.any():
        zero = boxes.new_zeros((0,))
        return boxes.new_zeros((0, 4)), zero, zero.long()
    bboxes = bboxes[valid]
    sel_scores = scores[valid]
    labels = valid.nonzero(as_tuple=False)[:, 1]
    keep = torchvision.ops.batched_nms(bboxes, sel_scores, labels, iou_thr)
    if max_num > 0:
        keep = keep[:max_num]
    return bboxes[keep], sel_scores[keep], labels[keep]


def postprocess_predictions(
    cls_scores: torch.Tensor,  # (B, N, C)
    boxes: torch.Tensor,  # (B, N, 4) xyxy in model-input pixels
    orig_sizes,  # list of (orig_w, orig_h) per batch element
    input_size: Tuple[int, int] = (640, 640),
    conf_thres: float = 0.05,
    iou_thres: float = 0.7,
    max_det: int = 100,
):
    """Per-batch NMS + rescale to original image frame.

    Returns a list (length B) of dicts with keys ``boxes`` (xyxy, original
    pixel coords), ``scores``, ``classes``.
    """
    B = cls_scores.size(0)
    h_in, w_in = input_size
    out = []
    for i in range(B):
        boxes_i = boxes[i]
        scores_i = cls_scores[i]
        det_boxes, det_scores, det_labels = multiclass_nms(
            boxes_i, scores_i, conf_thres, iou_thres, max_num=max_det
        )
        ow, oh = orig_sizes[i]
        if det_boxes.numel() > 0:
            sx = ow / w_in
            sy = oh / h_in
            det_boxes = det_boxes.clone()
            det_boxes[:, 0::2] *= sx
            det_boxes[:, 1::2] *= sy
            det_boxes[:, 0::2].clamp_(min=0, max=ow)
            det_boxes[:, 1::2].clamp_(min=0, max=oh)
        out.append(
            {
                "boxes": det_boxes,
                "scores": det_scores,
                "classes": det_labels,
            }
        )
    return out
