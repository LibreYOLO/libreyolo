"""YOLOv9 E2E (NMS-free) postprocessing.

``libreyolo/models/yolo9_e2e/utils.py`` re-exports everything here for
backward compatibility.
"""

from collections.abc import Mapping
from typing import Dict, Tuple, Union

import torch

from .common import _input_size_hw


def _scale_and_clip_boxes(
    boxes: torch.Tensor,
    input_size: Union[int, Tuple[int, int]],
    original_size: Tuple[int, int] | None,
    letterbox: bool,
    letterbox_pad: str | None = None,
) -> torch.Tensor:
    if original_size is None or len(boxes) == 0:
        return boxes

    boxes = boxes.clone()
    orig_w, orig_h = original_size
    input_h, input_w = _input_size_hw(input_size)

    if letterbox:
        from ..preprocess.letterbox import unletterbox_xyxy

        boxes = unletterbox_xyxy(
            boxes, orig_w, orig_h, input_h, input_w, pad=letterbox_pad
        )
    else:
        scale_x = orig_w / input_w
        scale_y = orig_h / input_h
        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y

    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, orig_w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, orig_h)
    return boxes


def postprocess(
    output,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    input_size: Union[int, Tuple[int, int]] = 640,
    original_size: Tuple[int, int] | None = None,
    max_det: int = 300,
    letterbox: bool = True,
    letterbox_pad: str | None = None,
) -> Dict:
    """Detections from the one-to-one branch: top-K selection, no NMS.

    A one-to-one head is trained to fire once per object, so the K best
    ``(anchor, class)`` pairs are kept as they are (sort-and-keep-top-k, as
    in DATE, github.com/YiqunChen1999/date, Apache-2.0).

    Args:
        output: Eval output dict (its ``"predictions"``) or the bare
            ``(B, 4 + nc, N)`` predictions tensor: xyxy boxes in input
            pixels, then per-class sigmoid scores.
        conf_thres: Keep selections scoring strictly above this.
        iou_thres: Unused; accepted for signature compatibility.
        input_size: Model input size, int or ``(height, width)``.
        original_size: Source image ``(width, height)``; boxes are mapped
            back to it and clipped when given.
        max_det: Maximum number of ``(anchor, class)`` selections.
        letterbox: Whether the input was letterboxed (else plainly resized).
        letterbox_pad: Letterbox pad placement used by preprocessing.

    Returns:
        ``{"boxes", "scores", "classes", "num_detections"}`` for the first
        image of the batch, in descending score order.
    """
    del iou_thres
    predictions = output["predictions"] if isinstance(output, Mapping) else output
    if predictions.dim() == 2:
        predictions = predictions.unsqueeze(0)

    # Flatten each image's (N, nc) score matrix (index = anchor * nc + class)
    # and keep its K best entries, highest first.
    boxes_all = predictions[:, :4, :].transpose(1, 2)  # (B, N, 4)
    scores_all = predictions[:, 4:, :].transpose(1, 2)  # (B, N, nc)
    batch, num_anchors, num_classes = scores_all.shape
    k = min(int(max_det), num_anchors * num_classes)
    if k <= 0:
        return {"boxes": [], "scores": [], "classes": [], "num_detections": 0}
    scores, flat_idx = scores_all.reshape(batch, -1).topk(k, dim=1)
    class_ids = flat_idx % num_classes
    anchor_idx = flat_idx // num_classes
    boxes = boxes_all.gather(1, anchor_idx.unsqueeze(-1).expand(-1, -1, 4))

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

    boxes = _scale_and_clip_boxes(
        boxes, input_size, original_size, letterbox, letterbox_pad
    )

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
