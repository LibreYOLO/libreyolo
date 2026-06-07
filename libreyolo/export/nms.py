"""Graph-embedded Non-Maximum Suppression for portable detection export.

Wraps a detector whose raw export output is ``(B, 4 + nc, N)`` — xyxy boxes in
input-pixel coordinates followed by per-class probabilities — and appends
class-aware NMS *inside* the model graph. The wrapped model returns a single
fixed-shape tensor ``(B, max_det, 6)`` whose rows are
``[x1, y1, x2, y2, score, class]``, zero-padded past the detection count so the
output shape stays static.

Suppression is expressed with :func:`torchvision.ops.nms`, which lowers to the
standard ONNX ``NonMaxSuppression`` operator. The exported model is therefore
self-contained and runs on any runtime that implements that operator (ONNX
Runtime CPU/GPU, OpenVINO, and others) with no external post-processing.

The wrapper is precision-agnostic: the suppression math runs in float32
regardless of the backbone precision, so it composes with fp32, fp16, and int8
exports. Only batch size 1 is supported — the graph indexes the first image and
emits a single image's detections.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.ops import nms as _nms


def class_offset(imgsz: tuple[int, int]) -> float:
    """Per-class coordinate offset guaranteeing class-independent suppression.

    Class-aware NMS shifts every box by ``class_id * offset``; for boxes of
    different classes to never overlap, the offset must exceed the spatial
    extent of any box. Twice the larger image side (plus one) covers boxes that
    extend slightly beyond the image border.
    """
    return 2.0 * float(max(int(imgsz[0]), int(imgsz[1]))) + 1.0


class EmbeddedNMSDetector(nn.Module):
    """Detector wrapper returning post-NMS detections of shape ``(1, max_det, 6)``.

    Args:
        model: Detection model in export mode whose forward returns
            ``(B, 4 + nc, N)`` — xyxy pixel boxes followed by per-class scores.
        conf: Score threshold; detections at or below it are discarded.
        iou: IoU threshold for suppression.
        max_det: Fixed number of output rows (zero-padded past the count).
        offset: Class-separation offset (see :func:`class_offset`).
    """

    def __init__(
        self, model: nn.Module, *, conf: float, iou: float, max_det: int, offset: float
    ):
        super().__init__()
        self.model = model
        self.conf = float(conf)
        self.iou = float(iou)
        self.max_det = int(max_det)
        self.offset = float(offset)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.model(x)
        # (B, 4 + nc, N) -> (N, 4 + nc); batch-1 graph. Suppression in float32.
        pred = raw[0].transpose(0, 1).float()
        boxes = pred[:, :4]  # xyxy, input pixels
        scores = pred[:, 4:]  # (N, nc) per-class probabilities
        conf, cls = scores.max(dim=1)  # best class per anchor
        cls_f = cls.to(boxes.dtype)

        # Class-aware NMS via the coordinate-offset trick: shift each box by a
        # per-class amount larger than the image so boxes of different classes
        # can never overlap and are suppressed independently of one another.
        boxes_off = boxes + cls_f[:, None] * self.offset
        keep = _nms(boxes_off, conf, self.iou)

        det_boxes = boxes[keep]
        det_conf = conf[keep]
        det_cls = cls_f[keep]

        # Zero out whole sub-threshold rows so they sink below the zero-padding
        # in the top-k selection below (consumers filter by score > 0).
        valid = (det_conf > self.conf).to(det_boxes.dtype)[:, None]
        row = torch.cat((det_boxes, det_conf[:, None], det_cls[:, None]), dim=1)
        row = row * valid  # (k, 6)

        # Guarantee at least max_det rows, then keep the top-scoring max_det.
        # This handles both k < max_det (zeros fill) and k > max_det (trim)
        # uniformly and yields a static (max_det, 6) output sorted by score.
        padded = torch.cat((row, row.new_zeros(self.max_det, 6)), dim=0)
        top = torch.topk(padded[:, 4], self.max_det).indices
        det = padded[top]
        # Reshape (not unsqueeze) with a constant shape so the exported graph
        # records a static (1, max_det, 6) output instead of a dynamic dim.
        return det.reshape(1, self.max_det, 6)
