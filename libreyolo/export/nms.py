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

Detection semantics mirror the library's own YOLO9 post-processing: candidates
are selected multi-label (every class scoring above ``conf`` for an anchor, not
just the best one), then suppressed class-aware. The suppression math runs in
float32 regardless of the backbone precision, so it composes with fp32 and int8
exports. Only batch size 1 is supported — the graph indexes the first image and
emits a single image's detections.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.ops import nms as _nms


class EmbeddedNMSDetector(nn.Module):
    """Detector wrapper returning post-NMS detections of shape ``(1, max_det, 6)``.

    Args:
        model: Detection model in export mode whose forward returns
            ``(B, 4 + nc, N)`` — xyxy pixel boxes followed by per-class scores.
        conf: Score threshold; only candidates strictly above it are kept.
        iou: IoU threshold for suppression.
        max_det: Fixed number of output rows (zero-padded past the count).
    """

    def __init__(self, model: nn.Module, *, conf: float, iou: float, max_det: int):
        super().__init__()
        self.model = model
        self.conf = float(conf)
        self.iou = float(iou)
        self.max_det = int(max_det)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.model(x)
        # (B, 4 + nc, N) -> (N, 4 + nc); batch-1 graph. Suppression in float32.
        pred = raw[0].transpose(0, 1).float()
        boxes_all = pred[:, :4]  # (N, 4) xyxy, input pixels
        scores_all = pred[:, 4:]  # (N, nc) per-class probabilities

        # Multi-label candidate selection: every (anchor, class) pair scoring
        # above conf becomes a detection, matching the YOLO9 post-processing.
        # Thresholding here (before NMS) also keeps suppression off the flood of
        # low-score anchors that can never be returned.
        cand = (scores_all > self.conf).nonzero()  # (K, 2): [anchor, class]
        anchor_idx = cand[:, 0]
        class_idx = cand[:, 1]
        cand_boxes = boxes_all[anchor_idx]  # (K, 4)
        cand_scores = scores_all[anchor_idx, class_idx]  # (K,)
        cand_cls = class_idx.to(boxes_all.dtype)  # (K,)

        # Class-aware NMS via the coordinate-offset trick. The per-class step is
        # derived from the actual decoded box range (over all N anchors — a
        # fixed, non-empty set), so different classes never overlap after the
        # shift even when boxes extend beyond the image border.
        lo = boxes_all.min()
        step = (boxes_all.max() - lo).clamp(min=1.0) + 1.0
        nmsbox = (cand_boxes - lo) + cand_cls[:, None] * step
        keep = _nms(nmsbox, cand_scores, self.iou)

        row = torch.cat(
            (cand_boxes[keep], cand_scores[keep, None], cand_cls[keep, None]), dim=1
        )  # (k, 6)

        # Guarantee at least max_det rows, then keep the top-scoring max_det.
        # This handles both k < max_det (zeros fill) and k > max_det (trim)
        # uniformly and yields a static (max_det, 6) output sorted by score.
        padded = torch.cat((row, row.new_zeros(self.max_det, 6)), dim=0)
        top = torch.topk(padded[:, 4], self.max_det).indices
        det = padded[top]
        # Reshape (not unsqueeze) with a constant shape so the exported graph
        # records a static (1, max_det, 6) output instead of a dynamic dim.
        return det.reshape(1, self.max_det, 6)
