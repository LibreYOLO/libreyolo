"""Graph-embedded Non-Maximum Suppression for portable detection export.

Wraps a detector whose raw export output is ``(B, 4 + nc, N)`` - xyxy boxes in
input-pixel coordinates followed by per-class probabilities - and appends
class-aware NMS *inside* the model graph. The first wrapped-model output is a
fixed-shape tensor ``(B, max_det, 6)`` whose rows are
``[x1, y1, x2, y2, score, class]``, zero-padded past the detection count so the
output shape stays static. The second output is the raw detector tensor, which
LibreYOLO backends use to preserve native post-processing parity when the
original image is not square.

Suppression is expressed with :func:`torchvision.ops.nms`, which lowers to the
standard ONNX ``NonMaxSuppression`` operator. Standalone consumers can use the
first output without external post-processing, while LibreYOLO can use the raw
auxiliary output when it needs original-image clipping before final NMS.

Detection semantics for the first output mirror the library's own YOLO9
post-processing on the exported input canvas: candidates are selected
multi-label (every class scoring above ``conf`` for an anchor, not just the best
one), then suppressed class-aware. The suppression math runs in float32
regardless of the backbone precision, so it composes with fp32 and int8 exports.
Only batch size 1 is supported - the graph indexes the first image and emits a
single image's detections.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.ops import nms as _nms

from ..models.yolo9.utils import _YOLO9_MAX_NMS_CANDIDATES


class EmbeddedNMSDetector(nn.Module):
    """Detector wrapper returning post-NMS detections plus raw model output.

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
        boxes_raw = pred[:, :4]  # (N, 4) xyxy, input pixels
        x1 = boxes_raw[:, 0].clamp(min=0.0, max=float(x.shape[-1]))
        y1 = boxes_raw[:, 1].clamp(min=0.0, max=float(x.shape[-2]))
        x2 = boxes_raw[:, 2].clamp(min=0.0, max=float(x.shape[-1]))
        y2 = boxes_raw[:, 3].clamp(min=0.0, max=float(x.shape[-2]))
        boxes_all = torch.stack((x1, y1, x2, y2), dim=1)
        scores_all = pred[:, 4:]  # (N, nc) per-class probabilities
        finite_boxes = torch.isfinite(boxes_all).all(dim=1)
        finite_scores = torch.isfinite(scores_all)
        safe_boxes_all = torch.where(
            torch.isfinite(boxes_all), boxes_all, torch.zeros_like(boxes_all)
        )
        safe_scores_all = torch.where(
            finite_boxes[:, None] & finite_scores,
            scores_all,
            scores_all.new_full(scores_all.shape, -1.0),
        )

        # Multi-label candidate selection: every (anchor, class) pair scoring
        # above conf becomes a detection, matching the YOLO9 post-processing.
        # Native YOLO9 caps candidates before NMS; taking the top scores before
        # thresholding is equivalent to threshold-then-cap, but bounds the ONNX
        # NonMaxSuppression input for low-conf exports.
        flat_scores = safe_scores_all.reshape(-1)
        num_classes = safe_scores_all.shape[1]
        max_nms = min(
            flat_scores.shape[0],
            max(self.max_det, _YOLO9_MAX_NMS_CANDIDATES),
        )
        top_scores, top_flat_idx = torch.topk(flat_scores, max_nms)
        score_mask = top_scores > self.conf
        top_scores = top_scores[score_mask]
        top_flat_idx = top_flat_idx[score_mask]
        anchor_idx = torch.floor(top_flat_idx.to(torch.float32) / float(num_classes)).to(
            torch.long
        )
        class_idx = top_flat_idx - anchor_idx * num_classes
        cand_boxes = safe_boxes_all[anchor_idx]  # (K, 4)
        cand_scores = top_scores  # (K,)
        cand_cls = class_idx.to(boxes_all.dtype)  # (K,)
        valid_boxes = (cand_boxes[:, 2] > cand_boxes[:, 0]) & (
            cand_boxes[:, 3] > cand_boxes[:, 1]
        )
        cand_boxes = cand_boxes[valid_boxes]
        cand_scores = cand_scores[valid_boxes]
        cand_cls = cand_cls[valid_boxes]

        # Class-aware NMS via the coordinate-offset trick. Use sanitized boxes
        # for the global range so non-finite anchors outside the candidate set
        # cannot poison the offset applied to valid detections.
        lo = safe_boxes_all.min()
        step = (safe_boxes_all.max() - lo).clamp(min=1.0) + 1.0
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
        return det.reshape(1, self.max_det, 6), raw
