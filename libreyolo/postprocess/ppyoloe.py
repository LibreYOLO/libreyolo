"""PP-YOLOE postprocessing.

The head already emits decoded ``xyxy`` boxes in input-canvas pixels and
sigmoid class probabilities, so this module only has to do the source
selection sequence: confidence filter, optional multi-label expansion,
pre-NMS top-k, class-aware NMS, max detections, then reverse the stretch
resize independently on x and y.

Source defaults (``PPYoloE.__init__`` in the pinned super-gradients commit):
conf 0.5, IoU 0.7, pre-NMS top-k 1024, max 300 detections, multi-label on,
class-aware NMS. There is no objectness term to multiply in.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Sequence, Tuple, Union

from ..utils.lazy import lazy_module

from .common import _input_size_hw

# torch/torchvision are resolved on first use so this module stays importable
# in a torch-free ONNX deployment (discussions/711). ``backends/base.py``
# imports PPYOLOE_PRE_NMS_TOP_K from here at module scope.
torch = lazy_module("torch")
torchvision = lazy_module("torchvision")

__all__ = [
    "PPYOLOE_DEFAULT_CONF",
    "PPYOLOE_DEFAULT_IOU",
    "PPYOLOE_PRE_NMS_TOP_K",
    "PPYOLOE_MAX_PREDICTIONS",
    "extract_decoded_predictions",
    "postprocess",
]


PPYOLOE_DEFAULT_CONF = 0.5
PPYOLOE_DEFAULT_IOU = 0.7
PPYOLOE_PRE_NMS_TOP_K = 1024
PPYOLOE_MAX_PREDICTIONS = 300


def extract_decoded_predictions(output: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pull ``(boxes, scores)`` out of any PP-YOLOE forward return shape.

    Eager ``eval()`` returns ``((boxes, scores), raw)``; ``LibrePPYOLOE._forward``
    normalises that to a named-slot mapping; the export wrapper and traced
    graphs return the ``(boxes, scores)`` pair alone.
    """
    if isinstance(output, Mapping):
        if "boxes" in output and "scores" in output:
            return output["boxes"], output["scores"]
        raise ValueError(
            "PP-YOLOE output mapping is missing the 'boxes'/'scores' slots; "
            f"got keys {sorted(output)}."
        )
    if isinstance(output, (tuple, list)) and len(output) == 2:
        first, second = output
        if torch.is_tensor(first) and torch.is_tensor(second):
            if first.shape[-1] == 4:
                return first, second
            if second.shape[-1] == 4:
                return second, first
            raise ValueError(
                "Ambiguous PP-YOLOE output: neither tensor has 4 box columns "
                f"(shapes {tuple(first.shape)}, {tuple(second.shape)})."
            )
        if isinstance(first, (tuple, list)) and len(first) == 2:
            return extract_decoded_predictions(first)
    raise ValueError(f"Unsupported PP-YOLOE output format: {type(output)!r}")


def postprocess(
    output: Any,
    conf_thres: float = PPYOLOE_DEFAULT_CONF,
    iou_thres: float = PPYOLOE_DEFAULT_IOU,
    input_size: Union[int, Tuple[int, int], Sequence[int]] = 640,
    original_size: Tuple[int, int] | None = None,
    ratio: float = 1.0,  # unused; kept for signature parity across families
    max_det: int = PPYOLOE_MAX_PREDICTIONS,
    nms_top_k: int = PPYOLOE_PRE_NMS_TOP_K,
    multi_label: bool = True,
    class_agnostic_nms: bool = False,
    batch_index: int = 0,
) -> dict:
    """Decode one image's PP-YOLOE detections to original-image coordinates."""
    tvo = torchvision.ops

    _ = ratio
    pred_bboxes, pred_scores = extract_decoded_predictions(output)
    if pred_bboxes.ndim == 3:
        pred_bboxes = pred_bboxes[batch_index]
        pred_scores = pred_scores[batch_index]

    # torchvision NMS has no fp16 CPU kernel; upstream casts for the same reason.
    pred_bboxes = pred_bboxes.float()
    pred_scores = pred_scores.float()

    if multi_label:
        anchor_idx, class_idx = (pred_scores > conf_thres).nonzero(as_tuple=False).T
        boxes = pred_bboxes[anchor_idx]
        scores = pred_scores[anchor_idx, class_idx]
        classes = class_idx
    else:
        scores, classes = torch.max(pred_scores, dim=1)
        keep = scores >= conf_thres
        boxes, scores, classes = pred_bboxes[keep], scores[keep], classes[keep]

    if scores.numel() == 0:
        return {"boxes": [], "scores": [], "classes": [], "num_detections": 0}

    if nms_top_k and scores.size(0) > nms_top_k:
        top = torch.topk(scores, k=nms_top_k, largest=True).indices
        boxes, scores, classes = boxes[top], scores[top], classes[top]

    if class_agnostic_nms:
        keep = tvo.nms(boxes, scores, iou_threshold=iou_thres)
    else:
        keep = tvo.batched_nms(boxes, scores, classes, iou_threshold=iou_thres)
    # batched_nms returns indices ordered by descending score, so the head of
    # the list is already the top-``max_det`` selection upstream applies.
    if max_det and keep.numel() > max_det:
        keep = keep[:max_det]
    boxes, scores, classes = boxes[keep], scores[keep], classes[keep]

    if original_size is not None:
        input_h, input_w = _input_size_hw(input_size)
        orig_w, orig_h = original_size
        # Stretch resize, not letterbox: x and y scale independently.
        boxes = boxes.clone()
        boxes[:, [0, 2]] *= orig_w / input_w
        boxes[:, [1, 3]] *= orig_h / input_h
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, orig_w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, orig_h)

    return {
        "boxes": boxes.detach().cpu().numpy().tolist(),
        "scores": scores.detach().cpu().numpy().tolist(),
        "classes": classes.detach().cpu().numpy().astype(int).tolist(),
        "num_detections": int(boxes.shape[0]),
    }
