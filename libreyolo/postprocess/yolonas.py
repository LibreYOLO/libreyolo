"""YOLO-NAS postprocessing (detect + pose).

YOLO-NAS letterboxes by resizing the longest side to ``resize_size``
(636 for detect — NOT the 640 canvas) and center-padding to the canvas,
so the inverse transform must subtract the centered pad offset computed
from ``resize_size`` before dividing by the ratio.

Moved verbatim from ``libreyolo/models/yolonas/utils.py``, which re-exports
everything here for backward compatibility.
"""

from __future__ import annotations

from typing import Tuple, Union

from ..utils.lazy import lazy_module

from .common import _input_size_hw


# torch/torchvision are resolved on first use so this module stays
# importable in a torch-free ONNX deployment (discussions/711).
torch = lazy_module("torch")
torchvision = lazy_module("torchvision")

YOLO_NAS_RESIZE_SIZE = 636
YOLO_NAS_POSE_RESIZE_SIZE = 640
YOLO_NAS_OBB_RESIZE_SIZE = 1024
YOLO_NAS_PRE_NMS_TOP_K = 1000
# Upstream YoloNASRPostPredictionCallback keeps at most 1000 candidates before
# rotated NMS (recipes/.../dota_yolo_nas_r_s.yaml).
YOLO_NAS_OBB_PRE_NMS_TOP_K = 1000


def _extract_decoded_predictions(output):
    if isinstance(output, dict):
        boxes = output["boxes"]
        scores = output["scores"]
        return boxes, scores

    if isinstance(output, tuple):
        if len(output) == 2 and isinstance(output[0], tuple):
            return output[0]
        if len(output) == 2 and all(isinstance(x, torch.Tensor) for x in output):
            return output

    raise TypeError(
        f"Unsupported YOLO-NAS output format for postprocess: {type(output)!r}"
    )


def postprocess(
    output,
    conf_thres: float = 0.01,
    iou_thres: float = 0.7,
    input_size: int = 640,
    original_size: Tuple[int, int] | None = None,
    max_det: int = 300,
    letterbox: bool = True,
    resize_size: int = YOLO_NAS_RESIZE_SIZE,
    pre_nms_top_k: int = YOLO_NAS_PRE_NMS_TOP_K,
    **kwargs,
):
    boxes, scores = _extract_decoded_predictions(output)

    if boxes.dim() == 3:
        boxes = boxes[0]
    if scores.dim() == 3:
        scores = scores[0]

    max_scores, class_ids = torch.max(scores, dim=1)
    mask = max_scores > conf_thres
    if not mask.any():
        return {"boxes": [], "scores": [], "classes": [], "num_detections": 0}

    boxes = boxes[mask]
    scores = max_scores[mask]
    class_ids = class_ids[mask]

    if pre_nms_top_k and scores.numel() > pre_nms_top_k:
        topk = scores.topk(pre_nms_top_k)
        scores = topk.values
        boxes = boxes[topk.indices]
        class_ids = class_ids[topk.indices]

    if original_size is not None:
        input_h, input_w = _input_size_hw(input_size)
        effective_resize = min(resize_size, input_h, input_w)
        if letterbox:
            orig_w, orig_h = original_size
            r = min(effective_resize / orig_h, effective_resize / orig_w)
            new_w = int(round(orig_w * r))
            new_h = int(round(orig_h * r))
            offset_x = (input_w - new_w) // 2
            offset_y = (input_h - new_h) // 2
            boxes = boxes.clone()
            boxes[:, 0::2] = (boxes[:, 0::2] - offset_x) / r
            boxes[:, 1::2] = (boxes[:, 1::2] - offset_y) / r
        else:
            scale_x = original_size[0] / input_w
            scale_y = original_size[1] / input_h
            boxes = boxes.clone()
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y

        boxes[:, [0, 2]] = torch.clamp(boxes[:, [0, 2]], 0, original_size[0])
        boxes[:, [1, 3]] = torch.clamp(boxes[:, [1, 3]], 0, original_size[1])

        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        valid = (widths > 0) & (heights > 0)
        if not valid.all():
            boxes = boxes[valid]
            scores = scores[valid]
            class_ids = class_ids[valid]

    if boxes.numel() == 0:
        return {"boxes": [], "scores": [], "classes": [], "num_detections": 0}

    keep = torchvision.ops.batched_nms(boxes, scores, class_ids, iou_thres)
    if keep.numel() > max_det:
        keep = keep[:max_det]

    return {
        "boxes": boxes[keep].cpu(),
        "scores": scores[keep].cpu(),
        "classes": class_ids[keep].cpu(),
        "num_detections": int(keep.numel()),
    }


# ---------------------------------------------------------------------------
# Pose postprocess
# ---------------------------------------------------------------------------


def _undo_letterbox_xyxy(
    boxes: torch.Tensor,
    input_size: Union[int, Tuple[int, int]],
    original_size: Tuple[int, int],
    resize_size: int,
    padding_mode: str = "center",
) -> torch.Tensor:
    input_h, input_w = _input_size_hw(input_size)
    orig_w, orig_h = original_size
    resize_size = min(resize_size, input_h, input_w)
    r = min(resize_size / orig_h, resize_size / orig_w)
    new_w = int(round(orig_w * r))
    new_h = int(round(orig_h * r))
    if padding_mode == "bottom_right":
        offset_x = 0
        offset_y = 0
    else:
        offset_x = (input_w - new_w) // 2
        offset_y = (input_h - new_h) // 2
    boxes = boxes.clone()
    boxes[:, 0::2] = (boxes[:, 0::2] - offset_x) / r
    boxes[:, 1::2] = (boxes[:, 1::2] - offset_y) / r
    return boxes


def _undo_letterbox_xy(
    points: torch.Tensor,
    input_size: Union[int, Tuple[int, int]],
    original_size: Tuple[int, int],
    resize_size: int,
    padding_mode: str = "center",
) -> torch.Tensor:
    """Map ``(..., 2)`` points from letterbox space back to original-image pixels."""
    input_h, input_w = _input_size_hw(input_size)
    orig_w, orig_h = original_size
    resize_size = min(resize_size, input_h, input_w)
    r = min(resize_size / orig_h, resize_size / orig_w)
    new_w = int(round(orig_w * r))
    new_h = int(round(orig_h * r))
    if padding_mode == "bottom_right":
        offset_x = 0
        offset_y = 0
    else:
        offset_x = (input_w - new_w) // 2
        offset_y = (input_h - new_h) // 2
    pts = points.clone()
    pts[..., 0] = (pts[..., 0] - offset_x) / r
    pts[..., 1] = (pts[..., 1] - offset_y) / r
    return pts


def postprocess_pose(
    output,
    conf_thres: float = 0.01,
    iou_thres: float = 0.7,
    input_size: int = 640,
    original_size: Tuple[int, int] | None = None,
    pre_nms_max_predictions: int = 1000,
    post_nms_max_predictions: int = 300,
    letterbox: bool = True,
    resize_size: int = YOLO_NAS_POSE_RESIZE_SIZE,
    padding_mode: str = "bottom_right",
    **_,
):
    """Pose postprocess: top-K + per-image NMS + letterbox-aware decode.

    Mirrors super-gradients' ``YoloNASPosePostPredictionCallback`` but returns
    the LibreYOLO detection-shaped dict (``boxes``, ``scores``, ``classes``,
    ``num_detections``) plus a new ``keypoints`` key with shape
    ``(N, num_keypoints, 3)`` carrying ``(x, y, visibility)`` in original-image
    pixel coordinates.
    """
    if isinstance(output, dict):
        bboxes = output["boxes"]
        scores = output["scores"]
        pose_xy = output["keypoints_xy"]
        pose_conf = output["keypoints_conf"]
    elif (
        isinstance(output, tuple) and len(output) == 2 and isinstance(output[0], tuple)
    ):
        bboxes, scores, pose_xy, pose_conf = output[0]
    else:
        bboxes, scores, pose_xy, pose_conf = output

    if bboxes.dim() == 3:
        bboxes = bboxes[0]
        scores = scores[0]
        pose_xy = pose_xy[0]
        pose_conf = pose_conf[0]

    # scores: [A, nc]. Single-class pose keeps the exact historical path
    # (class-agnostic NMS, all-zero classes); multi-class pose takes the
    # highest-scoring class per anchor and runs per-class NMS.
    multiclass = scores.shape[-1] > 1
    if multiclass:
        scores, class_ids = scores.max(dim=-1)
    else:
        scores = scores.squeeze(-1)
        class_ids = None
    # `>=` matches super-gradients' YoloNASPosePostPredictionCallback boundary.
    mask = scores >= conf_thres
    if not mask.any():
        return {
            "boxes": torch.zeros((0, 4)),
            "scores": torch.zeros((0,)),
            "classes": torch.zeros((0,), dtype=torch.long),
            "num_detections": 0,
            "keypoints": torch.zeros((0, pose_xy.shape[-2], 3)),
        }

    bboxes = bboxes[mask].float()
    scores = scores[mask].float()
    pose_xy = pose_xy[mask].float()
    pose_conf = pose_conf[mask].float()
    if class_ids is not None:
        class_ids = class_ids[mask]

    if pre_nms_max_predictions and scores.numel() > pre_nms_max_predictions:
        topk = scores.topk(pre_nms_max_predictions)
        scores = topk.values
        bboxes = bboxes[topk.indices]
        pose_xy = pose_xy[topk.indices]
        pose_conf = pose_conf[topk.indices]
        if class_ids is not None:
            class_ids = class_ids[topk.indices]

    if original_size is not None:
        input_h, input_w = _input_size_hw(input_size)
        if letterbox:
            bboxes = _undo_letterbox_xyxy(
                bboxes, input_size, original_size, resize_size, padding_mode
            )
            pose_xy = _undo_letterbox_xy(
                pose_xy, input_size, original_size, resize_size, padding_mode
            )
        else:
            scale_x = original_size[0] / input_w
            scale_y = original_size[1] / input_h
            bboxes = bboxes.clone()
            bboxes[:, [0, 2]] *= scale_x
            bboxes[:, [1, 3]] *= scale_y
            pose_xy = pose_xy.clone()
            pose_xy[..., 0] *= scale_x
            pose_xy[..., 1] *= scale_y

        bboxes[:, [0, 2]] = torch.clamp(bboxes[:, [0, 2]], 0, original_size[0])
        bboxes[:, [1, 3]] = torch.clamp(bboxes[:, [1, 3]], 0, original_size[1])

        widths = bboxes[:, 2] - bboxes[:, 0]
        heights = bboxes[:, 3] - bboxes[:, 1]
        valid = (widths > 0) & (heights > 0)
        if not valid.all():
            bboxes = bboxes[valid]
            scores = scores[valid]
            pose_xy = pose_xy[valid]
            pose_conf = pose_conf[valid]
            if class_ids is not None:
                class_ids = class_ids[valid]

    if bboxes.numel() == 0:
        return {
            "boxes": torch.zeros((0, 4)),
            "scores": torch.zeros((0,)),
            "classes": torch.zeros((0,), dtype=torch.long),
            "num_detections": 0,
            "keypoints": torch.zeros((0, pose_xy.shape[-2], 3)),
        }

    if class_ids is not None:
        keep = torchvision.ops.batched_nms(bboxes, scores, class_ids, iou_thres)
    else:
        keep = torchvision.ops.nms(bboxes, scores, iou_thres)
    if keep.numel() > post_nms_max_predictions:
        keep = keep[:post_nms_max_predictions]

    bboxes = bboxes[keep].cpu()
    scores = scores[keep].cpu()
    pose_xy = pose_xy[keep].cpu()
    pose_conf = pose_conf[keep].cpu()
    keypoints = torch.cat([pose_xy, pose_conf.unsqueeze(-1)], dim=-1)
    if class_ids is not None:
        classes = class_ids[keep].cpu().long()
    else:
        classes = torch.zeros(scores.shape[0], dtype=torch.long)

    return {
        "boxes": bboxes,
        "scores": scores,
        "classes": classes,
        "num_detections": int(keep.numel()),
        "keypoints": keypoints,
    }


# ---------------------------------------------------------------------------
# OBB postprocess (YOLO-NAS-R)
# ---------------------------------------------------------------------------


def _extract_obb_predictions(output):
    """Accept the eager ``(decoded, raw)`` pair, the traced pair, or a dict."""
    if isinstance(output, dict):
        return output["boxes"], output["scores"]
    if isinstance(output, tuple):
        if len(output) == 2 and isinstance(output[0], tuple):
            return output[0]
        if len(output) == 2:
            return output
    raise TypeError(
        f"Unsupported YOLO-NAS OBB output format for postprocess: {type(output)!r}"
    )


def postprocess_obb(
    output,
    conf_thres: float = 0.1,
    iou_thres: float = 0.25,
    input_size: int = 1024,
    original_size: Tuple[int, int] | None = None,
    max_det: int = 300,
    letterbox: bool = True,
    resize_size: int = YOLO_NAS_OBB_RESIZE_SIZE,
    pre_nms_top_k: int = YOLO_NAS_OBB_PRE_NMS_TOP_K,
    **kwargs,
):
    """Decode YOLO-NAS-R rotated predictions into the public OBB contract.

    The head emits ``boxes [B, A, 5]`` as ``cx, cy, w, h, r`` in *model input*
    pixels with the upstream angle range ``[-3*pi/4, pi/4]``, plus sigmoid
    ``scores [B, A, C]``. This function:

    1. thresholds on confidence and keeps at most ``pre_nms_top_k`` candidates
       (upstream default 1000);
    2. maps boxes back onto the original image canvas by undoing the
       longest-side rescale and bottom-right padding (a uniform scale, so the
       rectangle stays a rectangle and the angle is unchanged);
    3. canonicalises to LibreYOLO's long-side ``[-pi/2, pi/2)`` contract --
       a representation change only, the polygon is untouched;
    4. runs the shared exact rotated NMS.

    Note this is *not* upstream's Gaussian matrix soft-NMS: LibreYOLO's OBB
    contract uses one exact rotated NMS across families (see
    ``libreyolo/postprocess/obb_ops.py``). Raw pre-NMS parity against upstream
    is what ``weights/parity_yolonas_obb.py`` gates on.
    """
    from .obb_ops import (
        canonicalize_xywhr_tensor,
        rotated_nms_keep_indices,
        xywhr_to_xyxy,
    )

    empty = {
        "boxes": [],
        "scores": [],
        "classes": [],
        "obb": [],
        "num_detections": 0,
    }

    boxes, scores = _extract_obb_predictions(output)
    if boxes.dim() == 3:
        boxes = boxes[0]
    if scores.dim() == 3:
        scores = scores[0]

    max_scores, class_ids = torch.max(scores, dim=1)
    # ``>=`` matches upstream YoloNASRPostPredictionCallback's boundary.
    mask = max_scores >= conf_thres
    if not mask.any():
        return empty

    xywhr = boxes[mask].float()
    max_scores = max_scores[mask].float()
    class_ids = class_ids[mask]

    if pre_nms_top_k and max_scores.numel() > pre_nms_top_k:
        topk = max_scores.topk(pre_nms_top_k)
        max_scores = topk.values
        xywhr = xywhr[topk.indices]
        class_ids = class_ids[topk.indices]

    if original_size is not None:
        input_h, input_w = _input_size_hw(input_size)
        if letterbox:
            orig_w, orig_h = original_size
            effective_resize = min(resize_size, input_h, input_w)
            ratio = min(effective_resize / orig_h, effective_resize / orig_w)
            # Bottom-right padding: no offset to subtract, and the rescale is
            # uniform, so centres and sides scale while the angle is invariant.
            xywhr = xywhr.clone()
            xywhr[:, :4] = xywhr[:, :4] / ratio
        else:
            scale_x = original_size[0] / input_w
            scale_y = original_size[1] / input_h
            if abs(scale_x - scale_y) > 1e-6:
                raise ValueError(
                    "YOLO-NAS OBB requires an aspect-preserving resize: "
                    "non-uniform x/y scaling turns a rotated rectangle into a "
                    "parallelogram. Use letterbox=True."
                )
            xywhr = xywhr.clone()
            xywhr[:, :4] *= scale_x

        xywhr[:, 0].clamp_(0, original_size[0])
        xywhr[:, 1].clamp_(0, original_size[1])

    valid = (xywhr[:, 2] > 0) & (xywhr[:, 3] > 0)
    if not valid.any():
        return empty
    if not valid.all():
        xywhr = xywhr[valid]
        max_scores = max_scores[valid]
        class_ids = class_ids[valid]

    xywhr = canonicalize_xywhr_tensor(xywhr)

    keep = rotated_nms_keep_indices(xywhr, max_scores, class_ids, iou_thres, max_det)
    if len(keep) == 0:
        return empty

    xywhr = xywhr[keep]
    scores_out = max_scores[keep]
    classes_out = class_ids[keep]

    aabb = xywhr_to_xyxy(xywhr)
    if original_size is not None:
        aabb[:, [0, 2]] = torch.clamp(aabb[:, [0, 2]], 0, original_size[0])
        aabb[:, [1, 3]] = torch.clamp(aabb[:, [1, 3]], 0, original_size[1])

    obb_out = torch.cat(
        (xywhr, scores_out[:, None], classes_out[:, None].float()), dim=1
    )
    return {
        "boxes": aabb.detach().cpu(),
        "scores": scores_out.detach().cpu(),
        "classes": classes_out.detach().cpu().long(),
        "obb": obb_out.detach().cpu(),
        "num_detections": int(xywhr.shape[0]),
    }
