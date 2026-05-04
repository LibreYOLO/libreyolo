"""YOLO-NAS preprocessing, postprocessing, and checkpoint helpers."""

from __future__ import annotations

from typing import Mapping, MutableMapping, Tuple

import numpy as np
import torch
import torchvision.ops
from PIL import Image

from ...utils.image_loader import ImageInput, ImageLoader

YOLO_NAS_RESIZE_SIZE = 636
YOLO_NAS_PRE_NMS_TOP_K = 1000


def unwrap_yolonas_checkpoint(
    checkpoint: Mapping | MutableMapping,
):
    """Extract the actual state dict from common YOLO-NAS checkpoint layouts.

    Official SuperGradients checkpoints typically store weights under ``net``,
    while training checkpoints may also contain ``ema_net``. Prefer EMA weights
    when present so downstream loading mirrors SG's own behavior.
    """
    if not isinstance(checkpoint, Mapping):
        return checkpoint

    for key in ("ema_net", "net", "model", "state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, Mapping):
            return value

    return checkpoint


def preprocess_numpy(
    img_rgb_hwc: np.ndarray,
    input_size: int = 640,
    pad_value: int = 114,
    resize_size: int = YOLO_NAS_RESIZE_SIZE,
) -> Tuple[np.ndarray, float]:
    """Resize longest side to ``resize_size``, center-pad to ``input_size``."""
    orig_h, orig_w = img_rgb_hwc.shape[:2]
    ratio = min(resize_size / orig_h, resize_size / orig_w)
    new_w, new_h = int(round(orig_w * ratio)), int(round(orig_h * ratio))

    img_resized = Image.fromarray(img_rgb_hwc).resize(
        (new_w, new_h), Image.Resampling.BILINEAR
    )

    padded = Image.new(
        "RGB", (input_size, input_size), (pad_value, pad_value, pad_value)
    )
    offset_x = (input_size - new_w) // 2
    offset_y = (input_size - new_h) // 2
    padded.paste(img_resized, (offset_x, offset_y))

    arr = np.array(padded, dtype=np.float32) / 255.0
    return arr.transpose(2, 0, 1), ratio


def preprocess_image(
    image: ImageInput,
    input_size: int = 640,
    color_format: str = "auto",
) -> Tuple[torch.Tensor, Image.Image, Tuple[int, int], float]:
    img = ImageLoader.load(image, color_format=color_format)
    original_size = img.size
    original_img = img.copy()

    img_chw, ratio = preprocess_numpy(np.array(img), input_size=input_size)
    img_tensor = torch.from_numpy(img_chw).unsqueeze(0)
    return img_tensor, original_img, original_size, ratio


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
        if letterbox:
            orig_w, orig_h = original_size
            r = min(resize_size / orig_h, resize_size / orig_w)
            new_w = int(round(orig_w * r))
            new_h = int(round(orig_h * r))
            offset_x = (input_size - new_w) // 2
            offset_y = (input_size - new_h) // 2
            boxes = boxes.clone()
            boxes[:, 0::2] = (boxes[:, 0::2] - offset_x) / r
            boxes[:, 1::2] = (boxes[:, 1::2] - offset_y) / r
        else:
            scale_x = original_size[0] / input_size
            scale_y = original_size[1] / input_size
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
    input_size: int,
    original_size: Tuple[int, int],
    resize_size: int,
) -> torch.Tensor:
    orig_w, orig_h = original_size
    r = min(resize_size / orig_h, resize_size / orig_w)
    new_w = int(round(orig_w * r))
    new_h = int(round(orig_h * r))
    offset_x = (input_size - new_w) // 2
    offset_y = (input_size - new_h) // 2
    boxes = boxes.clone()
    boxes[:, 0::2] = (boxes[:, 0::2] - offset_x) / r
    boxes[:, 1::2] = (boxes[:, 1::2] - offset_y) / r
    return boxes


def _undo_letterbox_xy(
    points: torch.Tensor,
    input_size: int,
    original_size: Tuple[int, int],
    resize_size: int,
) -> torch.Tensor:
    """Map ``(..., 2)`` points from letterbox space back to original-image pixels."""
    orig_w, orig_h = original_size
    r = min(resize_size / orig_h, resize_size / orig_w)
    new_w = int(round(orig_w * r))
    new_h = int(round(orig_h * r))
    offset_x = (input_size - new_w) // 2
    offset_y = (input_size - new_h) // 2
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
    resize_size: int = YOLO_NAS_RESIZE_SIZE,
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
    else:
        bboxes, scores, pose_xy, pose_conf = output

    if bboxes.dim() == 3:
        bboxes = bboxes[0]
        scores = scores[0]
        pose_xy = pose_xy[0]
        pose_conf = pose_conf[0]

    scores = scores.squeeze(-1)
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

    if pre_nms_max_predictions and scores.numel() > pre_nms_max_predictions:
        topk = scores.topk(pre_nms_max_predictions)
        scores = topk.values
        bboxes = bboxes[topk.indices]
        pose_xy = pose_xy[topk.indices]
        pose_conf = pose_conf[topk.indices]

    if original_size is not None:
        if letterbox:
            bboxes = _undo_letterbox_xyxy(bboxes, input_size, original_size, resize_size)
            pose_xy = _undo_letterbox_xy(pose_xy, input_size, original_size, resize_size)
        else:
            scale_x = original_size[0] / input_size
            scale_y = original_size[1] / input_size
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

    if bboxes.numel() == 0:
        return {
            "boxes": torch.zeros((0, 4)),
            "scores": torch.zeros((0,)),
            "classes": torch.zeros((0,), dtype=torch.long),
            "num_detections": 0,
            "keypoints": torch.zeros((0, pose_xy.shape[-2], 3)),
        }

    keep = torchvision.ops.nms(bboxes, scores, iou_thres)
    if keep.numel() > post_nms_max_predictions:
        keep = keep[:post_nms_max_predictions]

    bboxes = bboxes[keep].cpu()
    scores = scores[keep].cpu()
    pose_xy = pose_xy[keep].cpu()
    pose_conf = pose_conf[keep].cpu()
    keypoints = torch.cat([pose_xy, pose_conf.unsqueeze(-1)], dim=-1)
    classes = torch.zeros(scores.shape[0], dtype=torch.long)

    return {
        "boxes": bboxes,
        "scores": scores,
        "classes": classes,
        "num_detections": int(keep.numel()),
        "keypoints": keypoints,
    }
