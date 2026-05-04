"""LibreEC preprocessing and postprocessing helpers.

EC's input pipeline is square resize (no letterbox), divide by 255, then
ImageNet (mean, std) normalization. The ImageNet normalization is what
distinguishes EC's preprocess from D-FINE's. Output postprocessing is
DETR-style top-K (no NMS).
"""

from __future__ import annotations

from typing import Any, Mapping, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ...utils.image_loader import ImageInput, ImageLoader
from ..dfine.box_ops import box_cxcywh_to_xyxy


def unwrap_ec_checkpoint(checkpoint: Mapping | Any):
    """Extract the model state_dict from upstream/Libre EC checkpoint formats."""
    if not isinstance(checkpoint, Mapping):
        return checkpoint
    ema = checkpoint.get("ema")
    if isinstance(ema, Mapping):
        module = ema.get("module")
        if isinstance(module, Mapping):
            return module
    for key in ("model", "state_dict"):
        v = checkpoint.get(key)
        if isinstance(v, Mapping):
            return v
    return checkpoint


_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_numpy(
    img_rgb_hwc: np.ndarray, input_size: int = 640
) -> Tuple[np.ndarray, float]:
    """EC preprocess: square resize + /255 + ImageNet (mean, std).

    Mirrors upstream val transforms (`Resize -> ConvertPILImage(scale=True) ->
    Normalize(IMAGENET)`). The ImageNet normalization is what distinguishes
    EC's preprocess from D-FINE's; missing it costs ~2 mAP on COCO val.
    """
    img_resized = Image.fromarray(img_rgb_hwc).resize(
        (input_size, input_size), Image.Resampling.BILINEAR
    )
    arr = np.array(img_resized, dtype=np.float32) / 255.0
    arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
    return arr.transpose(2, 0, 1), 1.0


def preprocess_image(
    image: ImageInput,
    input_size: int = 640,
    color_format: str = "auto",
) -> Tuple[torch.Tensor, Image.Image, Tuple[int, int], float]:
    img = ImageLoader.load(image, color_format=color_format)
    original_size = img.size
    original_img = img.copy()
    chw, ratio = preprocess_numpy(np.array(img), input_size=input_size)
    return torch.from_numpy(chw).unsqueeze(0), original_img, original_size, ratio


def postprocess_seg(
    outputs,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    original_size: Tuple[int, int] | None = None,
    max_det: int = 300,
    **_unused,
):
    """Decode ECSeg output dict into LibreYOLO detections + masks.

    Mirrors upstream's ``PostProcessor`` for the seg branch:
      1. top-K over flattened (Q × C) sigmoid scores
      2. gather corresponding boxes (cxcywh → xyxy, scaled by orig size)
      3. gather corresponding mask logits, interpolate to (orig_h, orig_w),
         threshold at 0
    """
    out_logits = outputs["pred_logits"]
    out_bbox = outputs["pred_boxes"]
    mask_pred = outputs.get("pred_masks")
    if out_logits.dim() == 3:
        out_logits = out_logits[0]
        out_bbox = out_bbox[0]
        if mask_pred is not None:
            mask_pred = mask_pred[0]

    num_classes = out_logits.shape[-1]
    prob = out_logits.sigmoid()
    topk_values, topk_indices = torch.topk(prob.view(-1), min(max_det, prob.numel()))
    scores = topk_values
    query_idx = topk_indices // num_classes
    class_idx = topk_indices % num_classes

    boxes_xyxy = box_cxcywh_to_xyxy(out_bbox)
    boxes = boxes_xyxy[query_idx]

    keep = scores > conf_thres
    scores = scores[keep]
    class_idx = class_idx[keep]
    boxes = boxes[keep]
    sel_query = query_idx[keep]

    masks = None
    if original_size is not None:
        ow, oh = original_size
        scale = torch.tensor([ow, oh, ow, oh], dtype=boxes.dtype, device=boxes.device)
        boxes = boxes * scale
        if mask_pred is not None and sel_query.numel() > 0:
            sel_masks = mask_pred[sel_query]  # (N, H, W)
            sel_masks = F.interpolate(
                sel_masks.unsqueeze(1).float(),
                size=(int(oh), int(ow)),
                mode="bilinear",
                align_corners=False,
            )
            masks = (sel_masks.squeeze(1) > 0.0).cpu()

    out = {
        "num_detections": int(boxes.shape[0]),
        "boxes": boxes.cpu().numpy()
        if boxes.numel() > 0
        else np.zeros((0, 4), dtype=np.float32),
        "scores": scores.cpu().numpy()
        if scores.numel() > 0
        else np.zeros((0,), dtype=np.float32),
        "classes": class_idx.cpu().numpy()
        if class_idx.numel() > 0
        else np.zeros((0,), dtype=np.int64),
    }
    if masks is not None:
        out["masks"] = masks
    return out


def postprocess_pose(
    outputs,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    original_size: Tuple[int, int] | None = None,
    max_det: int = 60,
    num_keypoints: int = 17,
    **_unused,
):
    """Decode ECPose output dict into LibreYOLO detections + keypoints.

    Mirrors upstream's ``DETRPosePostProcessor``: top-K over flattened
    (num_queries * num_classes) class logits, gather the corresponding
    keypoints, append visibility=1 to each. No NMS — DETR-style.

    Returns the same dict shape as EC's ``postprocess`` plus a ``keypoints``
    tensor of shape ``(N, num_keypoints, 3)`` with (x, y, vis).
    """
    out_logits = outputs["pred_logits"]
    out_kpts = outputs["pred_keypoints"]  # (B, Q, K*2) flattened
    if out_logits.dim() == 3:
        out_logits = out_logits[0]
        out_kpts = out_kpts[0]

    num_classes = out_logits.shape[-1]
    prob = out_logits.sigmoid()
    topk_values, topk_indices = torch.topk(prob.view(-1), min(max_det, prob.numel()))
    scores = topk_values
    query_idx = topk_indices // num_classes
    class_idx = topk_indices % num_classes

    kpts_xy = out_kpts.unflatten(-1, (num_keypoints, 2))[query_idx]  # (N, K, 2) in [0,1]

    keep = scores >= conf_thres
    scores = scores[keep]
    class_idx = class_idx[keep]
    kpts_xy = kpts_xy[keep]

    if original_size is not None and kpts_xy.numel() > 0:
        ow, oh = original_size
        scale = torch.tensor([ow, oh], dtype=kpts_xy.dtype, device=kpts_xy.device)
        kpts_xy = kpts_xy * scale
        # Box bbox is the per-instance keypoint extent (DETR pose has no box head).
        x_min = kpts_xy[..., 0].min(dim=-1).values
        y_min = kpts_xy[..., 1].min(dim=-1).values
        x_max = kpts_xy[..., 0].max(dim=-1).values
        y_max = kpts_xy[..., 1].max(dim=-1).values
        boxes = torch.stack([x_min, y_min, x_max, y_max], dim=-1)
    else:
        boxes = torch.zeros((kpts_xy.shape[0], 4), dtype=kpts_xy.dtype, device=kpts_xy.device)

    # Append visibility=1 channel — the DETR pose head only predicts xy.
    vis = torch.ones((*kpts_xy.shape[:-1], 1), dtype=kpts_xy.dtype, device=kpts_xy.device)
    keypoints = torch.cat([kpts_xy, vis], dim=-1)

    return {
        "num_detections": int(boxes.shape[0]),
        "boxes": boxes.cpu().numpy()
        if boxes.numel() > 0
        else np.zeros((0, 4), dtype=np.float32),
        "scores": scores.cpu().numpy()
        if scores.numel() > 0
        else np.zeros((0,), dtype=np.float32),
        "classes": np.zeros((scores.shape[0],), dtype=np.int64),
        "keypoints": keypoints.cpu().numpy()
        if keypoints.numel() > 0
        else np.zeros((0, num_keypoints, 3), dtype=np.float32),
    }


def postprocess(
    outputs,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    original_size: Tuple[int, int] | None = None,
    max_det: int = 300,
    **_unused,
):
    """Decode EC output dict into LibreYOLO detections dict (DETR-style top-K)."""
    out_logits = outputs["pred_logits"]
    out_bbox = outputs["pred_boxes"]
    if out_logits.dim() == 3:
        out_logits = out_logits[0]
        out_bbox = out_bbox[0]

    num_classes = out_logits.shape[-1]
    prob = out_logits.sigmoid()
    topk_values, topk_indices = torch.topk(prob.view(-1), min(max_det, prob.numel()))
    scores = topk_values
    query_idx = topk_indices // num_classes
    class_idx = topk_indices % num_classes

    boxes_xyxy = box_cxcywh_to_xyxy(out_bbox)
    boxes = boxes_xyxy[query_idx]

    keep = scores > conf_thres
    scores = scores[keep]
    class_idx = class_idx[keep]
    boxes = boxes[keep]

    if original_size is not None:
        ow, oh = original_size
        scale = torch.tensor([ow, oh, ow, oh], dtype=boxes.dtype, device=boxes.device)
        boxes = boxes * scale

    return {
        "num_detections": int(boxes.shape[0]),
        "boxes": boxes.cpu().numpy()
        if boxes.numel() > 0
        else np.zeros((0, 4), dtype=np.float32),
        "scores": scores.cpu().numpy()
        if scores.numel() > 0
        else np.zeros((0,), dtype=np.float32),
        "classes": class_idx.cpu().numpy()
        if class_idx.numel() > 0
        else np.zeros((0,), dtype=np.int64),
    }
