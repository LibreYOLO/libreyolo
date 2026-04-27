"""LibreECDet preprocessing and postprocessing helpers.

ECDet's input pipeline is identical to D-FINE's: square resize to 640, no
letterbox, divide by 255, no ImageNet normalization. Output postprocessing is
also DETR-style top-K (no NMS).
"""

from __future__ import annotations

from typing import Any, Mapping, Tuple

import numpy as np
import torch
from PIL import Image

from ...utils.image_loader import ImageInput, ImageLoader
from ..dfine.box_ops import box_cxcywh_to_xyxy


def unwrap_ecdet_checkpoint(checkpoint: Mapping | Any):
    """Extract the model state_dict from upstream/Libre ECDet checkpoint formats."""
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


def preprocess_numpy(img_rgb_hwc: np.ndarray, input_size: int = 640) -> Tuple[np.ndarray, float]:
    img_resized = Image.fromarray(img_rgb_hwc).resize(
        (input_size, input_size), Image.Resampling.BILINEAR
    )
    arr = np.array(img_resized, dtype=np.float32) / 255.0
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


def postprocess(
    outputs,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    original_size: Tuple[int, int] | None = None,
    max_det: int = 300,
    **_unused,
):
    """Decode ECDet output dict into LibreYOLO detections dict (DETR-style top-K)."""
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
        "boxes": boxes.cpu().numpy() if boxes.numel() > 0 else np.zeros((0, 4), dtype=np.float32),
        "scores": scores.cpu().numpy() if scores.numel() > 0 else np.zeros((0,), dtype=np.float32),
        "classes": class_idx.cpu().numpy() if class_idx.numel() > 0 else np.zeros((0,), dtype=np.int64),
    }
