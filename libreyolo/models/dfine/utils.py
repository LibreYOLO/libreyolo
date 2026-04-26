"""LibreDFINE preprocessing and postprocessing helpers."""

from __future__ import annotations

from typing import Any, Mapping, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ...utils.image_loader import ImageInput, ImageLoader
from .box_ops import box_cxcywh_to_xyxy


def unwrap_dfine_checkpoint(checkpoint: Mapping | Any):
    """Extract the state_dict from a D-FINE checkpoint.

    Upstream saves ``{"ema": {"module": state_dict, ...}, "model": state_dict, ...}``.
    Prefer EMA when present (matches upstream ``tools/inference_torch.py``).
    """
    if not isinstance(checkpoint, Mapping):
        return checkpoint

    ema = checkpoint.get("ema")
    if isinstance(ema, Mapping):
        module = ema.get("module")
        if isinstance(module, Mapping):
            return module

    for key in ("model", "state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, Mapping):
            return value

    return checkpoint


def preprocess_numpy(
    img_rgb_hwc: np.ndarray,
    input_size: int = 640,
) -> Tuple[np.ndarray, float]:
    """Preprocess an RGB HWC uint8 array to D-FINE input layout.

    Plain square resize to ``(input_size, input_size)``, no letterbox, no
    ImageNet normalization — just ``uint8 / 255``. Ratio is always 1.0
    because there's no padding.
    """
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

    img_chw, ratio = preprocess_numpy(np.array(img), input_size=input_size)
    img_tensor = torch.from_numpy(img_chw).unsqueeze(0)
    return img_tensor, original_img, original_size, ratio


def postprocess(
    outputs,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    original_size: Tuple[int, int] | None = None,
    max_det: int = 300,
    **_unused,
):
    """Decode D-FINE output dict into a LibreYOLO detections dict.

    D-FINE outputs DETR-style ``{"pred_logits", "pred_boxes"}``. Post-processing:
    sigmoid → top-K across (query × class) → box-cxcywh→xyxy → scale to orig.
    No NMS is applied (set prediction already).

    Returns dict with ``num_detections`` / ``boxes`` / ``scores`` / ``classes``.
    """
    out_logits = outputs["pred_logits"]  # (B, Q, nc)
    out_bbox = outputs["pred_boxes"]  # (B, Q, 4) cxcywh in [0, 1]

    if out_logits.dim() == 3:
        out_logits = out_logits[0]
        out_bbox = out_bbox[0]

    num_classes = out_logits.shape[-1]
    prob = out_logits.sigmoid()

    # Top-K across all (queries × classes).
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
        orig_w, orig_h = original_size
        scale = torch.tensor(
            [orig_w, orig_h, orig_w, orig_h], dtype=boxes.dtype, device=boxes.device
        )
        boxes = boxes * scale

    return {
        "num_detections": int(boxes.shape[0]),
        "boxes": boxes.cpu().numpy() if boxes.numel() > 0 else np.zeros((0, 4), dtype=np.float32),
        "scores": scores.cpu().numpy() if scores.numel() > 0 else np.zeros((0,), dtype=np.float32),
        "classes": (
            class_idx.cpu().numpy()
            if class_idx.numel() > 0
            else np.zeros((0,), dtype=np.int64)
        ),
    }
