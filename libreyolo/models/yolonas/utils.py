"""YOLO-NAS preprocessing, postprocessing, and checkpoint helpers."""

from __future__ import annotations

from typing import Mapping, MutableMapping, Tuple

import numpy as np
import torch
from PIL import Image

from ...utils.general import postprocess_detections
from ...utils.image_loader import ImageInput, ImageLoader


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
) -> Tuple[np.ndarray, float]:
    """Preprocess RGB HWC uint8 image for YOLO-NAS inference.

    This currently follows LibreYOLO's shared letterbox convention so validation
    and single-image inference scale boxes back consistently. A later parity
    pass can tighten this toward the exact SG processing pipeline.
    """
    orig_h, orig_w = img_rgb_hwc.shape[:2]
    ratio = min(input_size / orig_h, input_size / orig_w)
    new_w, new_h = int(orig_w * ratio), int(orig_h * ratio)

    img_resized = Image.fromarray(img_rgb_hwc).resize(
        (new_w, new_h), Image.Resampling.BILINEAR
    )

    padded = Image.new("RGB", (input_size, input_size), (pad_value, pad_value, pad_value))
    padded.paste(img_resized, (0, 0))

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
        "Unsupported YOLO-NAS output format for postprocess: "
        f"{type(output)!r}"
    )


def postprocess(
    output,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    input_size: int = 640,
    original_size: Tuple[int, int] | None = None,
    max_det: int = 300,
    letterbox: bool = True,
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

    return postprocess_detections(
        boxes=boxes[mask].clone(),
        scores=max_scores[mask],
        class_ids=class_ids[mask],
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        input_size=input_size,
        original_size=original_size,
        max_det=max_det,
        letterbox=letterbox,
    )
