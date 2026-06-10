"""YOLO-NAS preprocessing, postprocessing, and checkpoint helpers.

Postprocessing lives in ``libreyolo.postprocess.yolonas`` and is re-exported
here for backward compatibility (along with the resize-size constants it
shares with the preprocess side).
"""

from __future__ import annotations

from typing import Mapping, MutableMapping, Tuple

import numpy as np
import torch
from PIL import Image

from ...postprocess.yolonas import (  # noqa: F401  (backward-compatible re-exports)
    YOLO_NAS_PRE_NMS_TOP_K,
    YOLO_NAS_POSE_RESIZE_SIZE,
    YOLO_NAS_RESIZE_SIZE,
    _extract_decoded_predictions,
    _undo_letterbox_xy,
    _undo_letterbox_xyxy,
    postprocess,
    postprocess_pose,
)
from ...utils.image_loader import ImageInput, ImageLoader

YOLO_NAS_POSE_PAD_VALUE = 127


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
    padding_mode: str = "center",
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
    if padding_mode == "bottom_right":
        offset_x = 0
        offset_y = 0
    else:
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


def preprocess_pose_image(
    image: ImageInput,
    input_size: int = 640,
    color_format: str = "auto",
) -> Tuple[torch.Tensor, Image.Image, Tuple[int, int], float]:
    img = ImageLoader.load(image, color_format=color_format)
    original_size = img.size
    original_img = img.copy()
    rgb = np.array(img)
    bgr = rgb[:, :, ::-1]
    img_chw, ratio = preprocess_numpy(
        bgr,
        input_size=input_size,
        pad_value=YOLO_NAS_POSE_PAD_VALUE,
        resize_size=YOLO_NAS_POSE_RESIZE_SIZE,
        padding_mode="bottom_right",
    )
    img_tensor = torch.from_numpy(img_chw).unsqueeze(0)
    return img_tensor, original_img, original_size, ratio
