"""YOLO-NAS input preprocessing.

Moved verbatim from ``libreyolo/models/yolonas/utils.py``, which re-exports
everything here for backward compatibility. Lives outside ``models/`` so the
ONNX backend can import it without pulling torch; see the package docstring.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from PIL import Image

from ..postprocess.yolonas import (
    YOLO_NAS_OBB_RESIZE_SIZE,
    YOLO_NAS_POSE_RESIZE_SIZE,
    YOLO_NAS_RESIZE_SIZE,
)
from ..utils.image_loader import ImageInput, ImageLoader
from . import as_batched_input

YOLO_NAS_POSE_PAD_VALUE = 127
YOLO_NAS_OBB_PAD_VALUE = 114


def preprocess_numpy(
    img_rgb_hwc: np.ndarray,
    input_size: int = 640,
    pad_value: int = 114,
    resize_size: int = YOLO_NAS_RESIZE_SIZE,
    padding_mode: str = "center",
) -> Tuple[np.ndarray, float]:
    """Resize longest side to ``resize_size``, center-pad to ``input_size``."""
    orig_h, orig_w = img_rgb_hwc.shape[:2]
    if isinstance(input_size, (list, tuple)):
        input_h, input_w = int(input_size[0]), int(input_size[1])
        if input_h != input_w:
            # YOLO-NAS preprocessing resizes the longest side to a fixed
            # ``resize_size`` before padding, so a rectangular canvas would
            # either crop the image or leave most of it as padding. Reject
            # until the resize recipe itself is made aspect-aware.
            raise ValueError(
                f"YOLO-NAS does not support rectangular input sizes, got "
                f"({input_h}, {input_w}). Use a square imgsz."
            )
    else:
        input_h = input_w = int(input_size)
    resize_size = min(resize_size, input_h, input_w)
    ratio = min(resize_size / orig_h, resize_size / orig_w)
    new_w, new_h = int(round(orig_w * ratio)), int(round(orig_h * ratio))

    img_resized = Image.fromarray(img_rgb_hwc).resize(
        (new_w, new_h), Image.Resampling.BILINEAR
    )

    padded = Image.new("RGB", (input_w, input_h), (pad_value, pad_value, pad_value))
    if padding_mode == "bottom_right":
        offset_x = 0
        offset_y = 0
    else:
        offset_x = (input_w - new_w) // 2
        offset_y = (input_h - new_h) // 2
    padded.paste(img_resized, (offset_x, offset_y))

    arr = np.array(padded, dtype=np.float32) / 255.0
    return arr.transpose(2, 0, 1), ratio


def preprocess_image(
    image: ImageInput,
    input_size: int = 640,
    color_format: str = "auto",
):
    img = ImageLoader.load(image, color_format=color_format)
    original_size = img.size
    original_img = img.copy()

    img_chw, ratio = preprocess_numpy(np.array(img), input_size=input_size)
    return as_batched_input(img_chw), original_img, original_size, ratio


def preprocess_pose_image(
    image: ImageInput,
    input_size: int = 640,
    color_format: str = "auto",
):
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
    return as_batched_input(img_chw), original_img, original_size, ratio


def preprocess_obb_numpy(
    img_rgb_hwc: np.ndarray,
    input_size: int = YOLO_NAS_OBB_RESIZE_SIZE,
    pad_value: int = YOLO_NAS_OBB_PAD_VALUE,
    resize_size: int = YOLO_NAS_OBB_RESIZE_SIZE,
) -> Tuple[np.ndarray, float]:
    """YOLO-NAS-R (OBB) evaluation preprocessing.

    Reproduces the checkpoint's own ``processing_params`` pipeline:
    ``ReverseImageChannels`` -> ``OBBDetectionLongestMaxSizeRescale(1024)``
    -> ``OBBDetectionBottomRightPadding(1024, pad_value=114)``
    -> ``StandardizeImage(max_value=255)``.

    Unlike the detect/pose paths this resamples with ``cv2.INTER_LINEAR``
    rather than PIL, matching upstream's rescale: DOTA imagery is downscaled
    hard to reach 1024 and PIL's antialiased filter would not agree.

    Takes RGB HWC uint8 and returns ``(CHW float32 in [0, 1] BGR, ratio)``.
    """
    import cv2

    orig_h, orig_w = img_rgb_hwc.shape[:2]
    if isinstance(input_size, (list, tuple)):
        input_h, input_w = int(input_size[0]), int(input_size[1])
    else:
        input_h = input_w = int(input_size)
    if input_h != input_w:
        raise ValueError(
            f"YOLO-NAS OBB does not support rectangular input sizes, got "
            f"({input_h}, {input_w}). Use a square imgsz."
        )
    resize_size = min(int(resize_size), input_h, input_w)

    ratio = min(resize_size / orig_h, resize_size / orig_w)
    new_w, new_h = int(round(orig_w * ratio)), int(round(orig_h * ratio))

    bgr = np.ascontiguousarray(img_rgb_hwc[:, :, ::-1])
    if (new_w, new_h) != (orig_w, orig_h):
        bgr = cv2.resize(bgr, dsize=(new_w, new_h), interpolation=cv2.INTER_LINEAR)

    padded = np.full((input_h, input_w, 3), pad_value, dtype=np.uint8)
    padded[:new_h, :new_w] = bgr  # bottom-right padding

    arr = padded.astype(np.float32) / 255.0
    return arr.transpose(2, 0, 1), ratio


def preprocess_obb_image(
    image: ImageInput,
    input_size: int = YOLO_NAS_OBB_RESIZE_SIZE,
    color_format: str = "auto",
):
    img = ImageLoader.load(image, color_format=color_format)
    original_size = img.size
    original_img = img.copy()

    img_chw, ratio = preprocess_obb_numpy(np.array(img), input_size=input_size)
    return as_batched_input(img_chw), original_img, original_size, ratio


__all__ = [
    "YOLO_NAS_OBB_PAD_VALUE",
    "YOLO_NAS_POSE_PAD_VALUE",
    "preprocess_numpy",
    "preprocess_image",
    "preprocess_obb_image",
    "preprocess_obb_numpy",
    "preprocess_pose_image",
]
