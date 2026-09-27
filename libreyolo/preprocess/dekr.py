"""DEKR pose preprocessing.

Reproduces the pinned SuperGradients validation pipeline for
``coco2017_pose_dekr_w32_no_dc``:
``KeypointsLongestMaxSize(640, 640)`` ->
``KeypointsPadIfNeeded(640, 640, image_pad_value=127, padding_mode="bottom_right")`` ->
``KeypointsImageStandardize(max_value=255)`` ->
``KeypointsImageNormalize(ImageNet mean/std)``.

Conventions adapted from ``Deci-AI/super-gradients`` commit
``63de22c404d5740f34f7706c302b37fce3c8fe5d`` (Apache-2.0), files
``training/transforms/keypoints/keypoints_longest_max_size.py`` and
``keypoints_pad_if_needed.py``.

Padding is anchored top-left (upstream ``bottom_right`` mode pads only the
bottom and right edges), so restoring a coordinate is a pure division by the
resize scale with no padding offset to subtract.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from ..utils.image_loader import ImageInput, ImageLoader

__all__ = [
    "DEKR_IMAGENET_MEAN",
    "DEKR_IMAGENET_STD",
    "DEKR_PAD_VALUE",
    "preprocess_image",
    "preprocess_numpy",
]

DEKR_PAD_VALUE = 127
DEKR_IMAGENET_MEAN = (0.485, 0.456, 0.406)
DEKR_IMAGENET_STD = (0.229, 0.224, 0.225)


def preprocess_numpy(
    img_rgb_hwc: np.ndarray,
    input_size: int = 640,
    pad_value: int = DEKR_PAD_VALUE,
) -> Tuple[np.ndarray, float]:
    """Resize, pad, standardize and normalize one RGB HWC image.

    Returns ``(CHW float32, scale)``. ``scale`` is the single longest-side
    resize factor; original-canvas coordinates are ``padded_xy / scale``.
    """
    import cv2

    if img_rgb_hwc.ndim != 3 or img_rgb_hwc.shape[2] != 3:
        raise ValueError(
            f"DEKR expects an RGB HWC image with three channels, got shape "
            f"{img_rgb_hwc.shape}"
        )
    if isinstance(input_size, (list, tuple)):
        target_h, target_w = int(input_size[0]), int(input_size[1])
    else:
        target_h = target_w = int(input_size)
    if target_h <= 0 or target_w <= 0:
        raise ValueError(f"input_size must be positive, got {input_size!r}")

    height, width = img_rgb_hwc.shape[:2]
    scale = min(target_h / height, target_w / width)

    resized = img_rgb_hwc
    if scale != 1.0:
        # int(dim * scale + 0.5) is upstream's round-half-up, not np.round's
        # round-half-to-even; the two disagree on exact .5 cases.
        new_h, new_w = (int(dim * scale + 0.5) for dim in (height, width))
        resized = cv2.resize(
            img_rgb_hwc, dsize=(new_w, new_h), interpolation=cv2.INTER_LINEAR
        )

    pad_bottom = max(0, target_h - resized.shape[0])
    pad_right = max(0, target_w - resized.shape[1])
    if pad_bottom or pad_right:
        resized = cv2.copyMakeBorder(
            resized,
            top=0,
            bottom=pad_bottom,
            left=0,
            right=pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=(pad_value, pad_value, pad_value),
        )

    chw = np.ascontiguousarray(resized.transpose(2, 0, 1), dtype=np.float32)
    chw /= 255.0
    mean = np.asarray(DEKR_IMAGENET_MEAN, dtype=np.float32).reshape(3, 1, 1)
    std = np.asarray(DEKR_IMAGENET_STD, dtype=np.float32).reshape(3, 1, 1)
    chw -= mean
    chw /= std
    return chw, float(scale)


def preprocess_image(
    image: ImageInput,
    input_size: int = 640,
    color_format: str = "auto",
):
    """Load and preprocess one image for the native DEKR inference path."""
    import torch

    img = ImageLoader.load(image, color_format=color_format)
    original_size = img.size
    original_img = img.copy()
    chw, scale = preprocess_numpy(np.asarray(img), input_size=input_size)
    return torch.from_numpy(chw).unsqueeze(0), original_img, original_size, scale
