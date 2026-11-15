"""InstantHMR person-crop preprocessing and output decoding.

Ported from InstantHMR ``instanthmr/inference.py``
(https://github.com/Naturalpad-studio/InstantHMR, Apache-2.0): the square 1.2x
crop around a tight person box, ImageNet normalisation, and the CLIFF
conditioning vector that tells the network where the box sits in the frame.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CROP_EXPAND = 1.2
# Focal length assumed when the camera is not calibrated, as a multiple of the
# image diagonal (about 50 degrees of horizontal field of view on 16:9).
FOCAL_FALLBACK_DIAG = 1.05


def default_focal(h: int, w: int, cliff_focal: bool = True) -> float:
    """Focal length in pixels used when the caller does not supply one."""
    return (FOCAL_FALLBACK_DIAG if cliff_focal else 1.0) * math.hypot(h, w)


def preprocess_person(
    image_rgb: np.ndarray,
    box_xyxy,
    focal: float,
    input_size: int = 288,
    cliff_focal: bool = True,
):
    """Return ``(crop CHW float32, cliff (3,), (x0, y0, side))`` for one person box."""
    h, w = image_rgb.shape[:2]
    x1, y1, x2, y2 = np.asarray(box_xyxy, dtype=np.float64)
    cx, cy, size = (x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1)
    if cliff_focal:
        cliff = [
            math.atan((cx - w / 2) / focal),
            math.atan((cy - h / 2) / focal),
            size / focal,
        ]
    else:
        cliff = [2 * cx / w - 1, 2 * cy / h - 1, size / max(w, h)]

    side = size * CROP_EXPAND
    sx, sy = cx - side / 2, cy - side / 2
    ix1, iy1 = math.floor(sx), math.floor(sy)
    ix2, iy2 = math.ceil(sx + side), math.ceil(sy + side)
    patch = image_rgb[max(0, iy1) : min(h, iy2), max(0, ix1) : min(w, ix2)]
    patch = cv2.copyMakeBorder(
        np.ascontiguousarray(patch),
        max(0, -iy1),
        max(0, iy2 - h),
        max(0, -ix1),
        max(0, ix2 - w),
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )
    crop = cv2.resize(patch, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    crop = ((crop.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD).transpose(
        2, 0, 1
    )
    return (
        np.ascontiguousarray(crop),
        np.asarray(cliff, dtype=np.float32),
        (float(sx), float(sy), float(side)),
    )


def crop_to_image_coords(joints_2d: np.ndarray, square) -> np.ndarray:
    """Map ``(..., 2)`` keypoints in [-1, 1] over the crop to original-image pixels."""
    x0, y0, side = square
    return (joints_2d + 1.0) * 0.5 * side + np.array([x0, y0], dtype=np.float32)
