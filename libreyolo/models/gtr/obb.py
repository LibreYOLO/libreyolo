"""GTR oriented-box (DOTA) task helpers: checkpoint detection and preprocessing.

Upstream evaluates on 1024px DOTA split patches (zero-padded at the bottom and
right) with RGB 0-1 input and ImageNet normalization. Arbitrary images are
resized to fit the square canvas without distortion, padded the same way, and
normalized; decoding is ``postprocess_obb`` (normalized ``(cx, cy, w, h,
theta/pi)``, long-edge angle in ``[0, pi)``), shared with RT-DETRv2 OBB.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from ...utils.image_loader import ImageLoader
from ...validation.preprocessors import RTDETRv2OBBValPreprocessor
from .obb_nn import DOTA_CLASSES, OBB_INPUT_SIZE, OBB_SIZES

__all__ = [
    "DOTA_NAMES",
    "OBB_INPUT_SIZES",
    "GTROBBValPreprocessor",
    "is_gtr_obb_state_dict",
    "preprocess_obb_image",
]

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
DOTA_NAMES = dict(enumerate(DOTA_CLASSES))
OBB_INPUT_SIZES = {size: OBB_INPUT_SIZE for size in OBB_SIZES}


def is_gtr_obb_state_dict(sd) -> bool:
    """Five-coordinate query/box heads mark an OBB checkpoint."""
    query_pos = sd.get("decoder.query_pos_head.layers.0.weight")
    pre_bbox = sd.get("decoder.pre_bbox_head.layers.2.weight")
    return (
        query_pos is not None
        and pre_bbox is not None
        and int(query_pos.shape[1]) == 5
        and int(pre_bbox.shape[0]) == 5
    )


def _letterbox_top_left(rgb: np.ndarray, size: tuple[int, int]):
    orig_h, orig_w = rgb.shape[:2]
    target_h, target_w = size
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = max(1, int(round(orig_w * scale)))
    new_h = max(1, int(round(orig_h * scale)))
    resized = np.asarray(
        Image.fromarray(rgb).resize((new_w, new_h), Image.Resampling.BILINEAR),
        dtype=np.uint8,
    )
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    canvas[:new_h, :new_w] = resized
    chw = canvas.astype(np.float32).transpose(2, 0, 1) / 255.0
    return ((chw - IMAGENET_MEAN) / IMAGENET_STD).astype(np.float32), scale


def preprocess_obb_image(image, input_size, color_format="auto"):
    """Return ``(tensor, original_pil, (orig_w, orig_h), scale)``."""
    img = ImageLoader.load(image, color_format=color_format)
    size = (input_size, input_size) if isinstance(input_size, int) else input_size
    chw, scale = _letterbox_top_left(np.asarray(img.convert("RGB")), tuple(size))
    return torch.from_numpy(chw).unsqueeze(0), img, img.size, scale


class GTROBBValPreprocessor(RTDETRv2OBBValPreprocessor):
    """RT-DETRv2 OBB letterbox plus ImageNet normalization."""

    @property
    def normalize(self) -> bool:
        # The OBB validator divides by 255 whenever pixels exceed 1; ImageNet
        # normalized inputs do, so it must leave them alone.
        return False

    @property
    def custom_normalization(self) -> bool:
        return True

    def __call__(self, img, targets, input_size):
        chw, padded_targets = super().__call__(img, targets, input_size)
        chw = ((chw - IMAGENET_MEAN) / IMAGENET_STD).astype(np.float32)
        return chw, padded_targets
