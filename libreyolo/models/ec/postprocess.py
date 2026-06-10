"""LibreEC preprocessing and postprocessing helpers.

EC's input pipeline is square resize (no letterbox), divide by 255, then
ImageNet (mean, std) normalization. The ImageNet normalization is what
distinguishes EC's preprocess from D-FINE's. Output postprocessing is
DETR-style top-K (no NMS); it lives in ``libreyolo.postprocess.ec`` and is
re-exported here for backward compatibility.
"""

from __future__ import annotations

from typing import Any, Mapping, Tuple

import numpy as np
import torch
from PIL import Image

from ...postprocess.ec import (  # noqa: F401  (backward-compatible re-exports)
    postprocess,
    postprocess_pose,
    postprocess_seg,
)
from ...utils.image_loader import ImageInput, ImageLoader


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
