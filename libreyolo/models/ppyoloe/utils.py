"""PP-YOLOE preprocessing and checkpoint helpers.

Preprocessing reproduces the source validation path for the released COCO
weights: RGB, **direct resize** (stretch) to the model input size, float
conversion, then channel normalization with mean ``[123.675, 116.28, 103.53]``
and std ``[58.395, 57.12, 57.375]`` on the 0-255 scale. This is not letterbox,
so the x and y scale factors differ and are reversed independently in
``libreyolo.postprocess.ppyoloe``.

Postprocessing lives in ``libreyolo.postprocess.ppyoloe`` (ADR 0005) and is
re-exported here for symmetry with the other families.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from ...postprocess.ppyoloe import postprocess  # noqa: F401  (re-export)
from ...utils.image_loader import ImageInput, ImageLoader

__all__ = [
    "PPYOLOE_MEAN",
    "PPYOLOE_STD",
    "preprocess_numpy",
    "preprocess_image",
    "postprocess",
]


PPYOLOE_MEAN = (123.675, 116.28, 103.53)
PPYOLOE_STD = (58.395, 57.12, 57.375)


def preprocess_numpy(
    img_rgb_hwc: np.ndarray,
    input_size: int | Tuple[int, int] = 640,
) -> Tuple[np.ndarray, float]:
    """Preprocess an RGB HWC uint8 image for PP-YOLOE inference.

    Returns ``(chw_float32, ratio)``. ``ratio`` is always 1.0: the stretch
    resize has no single scalar scale, and postprocessing recovers the two
    axis scales from the original and input sizes instead. It stays in the
    signature so PP-YOLOE flows through the same call sites as the
    letterbox families.
    """
    if isinstance(input_size, (list, tuple)):
        input_h, input_w = int(input_size[0]), int(input_size[1])
    else:
        input_h = input_w = int(input_size)

    arr = cv2.resize(
        img_rgb_hwc, (input_w, input_h), interpolation=cv2.INTER_LINEAR
    ).astype(np.float32)
    arr -= np.array(PPYOLOE_MEAN, dtype=np.float32)
    arr /= np.array(PPYOLOE_STD, dtype=np.float32)
    return arr.transpose(2, 0, 1), 1.0


def preprocess_image(
    image: ImageInput,
    input_size: int | Tuple[int, int] = 640,
    color_format: str = "auto",
) -> Tuple[torch.Tensor, Image.Image, Tuple[int, int], float]:
    img = ImageLoader.load(image, color_format=color_format)
    original_size = img.size
    original_img = img.copy()
    chw, ratio = preprocess_numpy(np.array(img), input_size)
    return torch.from_numpy(chw).unsqueeze(0), original_img, original_size, ratio
