"""LibreTinyFormer pre/postprocessing helpers.

TinyFormer shares DEIMv2's inference contract: plain square resize with
ImageNet normalisation (every size runs a DINO-lineage tower) and the DEIM
DETR postprocess (top-K over DFL-decoded boxes, no NMS).
"""

from __future__ import annotations

from functools import partial
from typing import Tuple

import numpy as np
import torch
from PIL import Image

from ...postprocess.deim import postprocess
from ...preprocess.deimv2 import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    preprocess_numpy as _deimv2_preprocess_numpy,
)
from ...utils.image_loader import ImageInput, ImageLoader
from ..deim.utils import unwrap_deim_checkpoint

__all__ = [
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "postprocess",
    "preprocess_image",
    "preprocess_numpy",
    "unwrap_deim_checkpoint",
]

# Every TinyFormer size expects ImageNet-normalised input.
preprocess_numpy = partial(_deimv2_preprocess_numpy, imagenet_norm=True)


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
