# SPDX-License-Identifier: Apache-2.0
# Adapted from the GLM-OCR SDK (https://github.com/zai-org/GLM-OCR, commit
# 0288f96bb9e78cb439883eaa725eeae90bd177c5, Apache-2.0, Copyright (c) Z.ai /
# zai-org): glmocr/utils/image_utils.py (smart_resize, load_image_to_base64,
# crop_image_region) and glmocr/dataloader/page_loader.py (prompt mapping and
# request defaults of build_request_from_image). See NOTICE.
"""Region crops and the image the GLM-OCR recognizer actually sees.

The SDK crops each layout region, resizes it to a multiple of 28 pixels with
bicubic interpolation and encodes it as a JPEG (PIL default quality) before
sending it to the model server. This module reproduces that in memory, so the
recognizer gets the same decoded pixels.
"""

from __future__ import annotations

import io
import math
from typing import Sequence, Tuple

from PIL import Image

# ``pipeline.page_loader`` defaults of ``glmocr/config.yaml``.
TASK_PROMPTS = {
    "text": "Text Recognition:",
    "table": "Table Recognition:",
    "formula": "Formula Recognition:",
}
MIN_PIXELS = 112 * 112
MAX_PIXELS = 71372800
IMAGE_FORMAT = "JPEG"
T_PATCH_SIZE = 2
PATCH_EXPAND_FACTOR = 1


def smart_resize(
    t: int,
    h: int,
    w: int,
    t_factor: int = 1,
    h_factor: int = 28,
    w_factor: int = 28,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = 14 * 14 * 4 * 15000,
) -> Tuple[int, int]:
    """Nearest size divisible by the factors with ``t*h*w`` in budget."""
    if t < t_factor:
        raise ValueError("Temporal dimension must be at least the temporal factor.")
    h_bar = round(h / h_factor) * h_factor
    w_bar = round(w / w_factor) * w_factor
    t_bar = round(t / t_factor) * t_factor
    if t_bar * h_bar * w_bar > max_pixels:
        beta = math.sqrt((t * h * w) / max_pixels)
        h_bar = math.floor(h / beta / h_factor) * h_factor
        w_bar = math.floor(w / beta / w_factor) * w_factor
    elif t_bar * h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (t * h * w))
        h_bar = math.ceil(h * beta / h_factor) * h_factor
        w_bar = math.ceil(w * beta / w_factor) * w_factor
    return h_bar, w_bar


def prepare_image(
    image: Image.Image,
    *,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
    image_format: str | None = IMAGE_FORMAT,
) -> Image.Image:
    """Resize like the SDK and round-trip through ``image_format``.

    ``image_format=None`` skips the encode/decode (lossless, not upstream).
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    w, h = image.size
    factor = 14 * 2 * PATCH_EXPAND_FACTOR
    h_bar, w_bar = smart_resize(
        t=T_PATCH_SIZE,
        h=h,
        w=w,
        t_factor=T_PATCH_SIZE,
        h_factor=factor,
        w_factor=factor,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    image = image.resize((w_bar, h_bar), Image.Resampling.BICUBIC)
    if not image_format:
        return image
    buffer = io.BytesIO()
    image.save(buffer, format=image_format)
    buffer.seek(0)
    decoded = Image.open(buffer)
    decoded.load()
    return decoded.convert("RGB") if decoded.mode != "RGB" else decoded


def crop_region(image: Image.Image, bbox_2d: Sequence[int]) -> Image.Image:
    """Crop a 0-1000 normalized box (the SDK's ``crop_image_region`` without
    a polygon mask, its default)."""
    width, height = image.size
    x1, y1, x2, y2 = bbox_2d
    return image.crop(
        (
            int(x1 * width / 1000),
            int(y1 * height / 1000),
            int(x2 * width / 1000),
            int(y2 * height / 1000),
        )
    )
