"""Image and mask preprocessing for LibreLaMa."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from ...utils.image_loader import ImageInput, ImageLoader
from .nn import ONNX_INPUT_SIZE


@dataclass(frozen=True)
class LaMaPredictionContext:
    """Original-canvas values required for exact final compositing."""

    original_rgb: np.ndarray
    fill_mask: np.ndarray


def _require_fixed_input_size(input_size: int | tuple[int, int]) -> None:
    if isinstance(input_size, (tuple, list)):
        valid = tuple(int(value) for value in input_size) == (
            ONNX_INPUT_SIZE,
            ONNX_INPUT_SIZE,
        )
    else:
        valid = int(input_size) == ONNX_INPUT_SIZE
    if not valid:
        raise ValueError(
            "LibreLaMa uses the official fixed 512x512 ONNX graph; "
            f"imgsz={input_size!r} is not supported."
        )


def _image_blob(rgb: np.ndarray) -> np.ndarray:
    bgr = np.ascontiguousarray(rgb[..., ::-1])
    return cv2.dnn.blobFromImage(
        bgr,
        scalefactor=1.0 / 255.0,
        size=(ONNX_INPUT_SIZE, ONNX_INPUT_SIZE),
        mean=(0.0, 0.0, 0.0),
        swapRB=False,
        crop=False,
        ddepth=cv2.CV_32F,
    )


def preprocess_numpy(
    img_rgb_hwc: np.ndarray,
    input_size: int | tuple[int, int] = ONNX_INPUT_SIZE,
) -> Tuple[np.ndarray, float]:
    """Convert RGB HWC to the graph's fixed BGR CHW float input."""

    _require_fixed_input_size(input_size)
    image = ImageLoader.load(np.asarray(img_rgb_hwc), color_format="rgb")
    rgb = np.asarray(image, dtype=np.uint8)
    return _image_blob(rgb)[0], 1.0


def preprocess_image_and_mask(
    image: ImageInput,
    mask: Any,
    *,
    color_format: str = "auto",
    input_size: int | tuple[int, int] = ONNX_INPUT_SIZE,
) -> tuple[
    torch.Tensor,
    Image.Image,
    tuple[int, int],
    float,
    LaMaPredictionContext,
]:
    """Load aligned inputs and build fixed image/mask blobs.

    Every nonzero mask pixel means "fill". The original binary mask and RGB
    pixels are retained so postprocessing can restore all unmasked pixels
    exactly, byte for byte, after the fixed 512 inference canvas is resized.
    """

    _require_fixed_input_size(input_size)
    image_rgb = ImageLoader.load(image, color_format=color_format)
    mask_rgb = ImageLoader.load(mask, color_format="auto")
    if mask_rgb.size != image_rgb.size:
        raise ValueError(
            "LibreLaMa mask must use the same original canvas as the image; "
            f"image size={image_rgb.size}, mask size={mask_rgb.size}."
        )

    original_rgb = np.ascontiguousarray(np.asarray(image_rgb, dtype=np.uint8))
    mask_array = np.asarray(mask_rgb, dtype=np.uint8)
    fill_mask = np.ascontiguousarray(np.any(mask_array != 0, axis=2))

    image_blob = _image_blob(original_rgb)
    mask_512 = cv2.resize(
        fill_mask.astype(np.uint8),
        (ONNX_INPUT_SIZE, ONNX_INPUT_SIZE),
        interpolation=cv2.INTER_NEAREST,
    ).astype(np.float32)
    mask_blob = mask_512[None, None]
    guided = np.concatenate((image_blob, mask_blob), axis=1)
    context = LaMaPredictionContext(original_rgb=original_rgb, fill_mask=fill_mask)
    return (
        torch.from_numpy(np.ascontiguousarray(guided)),
        image_rgb,
        image_rgb.size,
        1.0,
        context,
    )


__all__ = [
    "LaMaPredictionContext",
    "preprocess_image_and_mask",
    "preprocess_numpy",
]
