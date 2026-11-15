# SPDX-License-Identifier: Apache-2.0
# Adapted from Falcon OCR (Technology Innovation Institute, Apache-2.0):
# processing_falcon_ocr.py of the Hugging Face release tiiuae/Falcon-OCR @
# 0bc28f89c90fbe8bdb063097fab144d4ce4c19b5, the same code TII publishes in
# https://github.com/tiiuae/Falcon-Perception (falcon_perception/data.py, commit
# c457916c9974efbacfa91f0f6ecc2c49c6543e56, Copyright (c) 2025 TII). See NOTICE.
"""Falcon OCR input pipeline: image resize, patch layout, token ids, positions.

The release resizes the page so both sides fall in ``[64, 1536]``, snaps each
side to a multiple of the 16-pixel patch, scales pixels to ``[-1, 1]`` and
writes one ``<|image|>`` token per patch, wrapped as
``<|image_cls|> <|image_reg_1..4|> <|image|>*N <|end_of_image|>``. Text tokens
carry a 1D position; patch tokens share the image's position and carry a 2D
``(h, w)`` coordinate on a grid whose extent keeps the aspect ratio.

Batches of one: the release left-pads every image to 1536x1536 and selects the
valid patches with a mask, which is the same as patchifying the unpadded image
row by row, so that is what this module does.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image

PATCH_SIZE = 16
MIN_DIMENSION = 64
MAX_DIMENSION = 1536
MIN_PIXELS = 56 * 56
MAX_PIXELS = 28 * 28 * 1280 * 10


@dataclass(frozen=True)
class FalconOCRTokens:
    """Special token ids of the release (``config.json`` / ``model_args.json``)."""

    img_id: int = 227
    image_cls_token_id: int = 244
    image_reg_token_ids: Tuple[int, int, int, int] = (245, 246, 247, 248)
    img_end_id: int = 230
    eos_id: int = 11

    @classmethod
    def from_config(cls, config: dict) -> "FalconOCRTokens":
        return cls(
            img_id=int(config["img_id"]),
            image_cls_token_id=int(config["image_cls_token_id"]),
            image_reg_token_ids=tuple(
                int(config[f"image_reg_{i}_token_id"]) for i in range(1, 5)
            ),
            img_end_id=int(config["img_end_id"]),
            eos_id=int(config["eos_id"]),
        )


def resize_image_if_necessary(
    image: Image.Image,
    shortest_dimension: int = MIN_DIMENSION,
    longest_dimension: int = MAX_DIMENSION,
) -> Image.Image:
    """Bring both sides into ``[shortest, longest]`` keeping the aspect ratio."""
    original_width, original_height = image.size
    aspect_ratio = original_width / original_height

    if (
        shortest_dimension <= original_width <= longest_dimension
        and shortest_dimension <= original_height <= longest_dimension
    ):
        return image

    is_vertical_image = original_width < original_height
    if original_width < shortest_dimension or original_height < shortest_dimension:
        if is_vertical_image:
            new_width = shortest_dimension
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = shortest_dimension
            new_width = int(new_height * aspect_ratio)
    else:
        if is_vertical_image:
            new_width = longest_dimension
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = longest_dimension
            new_width = int(new_height * aspect_ratio)

    if new_width > longest_dimension:
        new_width = longest_dimension
        new_height = int(new_width / aspect_ratio)
    if new_height > longest_dimension:
        new_height = longest_dimension
        new_width = int(new_height * aspect_ratio)

    # Pillow's default filter, as the release calls it.
    return image.resize((new_width, new_height))


def smart_resize_shape(
    height: int,
    width: int,
    factor: int = PATCH_SIZE,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
) -> Tuple[int, int]:
    """Target ``(height, width)``: multiples of ``factor`` within the pixel budget."""
    if height < factor or width < factor:
        raise ValueError(f"height={height} or width={width} must be at least {factor}")
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            "absolute aspect ratio must be smaller than 200, got "
            f"{max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = np.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = np.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return int(h_bar), int(w_bar)


def image_to_pixels(
    image: Image.Image,
    min_dimension: int = MIN_DIMENSION,
    max_dimension: int = MAX_DIMENSION,
) -> np.ndarray:
    """PIL page to the ``(H, W, 3)`` float32 array in ``[-1, 1]`` the model reads."""
    image = resize_image_if_necessary(image, min_dimension, max_dimension)
    if image.mode != "RGB":
        image = image.convert("RGB")
    width, height = image.size
    h_bar, w_bar = smart_resize_shape(height, width)
    image = image.resize((w_bar, h_bar), resample=Image.Resampling.BICUBIC)
    pixels = np.asarray(image, dtype=np.uint8)
    # Rescale in float64 then store float32, normalize with mean = std = 0.5.
    pixels = (pixels.astype(np.float64) * (1 / 255)).astype(np.float32)
    mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    std = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    return (pixels - mean) / std


def patchify(pixels: np.ndarray, patch_size: int = PATCH_SIZE) -> torch.Tensor:
    """``(H, W, C)`` to ``(H/p * W/p, p*p*C)`` patches, row-major, ``(ph pw c)`` inner."""
    height, width, channels = pixels.shape
    gh, gw = height // patch_size, width // patch_size
    x = torch.from_numpy(np.ascontiguousarray(pixels))
    x = x.reshape(gh, patch_size, gw, patch_size, channels)
    x = x.permute(0, 2, 1, 3, 4).reshape(gh * gw, patch_size * patch_size * channels)
    return x


def image_positions(grid_h: int, grid_w: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """2D patch coordinates: ``linspace(-sqrt(w/h), sqrt(w/h))`` across, likewise down."""
    width = torch.tensor(grid_w, dtype=torch.int32)
    height = torch.tensor(grid_h, dtype=torch.int32)
    xlim = torch.sqrt(width / height)
    ylim = torch.sqrt(height / width)
    xpos = torch.linspace(-xlim, xlim, int(width))
    ypos = torch.linspace(-ylim, ylim, int(height))
    wpos, hpos = torch.meshgrid(xpos, ypos, indexing="xy")
    return hpos.flatten(), wpos.flatten()


def build_input_ids(
    prompt_chunks: List[List[int]],
    num_patches: int,
    tokens: FalconOCRTokens,
) -> List[int]:
    """Text chunks around one image block (the prompt holds one ``<|image|>``)."""
    if len(prompt_chunks) != 2:
        raise ValueError(
            "Falcon OCR prompts must contain exactly one <|image|> placeholder, "
            f"got {len(prompt_chunks) - 1}."
        )
    block = [
        tokens.image_cls_token_id,
        *tokens.image_reg_token_ids,
        *([tokens.img_id] * num_patches),
        tokens.img_end_id,
    ]
    return list(prompt_chunks[0]) + block + list(prompt_chunks[1])


def positions(
    ids: torch.Tensor,
    grid_h: int,
    grid_w: int,
    tokens: FalconOCRTokens,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """1D positions ``(S,)`` and 2D patch positions ``(S, 2)`` (NaN off-image)."""
    spatial = ids == tokens.img_id
    no_increase = spatial | (ids == tokens.img_end_id)
    for reg in tokens.image_reg_token_ids:
        no_increase |= ids == reg
    if int(spatial.sum()) != grid_h * grid_w:
        raise ValueError("image token count does not match the patch grid")
    hpos, wpos = image_positions(grid_h, grid_w)
    h_s = torch.full(ids.shape, float("nan")).masked_scatter_(spatial, hpos)
    w_s = torch.full(ids.shape, float("nan")).masked_scatter_(spatial, wpos)
    t = torch.ones(ids.shape, dtype=torch.float)
    t[no_increase] = 0
    t = torch.cumsum(t, dim=0) - 1
    return t.long(), torch.stack([h_s, w_s], dim=-1)


@dataclass
class FalconOCRInputs:
    """One prepared request: token ids, positions and image patches."""

    ids: torch.Tensor  # (S,) long
    pos_t: torch.Tensor  # (S,) long
    pos_hw: torch.Tensor  # (S, 2) float, NaN off the image patches
    patches: torch.Tensor  # (N, 768) float32
    grid: Tuple[int, int]


def prepare_inputs(
    image: Image.Image,
    prompt_chunks: List[List[int]],
    tokens: FalconOCRTokens,
    min_dimension: int = MIN_DIMENSION,
    max_dimension: int = MAX_DIMENSION,
) -> FalconOCRInputs:
    pixels = image_to_pixels(image, min_dimension, max_dimension)
    grid_h, grid_w = pixels.shape[0] // PATCH_SIZE, pixels.shape[1] // PATCH_SIZE
    ids = torch.tensor(
        build_input_ids(prompt_chunks, grid_h * grid_w, tokens), dtype=torch.long
    )
    pos_t, pos_hw = positions(ids, grid_h, grid_w, tokens)
    return FalconOCRInputs(ids, pos_t, pos_hw, patchify(pixels), (grid_h, grid_w))


__all__ = [
    "FalconOCRInputs",
    "FalconOCRTokens",
    "MAX_DIMENSION",
    "MIN_DIMENSION",
    "build_input_ids",
    "image_positions",
    "image_to_pixels",
    "patchify",
    "positions",
    "prepare_inputs",
    "resize_image_if_necessary",
    "smart_resize_shape",
]
