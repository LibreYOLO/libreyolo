# Copyright (c) 2025 Technology Innovation Institute (TII), UAE.
# SPDX-License-Identifier: Apache-2.0
#
# Ported from https://github.com/tiiuae/Falcon-Perception (Apache-2.0),
# commit c457916c9974efbacfa91f0f6ecc2c49c6543e56:
#   falcon_perception/data.py, falcon_perception/__init__.py (tokenizer)
# See libreyolo/models/falcon_perception/NOTICE.
"""Image, prompt and position preprocessing for Falcon Perception.

Pure numpy / PIL. The numerics (bicubic smart-resize, 0.5/0.5 normalisation,
square canvas bucketing, 2D RoPE positions) follow the upstream batch engine
exactly so token streams and coordinates match it.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image

IMAGE_MEAN = [0.5, 0.5, 0.5]
IMAGE_STD = [0.5, 0.5, 0.5]

ANYUP_SIZE_STEP = 128
ANYUP_SIZE_MIN = 128
ANYUP_SIZE_MAX = 1024

PATCH_SIZE = 16
MIN_DIMENSION = 256
MAX_DIMENSION = 1024
HR_UPSAMPLE_RATIO = 8

QUERY_PREFIX = "Segment these expressions in the image:"


def build_prompt(query: str) -> str:
    """Perception prompt for one query (identical for detection and segmentation)."""
    return f"<|image|>{QUERY_PREFIX}<|start_of_query|>{query}<|REF_SEG|>"


# =============================================================================
# Tokenizer
# =============================================================================


class FalconTokenizer:
    """Thin wrapper over the ``tokenizers`` library exposing the special-token
    attributes the model needs (``image_token_id``, ``seg_token_id``, ...)."""

    def __init__(self, path: str):
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:  # pragma: no cover - ships with the vlm extra
            raise ImportError(
                "Falcon Perception needs the 'vlm' extra. Install with:\n"
                "    pip install 'libreyolo[vlm]'"
            ) from exc

        self._tok = Tokenizer.from_file(os.path.join(path, "tokenizer.json"))

        config_file = os.path.join(path, "tokenizer_config.json")
        stm_file = os.path.join(path, "special_tokens_map.json")
        config: dict = {}
        if os.path.isfile(config_file):
            config = json.loads(Path(config_file).read_text(encoding="utf-8"))
        stm: dict = {}
        if os.path.isfile(stm_file):
            stm = json.loads(Path(stm_file).read_text(encoding="utf-8"))

        self.special_tokens_map: dict[str, str] = {}
        for name, val in stm.items():
            if isinstance(val, str):
                self.special_tokens_map[name] = val
        for name, val in config.get("model_specific_special_tokens", {}).items():
            if isinstance(val, str):
                self.special_tokens_map[name] = val

        for name, token_str in self.special_tokens_map.items():
            setattr(self, name, token_str)
            setattr(self, name + "_id", self._tok.token_to_id(token_str))

        self.eos_token_id: int = self._tok.token_to_id(
            config.get("eos_token", stm.get("eos_token", "<|end_of_text|>"))
        )
        self.bos_token_id = None
        bos_str = config.get("bos_token") or stm.get("bos_token")
        if bos_str:
            self.bos_token_id = self._tok.token_to_id(bos_str)
        self.bos_id = self.bos_token_id
        self.pad_token_id: int = self._tok.token_to_id(
            config.get("pad_token", stm.get("pad_token", "<|pad|>"))
        )

    def encode(self, text: str) -> List[int]:
        return self._tok.encode(text).ids


# =============================================================================
# Image preprocessing
# =============================================================================


def snap_image_size(
    size: int,
    *,
    step: int = ANYUP_SIZE_STEP,
    min_size: int = ANYUP_SIZE_MIN,
    max_size: int = ANYUP_SIZE_MAX,
) -> int:
    size = min(max(int(size), min_size), max_size)
    snapped = ((size + step - 1) // step) * step
    return min(max(snapped, min_size), max_size)


def anyup_canvas_size(
    height: int,
    width: int,
    *,
    patch_size: int = PATCH_SIZE,
    upsample_ratio: int = HR_UPSAMPLE_RATIO,
    step: int = ANYUP_SIZE_STEP,
    min_size: int = ANYUP_SIZE_MIN,
    max_size: int = ANYUP_SIZE_MAX,
) -> Tuple[int, Tuple[int, int]]:
    """Square canvas and matching high-res output, both on the ``step`` grid."""
    need = max(height, width)
    need = ((need + patch_size - 1) // patch_size) * patch_size
    need = min(need, max_size)

    native_out = need // patch_size * upsample_ratio
    out_max = max_size // patch_size * upsample_ratio
    out = snap_image_size(
        native_out, step=step, min_size=min(min_size, out_max), max_size=out_max
    )
    canvas = out * patch_size // upsample_ratio
    return canvas, (out, out)


def resize_image_if_necessary(
    image: Image.Image,
    min_dimension: int = MIN_DIMENSION,
    max_dimension: int = MAX_DIMENSION,
) -> Image.Image:
    """Resize so both sides lie in [min_dimension, max_dimension], keeping aspect."""
    assert min_dimension <= max_dimension
    original_width, original_height = image.size
    aspect_ratio = original_width / original_height

    if (
        min_dimension <= original_width <= max_dimension
        and min_dimension <= original_height <= max_dimension
    ):
        return image

    is_vertical_image = original_width < original_height
    if original_width < min_dimension or original_height < min_dimension:
        if is_vertical_image:
            new_width = min_dimension
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = min_dimension
            new_width = int(new_height * aspect_ratio)
    else:
        if is_vertical_image:
            new_width = max_dimension
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = max_dimension
            new_width = int(new_height * aspect_ratio)

    if new_width > max_dimension:
        new_width = max_dimension
        new_height = int(new_width / aspect_ratio)
    if new_height > max_dimension:
        new_height = max_dimension
        new_width = int(new_height * aspect_ratio)

    return image.resize((new_width, new_height))


def smart_resize(
    image: np.ndarray,
    factor: int,
    min_pixels: int = 56 * 56,
    max_pixels: int = 28 * 28 * 1280 * 10,
) -> np.ndarray:
    """Round both sides to a multiple of ``factor`` (bicubic) within a pixel budget."""
    height, width = image.shape[0], image.shape[1]
    if height < factor or width < factor:
        raise ValueError(f"{height=} or {width=} must be larger than {factor=}")
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
    pil_img = Image.fromarray(image.astype(np.uint8) if image.dtype != np.uint8 else image)
    pil_img = pil_img.resize((w_bar, h_bar), Image.Resampling.BICUBIC)
    return np.array(pil_img)


def preprocess_image(image: Image.Image, patch_size: int = PATCH_SIZE) -> np.ndarray:
    """PIL RGB image -> ``(1, H, W, 3)`` float32 normalised to [-1, 1]."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    arr = np.array(image)
    arr = smart_resize(arr, factor=patch_size)
    arr = arr.astype(np.float32) * (1 / 255)
    mean = np.array(IMAGE_MEAN, dtype=np.float32)[None, None, :]
    std = np.array(IMAGE_STD, dtype=np.float32)[None, None, :]
    arr = (arr.astype(np.float32) - mean) / std
    return arr[None, ...]


def batch_images_with_mask(
    pixel_values: List[np.ndarray], max_h: int, max_w: int
) -> dict:
    """Pad images onto a ``(max_h, max_w)`` canvas with a boolean validity mask."""
    N = len(pixel_values)
    max_temporal = max(img.shape[0] for img in pixel_values)
    C = pixel_values[0].shape[-1]
    batched = np.zeros((N, max_temporal, max_h, max_w, C), dtype=np.float32)
    masks = np.zeros((N, max_temporal, max_h, max_w), dtype=bool)
    for i, img in enumerate(pixel_values):
        t, h, w = img.shape[0], img.shape[1], img.shape[2]
        batched[i, :t, :h, :w, :] = img
        masks[i, :t, :h, :w] = True
    return {"pixel_values": batched, "padding_mask": masks}


# =============================================================================
# RoPE positions
# =============================================================================


def _compute_image_spatial_positions(
    pixel_mask_THW: np.ndarray, spatial_patch_size: int, temporal_patch_size: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    T, H, W = pixel_mask_THW.shape
    tp, hp, wp = temporal_patch_size, spatial_patch_size, spatial_patch_size
    # "any" over each (tp, hp, wp) block: (T, H, W) -> (T/tp, H/hp, W/wp)
    mask_thw = pixel_mask_THW.reshape(T // tp, tp, H // hp, hp, W // wp, wp).any(
        axis=(1, 3, 5)
    )
    width = int(mask_thw.sum(axis=-1).astype(int).max())
    height = int(mask_thw.sum(axis=-2).astype(int).max())

    xlim = np.sqrt(width / height)
    ylim = np.sqrt(height / width)
    xpos = np.linspace(-xlim, xlim, width)
    ypos = np.linspace(-ylim, ylim, height)
    wpos, hpos = np.meshgrid(xpos, ypos, indexing="xy")
    return hpos.flatten(), wpos.flatten()


def _get_image_token_masks(tokens: np.ndarray, tokenizer: FalconTokenizer):
    spatial_mask = tokens == tokenizer.image_token_id
    no_increase_mask = (
        spatial_mask
        | (tokens == tokenizer.image_reg_1_token_id)
        | (tokens == tokenizer.image_reg_2_token_id)
        | (tokens == tokenizer.image_reg_3_token_id)
        | (tokens == tokenizer.image_reg_4_token_id)
        | (tokens == tokenizer.end_of_image_token_id)
    )
    return spatial_mask, no_increase_mask


def get_pos_thw(
    tokens: np.ndarray,
    pixel_masks_NTHW: np.ndarray,
    tokenizer: FalconTokenizer,
    spatial_patch_size: int,
    temporal_patch_size: int = 1,
    *,
    pad_token_id: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Temporal and (h, w) RoPE positions for a padded token batch."""
    spatial_img_token_mask_BS, no_increase_mask_BS = _get_image_token_masks(tokens, tokenizer)

    hpos_parts, wpos_parts = [], []
    for i in range(pixel_masks_NTHW.shape[0]):
        h, w = _compute_image_spatial_positions(
            pixel_masks_NTHW[i], spatial_patch_size, temporal_patch_size
        )
        hpos_parts.append(h)
        wpos_parts.append(w)

    hpos_N = np.concatenate(hpos_parts) if hpos_parts else np.empty(0)
    wpos_N = np.concatenate(wpos_parts) if wpos_parts else np.empty(0)

    expected_tokens = spatial_img_token_mask_BS.sum()
    if hpos_N.size != expected_tokens:
        raise ValueError(
            f"Mismatch between spatial image tokens ({expected_tokens}) and generated "
            f"positions ({hpos_N.size})."
        )

    hpos_BS = np.zeros_like(tokens, dtype=np.float32)
    wpos_BS = np.zeros_like(tokens, dtype=np.float32)
    hpos_BS[spatial_img_token_mask_BS] = hpos_N
    wpos_BS[spatial_img_token_mask_BS] = wpos_N

    tpos_BS = np.ones_like(tokens, dtype=np.float32)
    tpos_BS[no_increase_mask_BS] = 0
    tpos_BS = np.cumsum(tpos_BS, axis=1) - 1
    tpos_BS[tokens == pad_token_id] = 0

    hw_pos_BS2 = np.stack([hpos_BS, wpos_BS], axis=-1)
    return tpos_BS.astype(np.int64), hw_pos_BS2


# =============================================================================
# Prompt tokenization
# =============================================================================


def calculate_image_tokens(image: np.ndarray, patch_size: int, merge_size: int = 1) -> int:
    height, width = image.shape[0], image.shape[1]
    return int((height * width) / (patch_size * patch_size * merge_size * merge_size))


def tokenize_prompt(
    prompt: str,
    image: np.ndarray,
    tokenizer: FalconTokenizer,
    patch_size: int,
    max_length: int,
) -> np.ndarray:
    """Tokenize ``prompt``, expanding ``<|image|>`` into the image token block."""
    img_reg_ids = [
        tokenizer.image_reg_1_token_id,
        tokenizer.image_reg_2_token_id,
        tokenizer.image_reg_3_token_id,
        tokenizer.image_reg_4_token_id,
    ]
    count = calculate_image_tokens(image, patch_size, 1)
    prompt_chunks = [tokenizer.encode(chunk) for chunk in prompt.split(tokenizer.image_token)]

    offset = 0
    input_ids: list[int] = []
    if (
        len(prompt_chunks) > 0
        and len(prompt_chunks[0]) > 0
        and (tokenizer.bos_id is not None and prompt_chunks[0][0] == tokenizer.bos_id)
    ):
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    image_block = [
        tokenizer.image_cls_token_id,
        *img_reg_ids,
        *([tokenizer.image_token_id] * count),
        tokenizer.end_of_image_token_id,
    ]
    separators = [image_block]
    if len(separators) != len(prompt_chunks):
        separators.append(separators[-1])

    interleaved = [ele for pair in zip(prompt_chunks, separators) for ele in pair][:-1]
    for index, x in enumerate(interleaved):
        if index % 2 != 0:
            if len(input_ids) + len(x) >= max_length:
                raise ValueError(
                    f"The image needs {len(x)} tokens but only "
                    f"{max_length - len(input_ids) - 1} fit in the prompt budget "
                    f"({max_length})."
                )
            input_ids.extend(x)
        else:
            input_ids.extend(x[offset:])
    return np.array(input_ids, dtype=np.int64)
