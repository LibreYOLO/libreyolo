"""Preprocessing for guided ViTMatte inference.

The contract follows the Apache-2.0 Transformers image processor carrying
Copyright 2023 The HuggingFace Inc. team; see the family ``NOTICE``.
"""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ...utils.image_loader import ImageInput, ImageLoader


PAD_MULTIPLE = 32


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    elif isinstance(value, np.ndarray):
        value = np.asarray(value)
    else:
        # The shared loader covers paths, URLs, PIL images and encoded bytes.
        value = np.asarray(ImageLoader.load(value, color_format="rgb"))
    return np.asarray(value)


def _single_channel_trimap(value: Any) -> np.ndarray:
    array = _to_numpy(value)
    if array.ndim == 4:
        if array.shape[0] != 1:
            raise ValueError("trimap= must contain one guide image, not a batch.")
        array = array[0]
    if (
        array.ndim == 3
        and array.shape[0] in (1, 3, 4)
        and array.shape[-1] not in (1, 3, 4)
    ):
        array = array.transpose(1, 2, 0)
    if array.ndim == 3:
        if array.shape[-1] == 1:
            array = array[..., 0]
        elif array.shape[-1] in (3, 4):
            channels = array[..., :3].astype(np.float32, copy=False)
            if not (
                np.array_equal(channels[..., 0], channels[..., 1])
                and np.array_equal(channels[..., 0], channels[..., 2])
            ):
                raise ValueError("trimap= must be grayscale; RGB channels differ.")
            array = array[..., 0]
        else:
            raise ValueError(
                "trimap= must be a 2D guide or a one/three-channel image; "
                f"got shape {tuple(array.shape)}."
            )
    if array.ndim != 2:
        raise ValueError(
            f"trimap= must be a 2D guide image; got shape {tuple(array.shape)}."
        )
    return np.asarray(array, dtype=np.float32)


def normalize_trimap(value: Any) -> torch.Tensor:
    """Validate a three-level trimap and return ``(1, H, W)`` in ``[0, 1]``.

    Accepted encodings are exactly ``0/128/255`` and ``0/0.5/1`` (within a
    small floating-point tolerance). Arbitrary soft masks are rejected because
    ViTMatte's fourth channel is a categorical known/unknown guide.
    """
    array = _single_channel_trimap(value)
    if not np.isfinite(array).all():
        raise ValueError("trimap= contains NaN or infinite values.")

    normalized_encoding = (
        float(array.min()) >= -1e-6 and float(array.max()) <= 1.0 + 1e-6
    )
    allowed = np.asarray(
        (0.0, 0.5, 1.0) if normalized_encoding else (0.0, 128.0, 255.0),
        dtype=np.float32,
    )
    distances = np.abs(array[..., None] - allowed)
    nearest = distances.argmin(axis=-1)
    if bool((distances.min(axis=-1) > 1e-6).any()):
        invalid = np.unique(array[distances.min(axis=-1) > 1e-6])[:8]
        values = ", ".join(f"{float(item):g}" for item in invalid)
        raise ValueError(
            "trimap= accepts only values {0, 128, 255} or {0, 0.5, 1}; "
            f"found {values}."
        )

    snapped = allowed[nearest]
    if not normalized_encoding:
        snapped = snapped / 255.0
    return torch.from_numpy(snapped).unsqueeze(0).contiguous()


def _pad_to_multiple(
    tensor: torch.Tensor,
    multiple: int = PAD_MULTIPLE,
) -> torch.Tensor:
    height, width = int(tensor.shape[-2]), int(tensor.shape[-1])
    pad_height = (multiple - height % multiple) % multiple
    pad_width = (multiple - width % multiple) % multiple
    if not pad_height and not pad_width:
        return tensor
    # Zero is the reference processor's padding value: mean RGB after
    # normalization, and known background in the trimap channel.
    return F.pad(tensor, (0, pad_width, 0, pad_height), value=0.0)


def preprocess_guided_image(
    image: ImageInput,
    trimap: ImageInput,
    *,
    color_format: str = "auto",
) -> Tuple[torch.Tensor, Image.Image, Tuple[int, int], float]:
    """Build a padded four-channel ``RGB[-1,1] + trimap[0,1]`` tensor."""
    rgb_image = ImageLoader.load(image, color_format=color_format).convert("RGB")
    original_size = rgb_image.size
    rgb_array = np.asarray(rgb_image, dtype=np.float32)
    # This is algebraically equivalent to rescale(1/255), normalize(0.5,
    # 0.5), and bit-exact with the audited Transformers processor's fused
    # float32 path. Keeping the fused expression avoids a one-ULP rounding
    # difference around the midpoint.
    rgb_array = (rgb_array - 127.5) / 127.5
    rgb = torch.from_numpy(rgb_array).permute(2, 0, 1).contiguous()

    guide = normalize_trimap(trimap)
    height, width = rgb.shape[-2:]
    if tuple(guide.shape[-2:]) != (height, width):
        guide = F.interpolate(
            guide.unsqueeze(0),
            size=(height, width),
            mode="nearest",
        )[0]

    pixel_values = torch.cat([rgb, guide], dim=0).unsqueeze(0)
    return _pad_to_multiple(pixel_values), rgb_image, original_size, 1.0


def preprocess_numpy(
    image_rgba_hwc: np.ndarray,
    input_size: int | tuple[int, int] = 512,
) -> tuple[np.ndarray, float]:
    """Prepare an already combined RGB+trimap array for graph tooling.

    Export is intentionally not advertised yet, but BaseModel requires this
    family hook. A four-channel input is mandatory so graph tooling can never
    invent or silently omit the trimap.
    """
    del input_size
    array = np.asarray(image_rgba_hwc)
    if array.ndim != 3 or array.shape[-1] != 4:
        raise ValueError(
            "ViTMatte preprocessing requires an HxWx4 RGB+trimap array; "
            f"got {tuple(array.shape)}."
        )
    rgb = np.asarray(array[..., :3], dtype=np.float32)
    if float(rgb.max()) > 1.0:
        rgb = (rgb - 127.5) / 127.5
    else:
        rgb = (rgb - 0.5) / 0.5
    guide = normalize_trimap(array[..., 3]).numpy().transpose(1, 2, 0)
    combined = np.concatenate([rgb, guide], axis=-1)
    tensor = torch.from_numpy(combined.transpose(2, 0, 1)).unsqueeze(0).float()
    return _pad_to_multiple(tensor)[0].numpy(), 1.0


__all__ = [
    "PAD_MULTIPLE",
    "normalize_trimap",
    "preprocess_guided_image",
    "preprocess_numpy",
]
