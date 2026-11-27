"""Shared runtime contract for dense edge-detection specialists."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image


def preprocess_numpy(
    image_rgb_hwc: np.ndarray,
    input_size: int | Tuple[int, int],
) -> Tuple[np.ndarray, float]:
    """Stretch RGB input to a square and return canonical float32 CHW ``[0,1]``.

    ``input_size`` is one side or a square ``(h, w)`` pair (export verify
    passes the canvas as a pair).
    """
    image = np.asarray(image_rgb_hwc)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(
            f"edge preprocessing expects RGB HWC input, got shape {image.shape}"
        )
    if isinstance(input_size, (tuple, list)):
        height, width = (int(v) for v in input_size)
        if height != width:
            raise ValueError(f"edge preprocessing needs a square size, got {input_size}")
        input_size = height
    size = int(input_size)
    if size <= 0:
        raise ValueError(f"input_size must be positive, got {input_size}")
    resized = np.array(
        Image.fromarray(image.astype(np.uint8, copy=False)).resize(
            (size, size),
            Image.BILINEAR,
        ),
        copy=True,
    )
    chw = resized.transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.ascontiguousarray(chw), 1.0


class EdgeInferenceNet(nn.Module):
    """Put BGR mean subtraction and fused-edge sigmoid inside the graph.

    Exported models therefore share one external input contract: RGB float32
    in ``[0,1]``. The wrapped specialist core retains its upstream-compatible
    parameter names below the ``core.`` prefix.
    """

    def __init__(self, core: nn.Module):
        super().__init__()
        self.core = core
        self.register_buffer(
            "_mean_bgr",
            torch.tensor([103.939, 116.779, 123.68]).view(1, 3, 1, 1),
            persistent=False,
        )

    def forward(self, image_rgb: torch.Tensor) -> torch.Tensor:
        image_bgr = image_rgb[:, [2, 1, 0]] * 255.0
        logits = self.core(image_bgr - self._mean_bgr)[-1]
        return torch.sigmoid(logits)


def prefix_upstream_state_dict(state_dict: dict) -> dict:
    """Prefix a raw specialist core state dict for :class:`EdgeInferenceNet`."""
    if any(key.startswith("core.") for key in state_dict):
        return dict(state_dict)
    return {f"core.{key}": value for key, value in state_dict.items()}


def unprefixed_keys(state_dict: dict) -> set[str]:
    """Return a key-only view that accepts raw and LibreYOLO layouts."""
    return {
        key.removeprefix("core.") if key.startswith("core.") else key
        for key in state_dict
    }


__all__ = [
    "EdgeInferenceNet",
    "prefix_upstream_state_dict",
    "preprocess_numpy",
    "unprefixed_keys",
]
