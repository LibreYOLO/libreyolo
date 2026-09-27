"""Postprocessing for HVI-CIDNet low-light restoration."""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np
import torch


def postprocess(output: Any, original_size: Tuple[int, int]) -> np.ndarray:
    """Crop the padded prediction and return original-canvas uint8 RGB."""

    if isinstance(output, dict):
        output = output.get("restored", output.get("output", output.get("predictions")))
    if isinstance(output, (list, tuple)):
        output = output[0]
    restored = output if isinstance(output, torch.Tensor) else torch.as_tensor(output)
    if restored.ndim == 3:
        restored = restored.unsqueeze(0)
    if restored.ndim != 4 or restored.shape[1] != 3:
        raise ValueError(
            f"HVI-CIDNet postprocess expects [B, 3, H, W], got {tuple(restored.shape)}."
        )
    original_w, original_h = original_size
    restored = restored[0, :, :original_h, :original_w].detach().float().cpu()
    restored = restored.clamp(0, 1).permute(1, 2, 0).mul(255).round().byte()
    return restored.numpy()


__all__ = ["postprocess"]
