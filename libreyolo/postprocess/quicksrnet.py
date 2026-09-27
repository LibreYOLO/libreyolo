"""Postprocessing for LibreQuickSRNet super-resolution outputs."""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np
import torch


def _as_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, dict):
        output = output.get("restored", output.get("predictions", output.get("output")))
    if isinstance(output, (list, tuple)):
        output = output[0]
    if not isinstance(output, torch.Tensor):
        output = torch.as_tensor(output)
    return output


def postprocess(
    output: Any,
    original_size: Tuple[int, int],
    scale: int = 2,
) -> np.ndarray:
    """Crop to ``scale`` x original and return HWC uint8 RGB."""

    restored = _as_tensor(output).detach().float().cpu()
    if restored.ndim == 3:
        restored = restored.unsqueeze(0)
    if restored.ndim != 4 or restored.shape[1] != 3:
        raise ValueError(
            "QuickSRNet postprocess expects output shape [B, 3, H, W], "
            f"got {tuple(restored.shape)}."
        )
    orig_w, orig_h = original_size
    restored = restored[0, :, : orig_h * scale, : orig_w * scale]
    restored = restored.clamp_(0.0, 1.0)
    restored = restored.permute(1, 2, 0).mul(255.0).round().to(torch.uint8)
    return restored.numpy()


__all__ = ["postprocess"]
