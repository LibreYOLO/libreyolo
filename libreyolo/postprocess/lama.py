"""Centralized original-canvas postprocessing for LibreLaMa."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import torch


def _as_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, dict):
        output = output.get("output", output.get("restored", output.get("predictions")))
    if isinstance(output, (list, tuple)):
        output = output[0]
    if not isinstance(output, torch.Tensor):
        output = torch.as_tensor(output)
    return output


def postprocess(
    output: Any,
    original_size: tuple[int, int],
    *,
    original_rgb: np.ndarray,
    fill_mask: np.ndarray,
) -> np.ndarray:
    """Resize BGR graph output and preserve every unmasked source pixel exactly."""

    restored = _as_tensor(output).detach().float().cpu()
    if restored.ndim == 3:
        restored = restored.unsqueeze(0)
    if restored.ndim != 4 or restored.shape[0] != 1 or restored.shape[1] != 3:
        raise ValueError(
            "LibreLaMa postprocess expects output shape [1, 3, 512, 512], "
            f"got {tuple(restored.shape)}."
        )
    orig_w, orig_h = (int(original_size[0]), int(original_size[1]))
    original_rgb = np.asarray(original_rgb, dtype=np.uint8)
    fill_mask = np.asarray(fill_mask, dtype=bool)
    if original_rgb.shape != (orig_h, orig_w, 3):
        raise ValueError(
            "LibreLaMa original RGB context does not match original_size: "
            f"shape={original_rgb.shape}, size={original_size}."
        )
    if fill_mask.shape != (orig_h, orig_w):
        raise ValueError(
            "LibreLaMa fill mask context does not match original_size: "
            f"shape={fill_mask.shape}, size={original_size}."
        )

    bgr = restored[0].permute(1, 2, 0).numpy()
    bgr = cv2.resize(bgr, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    rgb = np.ascontiguousarray(bgr[..., ::-1])
    result = np.clip(rgb, 0.0, 255.0).astype(np.uint8)
    np.copyto(result, original_rgb, where=~fill_mask[..., None])
    return result


__all__ = ["postprocess"]
