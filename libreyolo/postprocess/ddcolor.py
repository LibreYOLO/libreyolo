"""Centralized postprocessing for LibreDDColor colorization outputs.

Adapted from ``piddnad/DDColor/ddcolor/pipeline.py`` at commit
``2adb63f2656ac41cbdf7b894cddd94121a3faf13`` under Apache-2.0. LibreYOLO
modifies the pipeline to consume request-local lightness context and return
the library's RGB ``Results.restored`` payload. See the family ``NOTICE``.
"""

from __future__ import annotations

from typing import Any, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def _as_ab_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, dict):
        output = output.get("ab", output.get("predictions", output.get("output")))
    if isinstance(output, (list, tuple)):
        output = output[0]
    if not isinstance(output, torch.Tensor):
        output = torch.as_tensor(output)
    # Preserve the upstream dtype through interpolation; the pinned pipeline
    # moves to CPU first and casts to float only after resizing.
    output = output.detach().cpu()
    if output.ndim == 3:
        output = output.unsqueeze(0)
    if output.ndim != 4 or output.shape[1] != 2:
        raise ValueError(
            "DDColor postprocess expects Lab ab shape [B, 2, H, W], "
            f"got {tuple(output.shape)}."
        )
    return output


def postprocess(
    output: Any,
    original_size: Tuple[int, int],
    *,
    original_l: np.ndarray,
) -> np.ndarray:
    """Combine predicted ``ab`` with source ``L`` and return HWC uint8 RGB.

    This deliberately retains the pinned upstream operation order: default
    nearest-neighbor ``F.interpolate``, Lab-to-BGR in OpenCV float space,
    round/cast to uint8, then BGR-to-RGB channel reversal for LibreYOLO's
    public result convention.
    """

    ab = _as_ab_tensor(output)
    original_w, original_h = original_size
    expected_l_shape = (original_h, original_w, 1)
    original_l = np.asarray(original_l, dtype=np.float32)
    if original_l.shape != expected_l_shape:
        raise ValueError(
            "DDColor original L plane does not match the source canvas: "
            f"expected {expected_l_shape}, got {original_l.shape}."
        )

    # No mode argument on purpose: the official pipeline relies on PyTorch's
    # default nearest-neighbor interpolation for the predicted chroma.
    resized_ab = (
        F.interpolate(ab, size=(original_h, original_w))[0]
        .float()
        .numpy()
        .transpose(1, 2, 0)
    )
    output_lab = np.concatenate((original_l, resized_ab), axis=-1)
    output_bgr = cv2.cvtColor(output_lab, cv2.COLOR_LAB2BGR)
    output_bgr_uint8 = (output_bgr * 255.0).round().astype(np.uint8)
    return np.ascontiguousarray(output_bgr_uint8[..., ::-1])


__all__ = ["postprocess"]
