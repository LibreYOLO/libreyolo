"""DDRNet semantic-logit extraction and source-resolution restoration."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def primary_logits(output: Any) -> torch.Tensor:
    """Return the main-head stride-8 logits, ignoring the auxiliary training pair."""
    logits = output
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    if isinstance(logits, dict):
        logits = logits.get("semantic_logits", logits.get("predictions"))
        if logits is None:
            raise RuntimeError(
                "DDRNet forward output carries no primary logits; got keys "
                f"{sorted(output)}"
            )
    logits = torch.as_tensor(logits)
    if logits.ndim != 4:
        raise ValueError(
            "DDRNet semantic postprocess expects [B, C, H, W] logits, got "
            f"shape {tuple(logits.shape)}."
        )
    return logits


def resize_logits(output: Any, original_size: tuple[int, int]) -> torch.Tensor:
    """Return primary float32 logits at ``original_size`` (width, height).

    Bilinear with ``align_corners=False`` from the stride-8 map straight to the
    source canvas, as the upstream evaluation does; the direct-resize
    preprocess leaves no padded region to trim.
    """
    orig_w, orig_h = original_size
    return F.interpolate(
        primary_logits(output).float(),
        size=(orig_h, orig_w),
        mode="bilinear",
        align_corners=False,
    )


def postprocess(
    output: Any,
    conf_thres: float,
    iou_thres: float,
    original_size: tuple[int, int],
    max_det: int = 300,
    **_unused,
) -> dict[str, torch.Tensor]:
    """Return one semantic class id per source pixel."""
    del conf_thres, iou_thres, max_det
    logits = resize_logits(output, original_size)
    return {"semantic": logits.argmax(dim=1)[0].cpu()}


__all__ = ["postprocess", "primary_logits", "resize_logits"]
