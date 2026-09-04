"""U-Net semantic-logit extraction and source-resolution restoration."""

from __future__ import annotations

from typing import Any, Tuple

import torch
import torch.nn.functional as F


def primary_logits(output: Any) -> torch.Tensor:
    """Return the main-head dense logits, ignoring an auxiliary training tuple."""
    logits = output
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    if isinstance(logits, dict):
        logits = logits.get("semantic_logits", logits.get("out", logits.get("predictions")))
        if logits is None:
            raise RuntimeError(
                "U-Net forward output carries no primary logits; got keys "
                f"{sorted(output)}"
            )
    logits = torch.as_tensor(logits)
    if logits.ndim != 4:
        raise ValueError(
            "U-Net semantic postprocess expects [B, C, H, W] logits, got "
            f"shape {tuple(logits.shape)}."
        )
    return logits


def resize_logits(output: Any, original_size: Tuple[int, int]) -> torch.Tensor:
    """Return primary float32 logits at ``original_size`` (width, height)."""
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
    original_size: Tuple[int, int],
    max_det: int = 300,
    **_unused,
) -> dict[str, torch.Tensor]:
    """Return one semantic class id per source pixel."""
    del conf_thres, iou_thres, max_det
    logits = resize_logits(output, original_size)
    return {"semantic": logits.argmax(dim=1)[0].cpu()}


__all__ = ["postprocess", "primary_logits", "resize_logits"]
