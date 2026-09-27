"""Postprocessing for guided ViTMatte alpha probabilities."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch


def postprocess(
    output: Any,
    original_size: Tuple[int, int],
) -> Dict[str, torch.Tensor]:
    """Crop a padded probability alpha to the original image canvas.

    ViTMatte applies sigmoid inside its detail-capture decoder. This function
    deliberately does not apply a second sigmoid or resize the alpha: public
    preprocessing changes only the bottom/right padding, so a crop restores
    the exact source canvas.
    """
    alpha = output
    if hasattr(alpha, "alphas"):
        alpha = alpha.alphas
    elif isinstance(alpha, dict):
        alpha = alpha.get("alphas", alpha.get("matte"))
    elif isinstance(alpha, (list, tuple)):
        alpha = alpha[0]
    if not isinstance(alpha, torch.Tensor):
        alpha = torch.as_tensor(alpha)

    if alpha.ndim == 4:
        if alpha.shape[0] != 1 or alpha.shape[1] != 1:
            raise ValueError(
                "ViTMatte postprocess expects one (1, 1, H, W) alpha; "
                f"got {tuple(alpha.shape)}."
            )
        alpha = alpha[0, 0]
    elif alpha.ndim == 3:
        if alpha.shape[0] != 1:
            raise ValueError(
                "ViTMatte postprocess expects one alpha channel; "
                f"got {tuple(alpha.shape)}."
            )
        alpha = alpha[0]
    elif alpha.ndim != 2:
        raise ValueError(
            f"ViTMatte postprocess expects a 2D alpha map; got {tuple(alpha.shape)}."
        )

    original_width, original_height = original_size
    if alpha.shape[-2] < original_height or alpha.shape[-1] < original_width:
        raise ValueError(
            "ViTMatte output is smaller than the source canvas: "
            f"output={tuple(alpha.shape)}, source={(original_height, original_width)}."
        )
    alpha = alpha[:original_height, :original_width]
    return {"matte": alpha.float().clamp(0.0, 1.0).detach().cpu()}


__all__ = ["postprocess"]
