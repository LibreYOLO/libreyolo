"""U-Net semantic loss: softmax CE on the main head plus 0.4x auxiliary CE.

This is the mmseg ``fcn_unet_s5-d16`` Cityscapes decode/aux loss pair
(``loss_weight`` 1.0 / 0.4), not the 2015 weighted touching-cell loss.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .nn import IGNORE_INDEX


class UNetLoss(nn.Module):
    """Pixel-wise cross-entropy over main and auxiliary U-Net logits."""

    def __init__(self, ignore_index: int = IGNORE_INDEX, aux_weight: float = 0.4) -> None:
        super().__init__()
        self.ignore_index = int(ignore_index)
        self.aux_weight = float(aux_weight)

    def _ce(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if tuple(logits.shape[-2:]) != tuple(target.shape[-2:]):
            logits = F.interpolate(
                logits, size=target.shape[-2:], mode="bilinear", align_corners=False
            )
        if bool((target != self.ignore_index).any()):
            return F.cross_entropy(logits, target, ignore_index=self.ignore_index)
        return logits.sum() * 0.0

    def forward(self, outputs: Any, targets: torch.Tensor) -> dict[str, torch.Tensor]:
        if isinstance(outputs, (tuple, list)):
            main, aux = outputs[0], outputs[1]
        else:
            main, aux = outputs, None
        if targets.ndim == 4 and targets.shape[1] == 1:
            targets = targets[:, 0]
        loss_main = self._ce(main, targets)
        components = {"loss_ce": loss_main, "loss": loss_main}
        if aux is not None:
            loss_aux = self._ce(aux, targets)
            components["loss_aux"] = loss_aux
            components["loss"] = loss_main + self.aux_weight * loss_aux
        return components


__all__ = ["IGNORE_INDEX", "UNetLoss"]
