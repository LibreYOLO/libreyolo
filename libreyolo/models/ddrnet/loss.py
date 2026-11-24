"""DDRNet semantic loss: OHEM cross-entropy on both heads, auxiliary at 0.4.

Paper (arXiv:2101.06085, section III-D and IV-B): the final loss is
``L = L_main + 0.4 * L_aux`` with the auxiliary head discarded at test time,
and Cityscapes training uses online hard example mining. The OHEM rule here is
the standard pixel-level one: a valid pixel contributes when its ground-truth
class probability is below ``thresh``, and the ``min_kept`` hardest valid
pixels always contribute, so a confident batch never averages over a handful
of pixels. ``thresh=0.9`` and ``min_kept=131072`` are the values of the
training code the official README points to.

Logits arrive at stride 8 and are bilinearly upsampled (``align_corners=False``)
to the label size before the loss, as upstream does.

Not reproduced: the Cityscapes-specific per-class weights upstream baked into
its criterion. They are dataset statistics, not part of the model; the loss is
unweighted on every dataset.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .nn import IGNORE_INDEX


class OhemCrossEntropy(nn.Module):
    """Pixel-level OHEM over softmax cross-entropy, ignore pixels excluded."""

    def __init__(
        self,
        ignore_index: int = IGNORE_INDEX,
        thresh: float = 0.9,
        min_kept: int = 131072,
    ) -> None:
        super().__init__()
        if not 0.0 < float(thresh) <= 1.0:
            raise ValueError(f"OHEM thresh must be in (0, 1], got {thresh}")
        self.ignore_index = int(ignore_index)
        self.thresh = float(thresh)
        self.min_kept = max(1, int(min_kept))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pixel_losses = F.cross_entropy(
            logits, target, ignore_index=self.ignore_index, reduction="none"
        ).reshape(-1)
        valid = target.reshape(-1) != self.ignore_index
        if not bool(valid.any()):
            return logits.sum() * 0.0

        with torch.no_grad():
            safe_target = torch.where(
                target == self.ignore_index, torch.zeros_like(target), target
            )
            gt_prob = (
                logits.detach()
                .float()
                .softmax(dim=1)
                .gather(1, safe_target.unsqueeze(1))
                .reshape(-1)[valid]
            )
            kept = min(self.min_kept, gt_prob.numel())
            kth_prob = gt_prob.sort().values[kept - 1]
            keep = (gt_prob < self.thresh) | (gt_prob <= kth_prob)

        return pixel_losses[valid][keep].mean()


class DDRNetLoss(nn.Module):
    """Main-head OHEM CE plus ``aux_weight`` times auxiliary-head OHEM CE."""

    def __init__(
        self,
        ignore_index: int = IGNORE_INDEX,
        aux_weight: float = 0.4,
        ohem: bool = True,
        ohem_thresh: float = 0.9,
        ohem_min_kept: int = 131072,
    ) -> None:
        super().__init__()
        self.ignore_index = int(ignore_index)
        self.aux_weight = float(aux_weight)
        self.ohem = bool(ohem)
        self.criterion = OhemCrossEntropy(
            ignore_index=ignore_index, thresh=ohem_thresh, min_kept=ohem_min_kept
        )

    def _loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if tuple(logits.shape[-2:]) != tuple(target.shape[-2:]):
            logits = F.interpolate(
                logits, size=target.shape[-2:], mode="bilinear", align_corners=False
            )
        if self.ohem:
            return self.criterion(logits, target)
        if bool((target != self.ignore_index).any()):
            return F.cross_entropy(logits, target, ignore_index=self.ignore_index)
        return logits.sum() * 0.0

    def forward(self, outputs: Any, targets: torch.Tensor) -> dict[str, torch.Tensor]:
        if isinstance(outputs, (tuple, list)):
            main, aux = outputs[0], outputs[1] if len(outputs) > 1 else None
        else:
            main, aux = outputs, None
        if targets.ndim == 4 and targets.shape[1] == 1:
            targets = targets[:, 0]
        targets = targets.long()
        loss_main = self._loss(main, targets)
        components = {"loss_ce": loss_main, "loss": loss_main}
        if aux is not None:
            loss_aux = self._loss(aux, targets)
            components["loss_aux"] = loss_aux
            components["loss"] = loss_main + self.aux_weight * loss_aux
        return components


__all__ = ["IGNORE_INDEX", "DDRNetLoss", "OhemCrossEntropy"]
