"""Training-time validation loss adapter for PP-YOLOE detection."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from ..base.validation_loss import padded_targets_to_flat_pixels
from ..yolonas.loss import PPYoloELoss
from .nn import LibrePPYOLOEModel


class PPYOLOEValidationLoss:
    """Evaluate the PP-YOLOE training loss from eval-mode raw head outputs.

    Validation always uses the task-aligned assigner, independent of where the
    training run currently sits in its ATSS schedule, so the reported number
    stays comparable across the assigner switch.
    """

    max_labels = 100

    def __init__(self, model: nn.Module, *, max_labels: int) -> None:
        if type(model) is not LibrePPYOLOEModel:
            raise TypeError("PP-YOLOE validation loss supports the detect model only")

        self.max_labels = int(max_labels)
        if self.max_labels < 1:
            raise ValueError("PP-YOLOE validation-loss max_labels must be at least 1")
        self.device = next(model.parameters()).device
        self.num_classes = int(model.nc)
        self.loss = PPYoloELoss(
            num_classes=self.num_classes,
            use_static_assigner=False,
            use_varifocal_loss=True,
            distributed_normalize=False,
        ).to(self.device)

    def __call__(
        self,
        predictions: Any,
        targets: torch.Tensor,
        *,
        image_size: tuple[int, int],
    ) -> Mapping[str, torch.Tensor | float]:
        del image_size  # PP-YOLOE assigns in pixels; the head carries the anchors.
        if not isinstance(predictions, Mapping) or "raw_predictions" not in predictions:
            raise ValueError(
                "PP-YOLOE validation loss requires eval output containing "
                "raw_predictions"
            )

        prepared = padded_targets_to_flat_pixels(
            targets[:, : self.max_labels],
            num_classes=self.num_classes,
            device=self.device,
            family="PP-YOLOE",
        )
        _, log_losses = self.loss(predictions["raw_predictions"], prepared)
        # ``log_losses`` is [cls, iou, dfl, total], each already multiplied by
        # its configured weight, so the three components sum to the total.
        return {
            "loss": log_losses[3],
            "loss/cls": log_losses[0],
            "loss/iou": log_losses[1],
            "loss/dfl": log_losses[2],
        }


__all__ = ["PPYOLOEValidationLoss"]
