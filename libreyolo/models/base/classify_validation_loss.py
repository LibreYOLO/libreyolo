"""Validation-loss adapter for the plain-softmax classification families.

ResNet, ConvNeXt, MobileNetV4 and EfficientNetV2 all train on a single
``F.cross_entropy(logits, targets)``, so one adapter covers the group. The
classification validator already holds the raw logits it scores top-1/top-5
from, which is exactly the criterion's input: there is no assignment step and
no second forward.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch

from ...training.classification import classification_loss

__all__ = ["ClassifyValidationLoss", "ClassifyValidationLossMixin"]


class ClassifyValidationLossMixin:
    """Opt a cross-entropy classification trainer into ``val_loss=True``.

    Mixed in ahead of ``BaseTrainer`` so it overrides the base gate, which
    rejects the flag for families that have not implemented it.
    """

    def validate_validation_loss_config(self) -> None:
        if not getattr(self.config, "val_loss", False):
            return
        task = getattr(getattr(self, "wrapper_model", None), "task", "classify")
        if task != "classify":
            raise ValueError(
                f"val_loss=True currently supports {self.get_model_family()} "
                "classification only; other tasks are not supported"
            )

    def build_validation_loss_adapter(self, model: torch.nn.Module):
        del model  # The validator feeds the adapter its own forward's logits.
        return ClassifyValidationLoss(
            device=self.device,
            family=self.get_model_family(),
            weights=getattr(self, "class_weights", None),
        )


class ClassifyValidationLoss:
    """Cross-entropy over the logits the classification validator produced."""

    # The validator hands over whole batches of class ids, not padded boxes,
    # so there is no target-capacity floor to raise.
    max_labels = None

    def __init__(self, *, device: torch.device, family: str, weights=None) -> None:
        self.device = device
        self.family = family
        self.weights = weights

    def __call__(
        self,
        predictions: Any,
        targets: torch.Tensor,
        *,
        image_size: tuple[int, int] | None = None,
    ) -> Mapping[str, torch.Tensor | float]:
        del image_size  # Cross-entropy does not depend on input geometry.
        logits = self._logits(predictions)
        labels = self._labels(
            targets, batch=logits.shape[0], num_classes=logits.shape[1]
        )
        loss = classification_loss(logits, labels, self.weights)
        return {"loss": loss, "loss/ce": loss}

    def _logits(self, predictions: Any) -> torch.Tensor:
        logits = predictions
        if isinstance(logits, (list, tuple)) and len(logits) == 1:
            logits = logits[0]
        if isinstance(logits, Mapping):
            logits = logits.get("logits", logits.get("predictions"))
        if not isinstance(logits, torch.Tensor):
            raise TypeError(
                f"{self.family} validation loss needs logits, got "
                f"{type(predictions).__name__}"
            )
        if logits.ndim != 2:
            raise ValueError(
                f"{self.family} validation logits must be [batch, classes], "
                f"got shape {tuple(logits.shape)}"
            )
        return logits.to(device=self.device, dtype=torch.float32)

    def _labels(
        self, targets: torch.Tensor, *, batch: int, num_classes: int
    ) -> torch.Tensor:
        if not isinstance(targets, torch.Tensor):
            targets = torch.as_tensor(targets)
        labels = targets.reshape(-1)
        if labels.numel() != batch:
            raise ValueError(
                f"{self.family} validation loss got {labels.numel()} labels "
                f"for {batch} images"
            )
        labels = labels.to(device=self.device, dtype=torch.long)
        # Checked against the head's own width, which is what cross_entropy
        # indexes during training too.
        invalid = (labels < 0) | (labels >= num_classes)
        if invalid.any():
            raise ValueError(
                f"{self.family} validation target class "
                f"{int(labels[invalid][0].item())} is outside "
                f"[0, {num_classes - 1}]"
            )
        return labels
