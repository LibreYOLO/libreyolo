"""Training-time validation loss adapter for YOLO9-E2E detection."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from ..yolo9.validation_loss import YOLO9ValidationLoss
from .loss import YOLO9E2ELoss
from .nn import LibreYOLO9E2EModel, YOLO9E2EHead

# The neck feature maps LibreYOLO9Model publishes in its eval output, in
# stride order. The one-to-many branch is rebuilt from these.
_FEATURE_KEYS = ("x8", "x16", "x32")


class YOLO9E2EValidationLoss:
    """Evaluate the two-branch E2E loss from eval-mode raw head outputs.

    Inference runs the one-to-one branch only, so the one-to-many branch is
    rebuilt here from the neck features the eval forward already published
    (``head.branch_outputs``, the dense towers). That is one extra head pass,
    not a second backbone/neck pass, and it keeps the reported total covering
    the same two branches as training.
    """

    def __init__(self, model: nn.Module, *, max_labels: int) -> None:
        head_is_dual = type(model.head) is YOLO9E2EHead
        if type(model) is not LibreYOLO9E2EModel or not head_is_dual:
            raise TypeError(
                "YOLO9-E2E validation loss supports the standard detect model only"
            )

        self.max_labels = int(max_labels)
        if self.max_labels < 1:
            raise ValueError("YOLO9-E2E validation-loss max_labels must be at least 1")
        self.head = model.head
        self.num_classes = int(model.head.num_classes)
        self.device = next(model.parameters()).device
        self.loss = YOLO9E2ELoss(
            num_classes=self.num_classes,
            reg_max=int(model.head.reg_max),
            strides=list(model.head.strides),
            image_size=None,
            device=self.device,
            distributed_normalize=False,
        )

    def __call__(
        self,
        predictions: Any,
        targets: torch.Tensor,
        *,
        image_size: tuple[int, int],
    ) -> Mapping[str, torch.Tensor | float]:
        if not isinstance(predictions, Mapping) or "raw_outputs" not in predictions:
            raise ValueError(
                "YOLO9-E2E validation loss requires eval output containing "
                "raw_outputs"
            )
        exclusive_outputs = predictions["raw_outputs"]
        if not isinstance(exclusive_outputs, (list, tuple)) or not exclusive_outputs:
            raise ValueError("YOLO9-E2E raw_outputs must be a non-empty feature list")

        features = self._neck_features(predictions)
        dense_outputs = self.head.branch_outputs(features)

        height, width = image_size
        prepared = YOLO9ValidationLoss._prepare_targets(
            targets[:, : self.max_labels],
            image_size=image_size,
            num_classes=self.num_classes,
            device=self.device,
        )
        self.loss.update_anchors([width, height])
        values = self.loss(dense_outputs, exclusive_outputs, prepared)
        return {
            "loss": values["total_loss"],
            "loss/box": values["box_loss"],
            "loss/cls": values["cls_loss"],
            "loss/dfl": values["dfl_loss"],
        }

    @staticmethod
    def _neck_features(predictions: Mapping[str, Any]) -> list[torch.Tensor]:
        features = []
        for key in _FEATURE_KEYS:
            entry = predictions.get(key)
            tensor = entry.get("features") if isinstance(entry, Mapping) else None
            if not isinstance(tensor, torch.Tensor):
                raise ValueError(
                    "YOLO9-E2E validation loss requires eval output containing "
                    f"{key}.features"
                )
            features.append(tensor)
        return features


__all__ = ["YOLO9E2EValidationLoss"]
