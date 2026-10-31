"""Training-time validation loss adapter for standard YOLO9 detection."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from .loss import YOLO9Loss
from .nn import LibreYOLO9Model, YOLO9Head


class YOLO9ValidationLoss:
    """Evaluate YOLO9's training loss from eval-mode raw head outputs."""

    def __init__(self, model: nn.Module, *, max_labels: int) -> None:
        # yolo9_p2 is the same dense head with a fourth stride, so it shares
        # this adapter; the strides below come from the head either way.
        if not isinstance(model, LibreYOLO9Model) or type(model.head) is not YOLO9Head:
            raise TypeError(
                "YOLO9 validation loss supports the standard detect model only"
            )

        self.max_labels = int(max_labels)
        if self.max_labels < 1:
            raise ValueError("YOLO9 validation-loss max_labels must be at least 1")
        self.num_classes = int(model.head.num_classes)
        self.device = next(model.parameters()).device
        self.loss = YOLO9Loss(
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
                "YOLO9 validation loss requires eval output containing raw_outputs"
            )
        raw_outputs = predictions["raw_outputs"]
        if not isinstance(raw_outputs, (list, tuple)) or not raw_outputs:
            raise ValueError("YOLO9 raw_outputs must be a non-empty feature list")

        height, width = image_size
        prepared = self._prepare_targets(
            targets[:, : self.max_labels],
            image_size=image_size,
            num_classes=self.num_classes,
            device=self.device,
        )
        self.loss.update_anchors([width, height])
        values = self.loss(raw_outputs, prepared)
        return {
            "loss": values["total_loss"],
            "loss/box": values["box_loss"],
            "loss/cls": values["cls_loss"],
            "loss/dfl": values["dfl_loss"],
        }

    @staticmethod
    def _prepare_targets(
        targets: torch.Tensor,
        *,
        image_size: tuple[int, int],
        num_classes: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Convert padded validation ``xyxy,class`` pixels to YOLO9 format."""
        if targets.ndim != 3 or targets.shape[-1] < 5:
            raise ValueError(
                "YOLO9 validation targets must have shape [batch, labels, >=5]"
            )

        source = targets[..., :5].to(
            device=device,
            dtype=torch.float32,
            non_blocking=device.type == "cuda",
        )
        valid = (source[..., 2] > source[..., 0]) & (source[..., 3] > source[..., 1])
        labels = source[..., 4]
        invalid_labels = valid & (
            (labels < 0) | (labels >= num_classes) | (labels != labels.round())
        )
        if invalid_labels.any():
            bad_label = float(labels[invalid_labels][0].item())
            raise ValueError(
                f"YOLO9 validation target class {bad_label:g} is outside "
                f"[0, {num_classes - 1}]"
            )

        batch_size = int(source.shape[0])
        max_targets = int(valid.sum(dim=1).max().item()) if batch_size > 0 else 0
        prepared = torch.zeros(
            (batch_size, max_targets, 5),
            dtype=torch.float32,
            device=device,
        )
        if max_targets == 0:
            return prepared
        prepared[..., 0] = -1

        height, width = image_size
        scale = torch.tensor(
            [width, height, width, height],
            dtype=torch.float32,
            device=device,
        )
        for batch_idx in range(batch_size):
            rows = source[batch_idx, valid[batch_idx]]
            count = int(rows.shape[0])
            if count == 0:
                continue
            prepared[batch_idx, :count, 0] = rows[:, 4]
            prepared[batch_idx, :count, 1:5] = (rows[:, :4] / scale).clamp_(0.0, 1.0)
        return prepared


__all__ = ["YOLO9ValidationLoss"]
