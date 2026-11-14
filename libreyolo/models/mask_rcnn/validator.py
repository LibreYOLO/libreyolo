"""Validation adapter for variable-resolution Mask R-CNN input."""

from __future__ import annotations

import logging
from typing import Any

import torch

from ...validation.detection_validator import SegmentationValidator

logger = logging.getLogger(__name__)


class MaskRCNNValidator(SegmentationValidator):
    """Run unresized images one at a time through the in-graph transform."""

    def _setup_dataloader(self):
        if self.config.batch_size != 1:
            logger.info(
                "Mask R-CNN validation uses batch_size=1 because source "
                "images retain their original dimensions."
            )
            self.config.batch_size = 1
        return super()._setup_dataloader()

    def _slice_batch_predictions(self, preds: Any, batch_idx: int) -> Any:
        if isinstance(preds, list) and all(isinstance(p, dict) for p in preds):
            return preds[batch_idx]
        if isinstance(preds, (list, tuple)) and all(
            isinstance(p, torch.Tensor) for p in preds
        ):
            # Exported graphs return (boxes, scores, labels, masks) for the
            # single image of the batch, with no batch axis to slice.
            return preds
        raise TypeError(
            "Mask R-CNN validation expected a list of detection dictionaries "
            "or the exported (boxes, scores, labels, masks) outputs"
        )
