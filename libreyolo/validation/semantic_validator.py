"""Semantic-segmentation validator for LibreYOLO.

Computes mean IoU and pixel accuracy from a confusion matrix accumulated over
a dense-mask validation split, reusing the :class:`BaseValidator` template
(setup -> iterate -> finalize). Predictions and targets are compared at the
model input resolution; ignore pixels (255 by default) are excluded.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..data.semantic_dataset import (
    IGNORE_INDEX,
    SemanticDataset,
    resolve_semantic_data,
    semantic_collate_fn,
)
from .base import BaseValidator

logger = logging.getLogger(__name__)


class SemanticValidator(BaseValidator):
    """mIoU / pixel-accuracy validator for the semantic task."""

    task = "semantic"

    def _setup_dataloader(self) -> DataLoader:
        if not self.config.data:
            raise ValueError("Semantic validation requires data= (a dataset YAML).")
        data_config = resolve_semantic_data(self.config.data)
        split = self.config.split or "val"

        resize_mode = getattr(self.model, "semantic_resize_mode", "letterbox")
        dataset = SemanticDataset(
            data_config,
            split=split,
            imgsz=self.config.imgsz,
            augment=False,
            resize_mode=resize_mode,
        )

        model_nc = getattr(self.model, "nb_classes", None)
        if model_nc is not None and int(model_nc) != dataset.nc:
            raise ValueError(
                f"Semantic dataset has {dataset.nc} classes but the model "
                f"predicts {int(model_nc)}. Use a matching dataset/checkpoint."
            )

        self._num_classes = dataset.nc
        self._class_names = dict(dataset.names)
        self._ignore_index = dataset.ignore_index
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.device.type == "cuda",
            collate_fn=semantic_collate_fn,
        )

    def _init_metrics(self) -> None:
        nc = getattr(self, "_num_classes", 0)
        self._confusion = torch.zeros((nc, nc), dtype=torch.int64)

    def _preprocess_batch(self, batch: Any) -> tuple:
        images, targets, img_info, img_ids = batch
        return images, targets, img_info, img_ids

    def _postprocess_predictions(self, preds: Any, batch: Any) -> Any:
        """Decode raw model output into ``[B, H, W]`` class maps."""
        logits = preds
        if isinstance(logits, dict):
            logits = logits.get("semantic_logits", logits.get("logits"))
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        logits = torch.as_tensor(logits)

        targets = batch[1]
        target_hw = tuple(targets.shape[-2:])
        if logits.ndim != 4:
            raise ValueError(
                f"Semantic validation expects [B, C, H, W] logits, got shape "
                f"{tuple(logits.shape)}."
            )
        if tuple(logits.shape[-2:]) != target_hw:
            logits = F.interpolate(
                logits.float(), size=target_hw, mode="bilinear", align_corners=False
            )
        return logits.argmax(dim=1)

    def _update_metrics(
        self, preds: Any, targets: Any, img_info: Any, img_ids: Any = None
    ) -> None:
        pred_maps = preds.detach().cpu().long().view(-1)
        target_maps = targets.detach().cpu().long().view(-1)

        valid = target_maps != self._ignore_index
        if not bool(valid.any()):
            return
        target_valid = target_maps[valid]
        pred_valid = pred_maps[valid].clamp_(0, self._num_classes - 1)

        index = target_valid * self._num_classes + pred_valid
        counts = torch.bincount(index, minlength=self._num_classes**2)
        self._confusion += counts.reshape(self._num_classes, self._num_classes)

    def _per_class_iou(self) -> torch.Tensor:
        confusion = self._confusion.double()
        true_positive = confusion.diag()
        union = confusion.sum(dim=0) + confusion.sum(dim=1) - true_positive
        iou = torch.full_like(true_positive, float("nan"))
        present = union > 0
        iou[present] = true_positive[present] / union[present]
        return iou

    def _compute_metrics(self) -> Dict[str, float]:
        total = self._confusion.sum()
        iou = self._per_class_iou()
        observed = ~torch.isnan(iou)
        miou = float(iou[observed].mean()) if bool(observed.any()) else 0.0
        accuracy = float(self._confusion.diag().sum() / total) if total > 0 else 0.0
        return {
            "metrics/mIoU": miou,
            "metrics/pixel_accuracy": accuracy,
            "fitness": miou,
        }

    def _print_results(self, metrics: Dict[str, float]) -> None:
        logger.info("=" * 50)
        logger.info("Semantic Segmentation Validation Results")
        logger.info("=" * 50)
        iou = self._per_class_iou()
        for class_id in range(self._num_classes):
            value = iou[class_id]
            if torch.isnan(value):
                continue
            name = self._class_names.get(class_id, str(class_id))
            logger.info("  IoU %-20s %.4f", name, float(value))
        logger.info("  mIoU:           %.4f", metrics.get("metrics/mIoU", 0.0))
        logger.info(
            "  pixel accuracy: %.4f", metrics.get("metrics/pixel_accuracy", 0.0)
        )
        logger.info("=" * 50)


__all__ = ["SemanticValidator", "IGNORE_INDEX"]
