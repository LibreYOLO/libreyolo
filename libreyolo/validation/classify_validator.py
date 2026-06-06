"""Image-classification validator for LibreYOLO.

Computes top-1 and top-5 accuracy over an ImageFolder-style validation split,
reusing the :class:`BaseValidator` template (setup -> iterate -> finalize).
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader

from ..data.classify_dataset import (
    ClassifyDataset,
    classify_collate_fn,
    get_class_names,
    resolve_classify_data,
)
from .base import BaseValidator

logger = logging.getLogger(__name__)


class ClassifyValidator(BaseValidator):
    """Top-1/top-5 accuracy validator for the classification task."""

    task = "classify"

    def _setup_dataloader(self) -> DataLoader:
        dataset_root = resolve_classify_data(self.config.data)
        split = self.config.split or "val"
        # Label indices are pinned to the train split so they line up with the
        # model head's output channels regardless of split folder ordering.
        train_classes = get_class_names(dataset_root, split="train")
        class_to_idx = {name: i for i, name in enumerate(train_classes)}

        dataset = ClassifyDataset(
            dataset_root=dataset_root,
            split=split,
            imgsz=self.config.imgsz,
            augment=False,
            class_to_idx=class_to_idx,
        )
        self._num_classes = len(train_classes)
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.device.type == "cuda",
            collate_fn=classify_collate_fn,
        )

    def _init_metrics(self) -> None:
        self._top1_correct = 0
        self._top5_correct = 0
        self._total = 0

    def _preprocess_batch(self, batch: Any) -> tuple:
        images, targets, img_info, img_ids = batch
        return images, targets, img_info, img_ids

    def _postprocess_predictions(self, preds: Any, batch: Any) -> Any:
        # ``preds`` are raw logits [B, num_classes]; nothing to decode.
        if isinstance(preds, (list, tuple)) and len(preds) == 1:
            preds = preds[0]
        if isinstance(preds, dict) or isinstance(preds, torch.Tensor):
            return preds
        return torch.as_tensor(preds)

    def _update_metrics(
        self, preds: Any, targets: Any, img_info: Any, img_ids: Any = None
    ) -> None:
        logits = preds
        if isinstance(logits, dict):
            logits = logits.get("logits", logits.get("predictions"))
        logits = logits.detach().float().cpu()
        targets = targets.detach().cpu().view(-1)

        num_classes = logits.shape[1]
        k = min(5, num_classes)
        topk = logits.topk(k, dim=1).indices  # [B, k]
        correct = topk == targets.unsqueeze(1)

        self._top1_correct += int(correct[:, 0].sum().item())
        self._top5_correct += int(correct.any(dim=1).sum().item())
        self._total += int(targets.numel())

    def _compute_metrics(self) -> Dict[str, float]:
        total = max(self._total, 1)
        top1 = self._top1_correct / total
        top5 = self._top5_correct / total
        return {
            "metrics/accuracy_top1": top1,
            "metrics/accuracy_top5": top5,
            "fitness": top1,
        }

    def _print_results(self, metrics: Dict[str, float]) -> None:
        logger.info("=" * 50)
        logger.info("Classification Validation Results")
        logger.info("=" * 50)
        logger.info("  top-1 accuracy: %.4f", metrics.get("metrics/accuracy_top1", 0.0))
        logger.info("  top-5 accuracy: %.4f", metrics.get("metrics/accuracy_top5", 0.0))
        logger.info("  images: %d", self._total)
        logger.info("=" * 50)
