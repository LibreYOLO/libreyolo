"""PICODET trainer.

Thin subclass of :class:`BaseTrainer` that:

* Reuses the YOLO-grid augmentation path (LibreYOLO's shared
  ``training/augment.py``) — at the cost of a documented gap vs Bo's
  upstream ``MinIoURandomCrop`` / ``PhotoMetricDistortion`` / multiscale
  resize. v1 ships with hflip + ImageNet normalisation; the upstream-
  faithful augmentations land in a follow-up.
* Converts BaseTrainer's padded ``(B, max_labels, 5)`` ``[class, cx, cy, w, h]``
  pixel-coord targets into the ``(gt_boxes_xyxy, gt_labels)`` per-image
  lists that :class:`PICODETLoss` consumes.
* Returns the loss dict.
"""

from __future__ import annotations

from typing import Dict, Type

import torch

from ...training.augment import MosaicMixupDataset, TrainTransform
from ...training.config import PICODETConfig, TrainConfig
from ...training.scheduler import WarmupCosineScheduler
from ...training.trainer import BaseTrainer
from .loss import PICODETLoss


class PICODETTrainer(BaseTrainer):
    """PICODET trainer."""

    @classmethod
    def _config_class(cls) -> Type[TrainConfig]:
        return PICODETConfig

    def get_model_family(self) -> str:
        return "picodet"

    def get_model_tag(self) -> str:
        return f"PICODET-{self.config.size}"

    def create_transforms(self):
        # Mosaic prob 0 + mixup prob 0 in PICODETConfig effectively makes
        # this a hflip + normalisation pipeline.
        preproc = TrainTransform(
            max_labels=50,
            flip_prob=self.config.flip_prob,
            hsv_prob=self.config.hsv_prob,
        )
        return preproc, MosaicMixupDataset

    def create_scheduler(self, iters_per_epoch: int):
        return WarmupCosineScheduler(
            lr=self.effective_lr,
            iters_per_epoch=iters_per_epoch,
            total_epochs=self.config.epochs,
            warmup_epochs=self.config.warmup_epochs,
            warmup_lr_start=self.config.warmup_lr_start,
            plateau_epochs=self.config.no_aug_epochs,
            min_lr_ratio=self.config.min_lr_ratio,
        )

    def get_loss_components(self, outputs: Dict) -> Dict[str, float]:
        return {
            "cls": float(outputs.get("loss_cls", 0)),
            "bbox": float(outputs.get("loss_bbox", 0)),
            "dfl": float(outputs.get("loss_dfl", 0)),
        }

    def on_setup(self) -> None:
        # Build the loss module once with the model's actual class count.
        nc = getattr(self.model.head, "num_classes", 80)
        reg_max = getattr(self.model.head, "reg_max", 7)
        strides = tuple(getattr(self.model.head, "strides", (8, 16, 32, 64)))
        self._loss_fn = PICODETLoss(
            num_classes=nc,
            reg_max=reg_max,
            strides=strides,
        ).to(self.device)

    def on_forward(self, imgs: torch.Tensor, targets: torch.Tensor, polygons=None) -> Dict:
        cls_scores, bbox_preds = self.model(imgs)

        # Targets: (B, max_labels, 5) [class, cx, cy, w, h] in pixel coords,
        # zero-padded. Convert to per-image (gt_boxes_xyxy, gt_labels) lists.
        gt_boxes_list = []
        gt_labels_list = []
        for b in range(targets.shape[0]):
            t = targets[b]
            valid = (t[:, 2:4] > 0).all(dim=1)  # w > 0 and h > 0
            t = t[valid]
            if t.shape[0] == 0:
                gt_boxes_list.append(t.new_zeros((0, 4)))
                gt_labels_list.append(t.new_zeros((0,), dtype=torch.long))
                continue
            cls = t[:, 0].long()
            cx, cy, w, h = t[:, 1], t[:, 2], t[:, 3], t[:, 4]
            x1 = cx - w * 0.5
            y1 = cy - h * 0.5
            x2 = cx + w * 0.5
            y2 = cy + h * 0.5
            gt_boxes_list.append(torch.stack([x1, y1, x2, y2], dim=-1))
            gt_labels_list.append(cls)

        return self._loss_fn(cls_scores, bbox_preds, gt_boxes_list, gt_labels_list)
