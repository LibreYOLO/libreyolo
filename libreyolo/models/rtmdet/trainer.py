"""RTMDet trainer.

Experimental trainer subclassing :class:`BaseTrainer`. Reuses the shared
mosaic+mixup augmentation pipeline. Documented gaps vs upstream
mmdet/mmyolo (these matter for full-paper-parity, not for the
fine-tune-from-pretrained path that ``allow_experimental=True`` gates):

- mmdet uses ``CachedMosaic`` / ``CachedMixUp`` (FIFO of decoded images) for
  throughput; we use the standard non-cached pair.
- The two-stage pipeline switch (drop mosaic+mixup for the last
  ``stage2_num_epochs=20`` epochs) is approximated via the shared
  ``no_aug_epochs`` mechanism on ``BaseTrainer``, which closes mosaic but
  doesn't swap the full pipeline. Acceptable for short fine-tunes; revisit
  for production training.
- mmdet uses paramwise weight decay (norm_decay_mult=0, bias_decay_mult=0).
  This trainer uses ``BaseTrainer``'s default param grouping.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Type

import torch

from ...training.augment import MosaicMixupDataset, TrainTransform
from ...training.config import RTMDetConfig, TrainConfig
from ...training.scheduler import WarmupCosineScheduler
from ...training.trainer import BaseTrainer
from .loss import RTMDetLoss


class RTMDetTrainer(BaseTrainer):
    """RTMDet detection trainer (experimental)."""

    @classmethod
    def _config_class(cls) -> Type[TrainConfig]:
        return RTMDetConfig

    def get_model_family(self) -> str:
        return "rtmdet"

    def get_model_tag(self) -> str:
        return f"RTMDet-{self.config.size}"

    def create_transforms(self):
        preproc = TrainTransform(
            max_labels=120,
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
        def _scalar(name: str) -> float:
            v = outputs.get(name, 0.0)
            return float(v.detach().item()) if isinstance(v, torch.Tensor) else float(v)

        return {"cls": _scalar("loss_cls"), "bbox": _scalar("loss_bbox")}

    def on_setup(self) -> None:
        nc = getattr(self.model.head, "num_classes", 80)
        strides = tuple(getattr(self.model.head, "strides", (8, 16, 32)))
        self._loss_fn = RTMDetLoss(num_classes=nc, strides=strides).to(self.device)

    def on_forward(
        self,
        imgs: torch.Tensor,
        targets: torch.Tensor,
        polygons: Optional[List] = None,
    ) -> Dict:
        cls_scores, bbox_preds = self.model(imgs)

        # Targets: (B, max_labels, 5) [class, cx, cy, w, h] in pixel coords,
        # zero-padded. Convert to per-image (gt_boxes_xyxy, gt_labels) lists.
        gt_boxes_list = []
        gt_labels_list = []
        for b in range(targets.shape[0]):
            t = targets[b]
            valid = (t[:, 3:5] > 0).all(dim=1)  # w > 0 and h > 0
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
