"""DAMO-YOLO trainer.

Reuses LibreYOLO's shared YOLO-grid augmentation pipeline (mosaic+mixup,
hflip, hsv, no_aug_epochs ramp-down) since upstream's recipe is the same
schedule. Targets arrive from BaseTrainer as padded ``(B, max_labels, 5)``
``[class, cx, cy, w, h]`` pixel-coord tensors; we convert to per-image
``(boxes_xyxy, labels)`` lists for the head's GFL+AlignOTA loss.
"""

from __future__ import annotations

from typing import Dict, Type

import torch

from ...training.augment import MosaicMixupDataset, TrainTransform
from ...training.config import DAMOYOLOConfig, TrainConfig
from ...training.scheduler import WarmupCosineScheduler
from ...training.trainer import BaseTrainer


class DAMOYOLOTrainer(BaseTrainer):
    """DAMO-YOLO trainer (mosaic+mixup, GFL+AlignOTA loss in head)."""

    @classmethod
    def _config_class(cls) -> Type[TrainConfig]:
        return DAMOYOLOConfig

    def get_model_family(self) -> str:
        return "damoyolo"

    def get_model_tag(self) -> str:
        return f"DAMOYOLO-{self.config.size}"

    def create_transforms(self):
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
        def _scalar(v):
            return float(v.detach()) if isinstance(v, torch.Tensor) else float(v)
        return {
            "cls": _scalar(outputs.get("loss_cls", 0)),
            "bbox": _scalar(outputs.get("loss_bbox", 0)),
            "dfl": _scalar(outputs.get("loss_dfl", 0)),
        }

    def on_mosaic_disable(self) -> None:
        """Final no_aug_epochs phase: drop mosaic, keep hflip + hsv."""
        if hasattr(self.train_loader.dataset, "close_mosaic"):
            self.train_loader.dataset.close_mosaic()

    def on_forward(self, imgs: torch.Tensor, targets: torch.Tensor, polygons=None) -> Dict:
        # BaseTrainer hands us images BGR uint8-style float32 from
        # MosaicMixupDataset / TrainTransform — but DAMO-YOLO trains on
        # RGB. The mosaic dataset emits images in cv2 BGR order, so swap
        # channels here. Both upstream and our val use RGB.
        imgs = imgs[:, [2, 1, 0], :, :]

        # Convert (B, max_labels, 5) [class, cx, cy, w, h] padded targets
        # into per-image dicts {boxes: xyxy, labels:}.
        target_list = []
        for b in range(targets.shape[0]):
            t = targets[b]
            valid = (t[:, 2:4] > 0).all(dim=1)
            t = t[valid]
            if t.shape[0] == 0:
                target_list.append(
                    {"boxes": t.new_zeros((0, 4)), "labels": t.new_zeros((0,), dtype=torch.long)}
                )
                continue
            cls = t[:, 0].long()
            cx, cy, w, h = t[:, 1], t[:, 2], t[:, 3], t[:, 4]
            boxes = torch.stack([cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5], dim=-1)
            target_list.append({"boxes": boxes, "labels": cls})

        return self.model(imgs, targets=target_list)
