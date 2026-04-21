"""YOLO-NAS trainer for native LibreYOLO training."""

from __future__ import annotations

from typing import Dict, Type

import torch

from ...training.config import TrainConfig, YOLONASConfig
from ...training.scheduler import CosineAnnealingScheduler
from ...training.trainer import BaseTrainer
from .loss import PPYoloELoss
from .transforms import YOLONASAffineMixupDataset, YOLONASTrainTransform


class YOLONASTrainer(BaseTrainer):
    @classmethod
    def _config_class(cls) -> Type[TrainConfig]:
        return YOLONASConfig

    def get_model_family(self) -> str:
        return "yolonas"

    def get_model_tag(self) -> str:
        return f"YOLO-NAS-{self.config.size}"

    def create_transforms(self):
        preproc = YOLONASTrainTransform(
            max_labels=100,
            flip_prob=self.config.flip_prob,
            hsv_prob=self.config.hsv_prob,
        )
        return preproc, YOLONASAffineMixupDataset

    def create_scheduler(self, iters_per_epoch: int):
        return CosineAnnealingScheduler(
            lr=self.effective_lr,
            iters_per_epoch=iters_per_epoch,
            total_epochs=self.config.epochs,
            warmup_epochs=self.config.warmup_epochs,
            warmup_lr_start=self.config.warmup_lr_start,
            min_lr_ratio=self.config.min_lr_ratio,
        )

    def on_setup(self):
        self.loss_fn = PPYoloELoss(
            num_classes=self.config.num_classes,
            use_static_assigner=False,
            use_varifocal_loss=True,
        ).to(self.device)

    def get_loss_components(self, outputs: Dict) -> Dict[str, float]:
        def _scalar(v):
            return v.item() if isinstance(v, torch.Tensor) else v

        return {
            "cls": _scalar(outputs.get("cls", 0)),
            "iou": _scalar(outputs.get("iou", 0)),
            "dfl": _scalar(outputs.get("dfl", 0)),
        }

    def on_forward(self, imgs: torch.Tensor, targets: torch.Tensor) -> Dict:
        model_outputs = self.model(imgs)
        total_loss, log_losses = self.loss_fn(model_outputs, targets)
        return {
            "total_loss": total_loss,
            "cls": log_losses[0],
            "iou": log_losses[1],
            "dfl": log_losses[2],
        }
