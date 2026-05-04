"""
YOLOv9 Trainer for LibreYOLO.

Thin subclass of BaseTrainer with yolo9-specific transforms, scheduler,
and loss extraction.
"""

import torch
from typing import Dict, Type

from libreyolo.training.trainer import BaseTrainer
from libreyolo.training.config import TrainConfig, YOLO9Config
from ...training.scheduler import LinearLRScheduler, CosineAnnealingScheduler
from .transforms import YOLO9TrainTransform, YOLO9MosaicMixupDataset


class YOLO9Trainer(BaseTrainer):
    """YOLOv9-specific trainer."""

    @classmethod
    def _config_class(cls) -> Type[TrainConfig]:
        return YOLO9Config

    def get_model_family(self) -> str:
        return "yolo9"

    def get_model_tag(self) -> str:
        return f"YOLOv9-{self.config.size}"

    def create_transforms(self):
        preproc = YOLO9TrainTransform(
            max_labels=100,
            flip_prob=self.config.flip_prob,
            hsv_prob=self.config.hsv_prob,
        )
        return preproc, YOLO9MosaicMixupDataset

    def create_scheduler(self, iters_per_epoch: int):
        scheduler_name = self.config.scheduler
        if scheduler_name == "linear":
            return LinearLRScheduler(
                lr=self.effective_lr,
                iters_per_epoch=iters_per_epoch,
                total_epochs=self.config.epochs,
                warmup_epochs=self.config.warmup_epochs,
                warmup_lr_start=self.config.warmup_lr_start,
                min_lr_ratio=self.config.min_lr_ratio,
            )
        elif scheduler_name in ("cos", "warmcos"):
            return CosineAnnealingScheduler(
                lr=self.effective_lr,
                iters_per_epoch=iters_per_epoch,
                total_epochs=self.config.epochs,
                warmup_epochs=self.config.warmup_epochs,
                warmup_lr_start=self.config.warmup_lr_start,
                min_lr_ratio=self.config.min_lr_ratio,
            )
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_name}")

    def get_loss_components(self, outputs: Dict) -> Dict[str, float]:
        def _scalar(v):
            return v.item() if isinstance(v, torch.Tensor) else v

        return {
            "box": _scalar(outputs.get("box", 0)),
            "cls": _scalar(outputs.get("cls", 0)),
            "dfl": _scalar(outputs.get("dfl", 0)),
        }

    def on_forward(self, imgs: torch.Tensor, targets: torch.Tensor, polygons=None) -> Dict:
        return self.model(imgs, targets=targets)
