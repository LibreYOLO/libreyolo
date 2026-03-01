"""
YOLOv9 Trainer for LibreYOLO.

Thin subclass of BaseTrainer with v9-specific transforms, scheduler,
and loss extraction.
"""

import torch
from typing import Dict

from libreyolo.training.trainer import BaseTrainer
from ...training.scheduler import LinearLRScheduler, CosineAnnealingScheduler
from .transforms import YOLO9TrainTransform, YOLO9MosaicMixupDataset


class YOLO9Trainer(BaseTrainer):
    """YOLOv9-specific trainer."""

    DEFAULT_CFG: Dict = {
        **BaseTrainer.DEFAULT_CFG,
        # YOLOv9-specific defaults
        "momentum": 0.937,
        "scheduler": "linear",
        "warmup_epochs": 3,
        "warmup_lr_start": 0.0001,
        "no_aug_epochs": 15,
        "min_lr_ratio": 0.01,
        "degrees": 0.0,
        "shear": 0.0,
        "mosaic_scale": (0.5, 1.5),
        "mixup_prob": 0.0,
        "ema_decay": 0.9999,
        "name": "yolo9_exp",
        "workers": 8,
    }

    def get_model_family(self) -> str:
        return "yolo9"

    def get_model_tag(self) -> str:
        return f"YOLOv9-{self.cfg['size']}"

    def create_transforms(self):
        preproc = YOLO9TrainTransform(
            max_labels=100,
            flip_prob=self.cfg["flip_prob"],
            hsv_prob=self.cfg["hsv_prob"],
        )
        return preproc, YOLO9MosaicMixupDataset

    def create_scheduler(self, iters_per_epoch: int):
        scheduler_name = self.cfg["scheduler"]
        if scheduler_name == "linear":
            return LinearLRScheduler(
                lr=self.effective_lr,
                iters_per_epoch=iters_per_epoch,
                total_epochs=self.cfg["epochs"],
                warmup_epochs=self.cfg["warmup_epochs"],
                warmup_lr_start=self.cfg["warmup_lr_start"],
                min_lr_ratio=self.cfg["min_lr_ratio"],
            )
        elif scheduler_name in ("cos", "warmcos"):
            return CosineAnnealingScheduler(
                lr=self.effective_lr,
                iters_per_epoch=iters_per_epoch,
                total_epochs=self.cfg["epochs"],
                warmup_epochs=self.cfg["warmup_epochs"],
                warmup_lr_start=self.cfg["warmup_lr_start"],
                min_lr_ratio=self.cfg["min_lr_ratio"],
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

    def on_forward(self, imgs: torch.Tensor, targets: torch.Tensor) -> Dict:
        return self.model(imgs, targets=targets)
