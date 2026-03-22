"""
YOLOv9 Trainer for LibreYOLO.

Thin subclass of BaseTrainer with yolo9-specific transforms, scheduler,
and loss extraction.
"""

import logging
import torch
import torch.nn as nn
from typing import Dict, Type

from libreyolo.training.trainer import BaseTrainer
from libreyolo.training.config import TrainConfig, YOLO9Config
from ...training.scheduler import LinearLRScheduler, CosineAnnealingScheduler
from .transforms import YOLO9TrainTransform, YOLO9MosaicMixupDataset

logger = logging.getLogger(__name__)


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

        components = {
            "box": _scalar(outputs.get("box", 0)),
            "cls": _scalar(outputs.get("cls", 0)),
            "dfl": _scalar(outputs.get("dfl", 0)),
        }
        if "mask" in outputs:
            components["mask"] = _scalar(outputs["mask"])
        return components

    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """Setup optimizer with differential LR for segmentation models.

        When the model has a segmentation head (proto + cv4), use full LR for
        seg params and reduced LR for backbone/neck/detection head.
        """
        is_seg = hasattr(self.model, 'segmentation') and self.model.segmentation

        if not is_seg:
            return super()._setup_optimizer()

        # Separate params into seg head vs everything else
        seg_names = {'head.proto', 'head.cv4'}
        seg_params_wd, seg_params_no_wd = [], []
        det_params_wd, det_params_no_wd = [], []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            is_seg_param = any(name.startswith(s) for s in seg_names)
            is_no_wd = 'bias' in name or 'bn' in name or 'BatchNorm' in name

            if is_seg_param:
                if is_no_wd:
                    seg_params_no_wd.append(param)
                else:
                    seg_params_wd.append(param)
            else:
                if is_no_wd:
                    det_params_no_wd.append(param)
                else:
                    det_params_wd.append(param)

        lr = self.effective_lr
        # Seg head: full LR (ratio=1.0)
        # Everything else: 0.01x LR (ratio=0.01)
        backbone_ratio = 0.01

        opt_name = self.config.optimizer
        param_groups = [
            {"params": seg_params_wd, "lr": lr, "weight_decay": self.config.weight_decay, "lr_ratio": 1.0},
            {"params": seg_params_no_wd, "lr": lr, "lr_ratio": 1.0},
            {"params": det_params_wd, "lr": lr * backbone_ratio, "weight_decay": self.config.weight_decay, "lr_ratio": backbone_ratio},
            {"params": det_params_no_wd, "lr": lr * backbone_ratio, "lr_ratio": backbone_ratio},
        ]

        if opt_name == "sgd":
            optimizer = torch.optim.SGD(
                param_groups, lr=lr,
                momentum=self.config.momentum, nesterov=self.config.nesterov,
            )
        elif opt_name == "adam":
            optimizer = torch.optim.Adam(param_groups, lr=lr)
        elif opt_name == "adamw":
            optimizer = torch.optim.AdamW(param_groups, lr=lr)
        else:
            raise ValueError(f"Unknown optimizer: {opt_name}")

        logger.info(f"Optimizer: {opt_name} (differential LR for seg)")
        logger.info(f"  - seg (wd):    {len(seg_params_wd)} params, lr_ratio=1.0")
        logger.info(f"  - seg (no_wd): {len(seg_params_no_wd)} params, lr_ratio=1.0")
        logger.info(f"  - det (wd):    {len(det_params_wd)} params, lr_ratio={backbone_ratio}")
        logger.info(f"  - det (no_wd): {len(det_params_no_wd)} params, lr_ratio={backbone_ratio}")
        return optimizer

    def on_forward(self, imgs: torch.Tensor, targets: torch.Tensor, masks=None) -> Dict:
        return self.model(imgs, targets=targets, mask_targets=masks)
