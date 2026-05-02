"""DEIMv2Trainer — native LibreYOLO training for DEIMv2.

This is a flat training port, not a wrapper around upstream ``DetSolver``.
It reuses DEIMTrainer's BaseTrainer integration shape while keeping DEIMv2's
size-specific recipes, DINOv3 optimizer grouping, epoch-aware matcher switch,
and optional GO-union criterion behavior.
"""

from __future__ import annotations

import math
from typing import Dict, Type

import torch

from ...training.config import (
    DEIMV2_SIZE_DEFAULTS,
    DEIMv2Config,
    TrainConfig,
)
from ..deim.trainer import DEIMTrainer
from .loss import DEIMv2Criterion
from .matcher import HungarianMatcher
from .nn import DINO_SIZES, normalize_size
from .transforms import DEIMPassThroughDataset, DEIMTrainTransform


class DEIMv2FlatCosineScheduler:
    """DEIMv2 flat-cosine scheduler using upstream's iteration warmup shape."""

    def __init__(
        self,
        lr: float,
        iters_per_epoch: int,
        total_epochs: int,
        warmup_iters: int = 2000,
        warmup_lr_start: float = 0.0,
        flat_epochs: int | None = None,
        no_aug_epochs: int = 12,
        min_lr_ratio: float = 0.5,
    ):
        self.lr = lr
        self.total_iters = iters_per_epoch * total_epochs
        self.warmup_iters = int(warmup_iters)
        self.warmup_lr_start = warmup_lr_start
        self.flat_iters = (
            int(iters_per_epoch * flat_epochs)
            if flat_epochs is not None
            else self.total_iters - int(iters_per_epoch * no_aug_epochs)
        )
        self.no_aug_iters = int(iters_per_epoch * no_aug_epochs)
        self.min_lr = lr * min_lr_ratio

    def update_lr(self, iters: int) -> float:
        if self.warmup_iters > 0 and iters <= self.warmup_iters:
            ratio = iters / float(self.warmup_iters)
            return self.warmup_lr_start + (self.lr - self.warmup_lr_start) * (
                ratio**2
            )
        if iters <= self.flat_iters:
            return self.lr
        if iters >= self.total_iters - self.no_aug_iters:
            return self.min_lr

        denom = max(1, self.total_iters - self.flat_iters - self.no_aug_iters)
        progress = (iters - self.flat_iters) / denom
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + (self.lr - self.min_lr) * cosine_decay


class DEIMv2Trainer(DEIMTrainer):
    """Native trainer for all released DEIMv2 sizes."""

    def __init__(self, *args, **kwargs):
        size = normalize_size(str(kwargs.get("size", "s")))
        if size not in DEIMV2_SIZE_DEFAULTS:
            raise ValueError(f"Unknown DEIMv2 size: {size!r}")
        kwargs["size"] = size
        epochs_overridden = kwargs.get("epochs") is not None

        recipe = DEIMV2_SIZE_DEFAULTS[size]
        for key, value in recipe.items():
            if key == "warmup_iters" and epochs_overridden:
                continue
            if kwargs.get(key) is None:
                kwargs[key] = value

        super().__init__(*args, **kwargs)

    @classmethod
    def _config_class(cls) -> Type[TrainConfig]:
        return DEIMv2Config

    def get_model_family(self) -> str:
        return "deimv2"

    def get_model_tag(self) -> str:
        return f"DEIMv2-{self.config.size}"

    @property
    def effective_lr(self) -> float:
        """DEIMv2 YAML recipes use absolute AdamW learning rates."""
        return self.config.lr0

    def create_transforms(self):
        preproc = DEIMTrainTransform(
            max_labels=120,
            flip_prob=self.config.flip_prob,
            imgsz=self.config.imgsz,
            imagenet_norm=self.config.size in DINO_SIZES,
            sanitize_min_size=int(self.config.sanitize_min_size),
        )
        return preproc, DEIMPassThroughDataset

    def create_scheduler(self, iters_per_epoch: int):
        return DEIMv2FlatCosineScheduler(
            lr=self.effective_lr,
            iters_per_epoch=iters_per_epoch,
            total_epochs=self.config.epochs,
            warmup_iters=(
                int(self.config.warmup_iters)
                if self.config.warmup_iters is not None
                else int(self.config.warmup_epochs * iters_per_epoch)
            ),
            warmup_lr_start=self.config.warmup_lr_start,
            flat_epochs=self.config.flat_epochs,
            no_aug_epochs=self.config.no_aug_epochs,
            min_lr_ratio=self.config.min_lr_ratio,
        )

    def on_setup(self):
        decoder_reg_max = getattr(getattr(self.model, "decoder", None), "reg_max", None)
        if decoder_reg_max is not None and int(self.config.reg_max) != int(
            decoder_reg_max
        ):
            raise ValueError(
                "DEIMv2 reg_max must match the fixed decoder configuration "
                f"({decoder_reg_max}); got {self.config.reg_max}."
            )

        matcher = HungarianMatcher(
            weight_dict={"cost_class": 2.0, "cost_bbox": 5.0, "cost_giou": 2.0},
            use_focal_loss=True,
            alpha=0.25,
            gamma=2.0,
            change_matcher=bool(self.config.change_matcher),
            iou_order_alpha=float(self.config.iou_order_alpha or 1.0),
            matcher_change_epoch=int(self.config.matcher_change_epoch or 10000),
        )
        self.criterion = DEIMv2Criterion(
            matcher=matcher,
            weight_dict={
                "loss_mal": 1.0,
                "loss_bbox": 5.0,
                "loss_giou": 2.0,
                "loss_fgl": 0.15,
                "loss_ddf": 1.5,
            },
            losses=list(self.config.losses or ("mal", "boxes", "local")),
            alpha=0.75,
            gamma=1.5,
            num_classes=self.config.num_classes,
            reg_max=self.config.reg_max,
            use_uni_set=bool(self.config.use_uni_set),
        ).to(self.device)

    def _compute_criterion_losses(self, outputs: Dict, target_list) -> Dict:
        return self.criterion(outputs, target_list, epoch=self.current_epoch)

    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """AdamW groups matching DEIMv2's HGNetv2/DINOv3 recipes."""
        backbone_wd, backbone_no_wd, head_wd, head_no_wd = [], [], [], []
        dino_backbone = self.config.size in DINO_SIZES

        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            is_norm_or_bias = (
                "norm" in name
                or ".bn." in name
                or "bias" in name
                or "lab.scale" in name
            )
            is_backbone_lr = (
                name.startswith("backbone.dinov3.")
                if dino_backbone
                else name.startswith("backbone.")
            )
            if is_backbone_lr and is_norm_or_bias:
                backbone_no_wd.append(p)
            elif is_backbone_lr:
                backbone_wd.append(p)
            elif is_norm_or_bias:
                head_no_wd.append(p)
            else:
                head_wd.append(p)

        lr = self.effective_lr
        wd = self.config.weight_decay
        bb_mult = float(
            self.config.backbone_lr_mult
            if self.config.backbone_lr_mult is not None
            else 1.0
        )

        param_groups = []
        if head_wd:
            param_groups.append(
                {"params": head_wd, "lr": lr, "weight_decay": wd, "lr_mult": 1.0}
            )
        if head_no_wd:
            param_groups.append(
                {"params": head_no_wd, "lr": lr, "weight_decay": 0.0, "lr_mult": 1.0}
            )
        if backbone_wd:
            param_groups.append(
                {
                    "params": backbone_wd,
                    "lr": lr * bb_mult,
                    "weight_decay": wd,
                    "lr_mult": bb_mult,
                }
            )
        if backbone_no_wd:
            param_groups.append(
                {
                    "params": backbone_no_wd,
                    "lr": lr * bb_mult,
                    "weight_decay": 0.0,
                    "lr_mult": bb_mult,
                }
            )

        return torch.optim.AdamW(param_groups, betas=(0.9, 0.999))

    def on_forward(self, imgs: torch.Tensor, targets: torch.Tensor) -> Dict:
        target_list = self._targets_to_detr(imgs, targets)
        outputs = self.model(imgs, targets=target_list)
        losses = self._compute_criterion_losses(outputs, target_list)
        return self._format_loss_outputs(losses)
