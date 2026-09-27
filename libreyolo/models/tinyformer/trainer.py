"""TinyFormerTrainer — native LibreYOLO training for TinyFormer.

TinyFormer trains exactly like DEIMv2's DINO sizes (same criterion, matcher,
Dense O2O-derived transform path, flat-cosine schedule, and DINOv3 optimizer
grouping), so this subclasses ``DEIMv2Trainer`` and only swaps the recipe
table and the family identity. The one behavioural difference: every
TinyFormer size is a DINO size, so ImageNet normalisation and the
backbone-LR split on ``backbone.dinov3.*`` apply unconditionally (DEIMv2
checks membership in its own ``DINO_SIZES``, which the "xl" code is not in).
"""

from __future__ import annotations

from typing import Type

import torch

from ...training.config import (
    TINYFORMER_SIZE_DEFAULTS,
    TinyFormerConfig,
    TrainConfig,
)
from ...training.optim import build_optimizer
from ..deimv2.trainer import DEIMv2Trainer
from ..deimv2.transforms import DEIMPassThroughDataset, DEIMTrainTransform
from .nn import normalize_size


class TinyFormerTrainer(DEIMv2Trainer):
    """Native trainer for all released TinyFormer PBM sizes."""

    def __init__(self, *args, **kwargs):
        size = normalize_size(str(kwargs.get("size", "s")))
        if size not in TINYFORMER_SIZE_DEFAULTS:
            raise ValueError(f"Unknown TinyFormer size: {size!r}")
        kwargs["size"] = size
        epochs_overridden = kwargs.get("epochs") is not None

        recipe = TINYFORMER_SIZE_DEFAULTS[size]
        for key, value in recipe.items():
            # Upstream's absolute warmup_iters is tuned for the recipe's full
            # epoch budget; on epoch override, fall back to warmup_epochs so
            # the scheduler scales to the shorter run.
            if key == "warmup_iters" and epochs_overridden:
                continue
            if kwargs.get(key) is None:
                kwargs[key] = value

        # Skip DEIMv2Trainer.__init__'s own recipe application ("xl" is not a
        # DEIMv2 size); jump straight to the DEIM trainer base.
        super(DEIMv2Trainer, self).__init__(*args, **kwargs)

    @classmethod
    def _config_class(cls) -> Type[TrainConfig]:
        return TinyFormerConfig

    def get_model_family(self) -> str:
        return "tinyformer"

    def get_model_tag(self) -> str:
        return f"TinyFormer-{self.config.size}"

    def create_transforms(self):
        preproc = DEIMTrainTransform(
            max_labels=120,
            flip_prob=self.config.flip_prob,
            imgsz=self.config.imgsz,
            imagenet_norm=True,
            sanitize_min_size=int(self.config.sanitize_min_size),
        )
        return preproc, DEIMPassThroughDataset

    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """AdamW groups: low-LR DINOv3 tower, base-LR SSA/encoder/decoder."""
        backbone_wd, backbone_no_wd, head_wd, head_no_wd = [], [], [], []

        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            is_norm_or_bias = (
                "norm" in name or ".bn." in name or "bias" in name
            )
            # Only the ViT tower gets the reduced LR; the SSA stem and the
            # proj_c* heads are freshly trained at the base LR (upstream's
            # optimizer regexes match ``.dinov3`` only).
            is_backbone_lr = name.startswith("backbone.dinov3.")
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

        return build_optimizer(torch.optim.AdamW, param_groups, betas=(0.9, 0.999))
