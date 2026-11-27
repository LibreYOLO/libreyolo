"""LoopDVR fine-tuning configs, validation-loss adapter, and trainers (detect / segment / pose).

LoopDVR shares the EdgeCrafter / D-FINE / DETRPose training recipe across all
three tasks:
- Detect: AdamW + FlatCosine + EMA 0.9999 + ImageNet-normalized inputs +
  ECCriterion (MAL + L1 + GIoU + FGL + DDF + contrastive denoising).
- Segment: ECSegTrainer recipe adding point-sampled BCE + Dice mask loss on
  the deferred spatial/query mask head.
- Pose: ECPoseTrainer (DETRPose) recipe with Hungarian keypoint matching,
  VFL + L1 + OKS losses, and contrastive keypoint denoising.

All three trainers override optimizer grouping so LoopDVR's looped ViT
backbone parameters (``looped_vit.*``) receive ``backbone_lr_mult`` and
norms/biases/positional/register tokens skip weight decay.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ...training.config import ECConfig, ECPoseConfig, ECSegConfig, TrainConfig
from ...training.optim import build_optimizer
from ..base.detr_validation_loss import DETRValidationLoss
from ..ec.pose_trainer import ECPoseTrainer
from ..ec.seg_trainer import ECSegTrainer
from ..ec.trainer import ECTrainer
from .nn import LibreLoopDVRModel


@dataclass(kw_only=True)
class LoopDVRConfig(ECConfig):
    """LoopDVR detection fine-tuning defaults."""

    name: str = "loopdvr_exp"

    @classmethod
    def from_kwargs(cls, /, **kwargs):
        cfg = super().from_kwargs(**kwargs)
        size = str(cfg.size).lower()
        if size in {"l", "x"}:
            if "backbone_lr_mult" not in kwargs:
                cfg.backbone_lr_mult = 0.01
            if "weight_decay" not in kwargs:
                cfg.weight_decay = 1.25e-4
        return cfg


@dataclass(kw_only=True)
class LoopDVRSegConfig(ECSegConfig):
    """LoopDVR instance-segmentation fine-tuning defaults."""

    name: str = "loopdvr_seg_exp"


@dataclass(kw_only=True)
class LoopDVRPoseConfig(ECPoseConfig):
    """LoopDVR pose-estimation fine-tuning defaults."""

    name: str = "loopdvr_pose_exp"

    @classmethod
    def from_kwargs(cls, /, **kwargs):
        cfg = super().from_kwargs(**kwargs)
        size = str(cfg.size).lower()
        if size in {"l", "x"}:
            if "backbone_lr_mult" not in kwargs:
                cfg.backbone_lr_mult = 0.005
            if "weight_decay" not in kwargs:
                cfg.weight_decay = 1.25e-4
        return cfg


def _build_loopdvr_optimizer(
    model: nn.Module, lr: float, wd: float, bb_mult: float
) -> torch.optim.Optimizer:
    """Build AdamW with {looped_vit backbone, head} x {wd, no-wd} groups."""
    backbone_wd, backbone_no_wd, head_wd, head_no_wd = [], [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_norm_or_bias = (
            "norm" in name
            or ".bn." in name
            or "bias" in name
            or "lab.scale" in name
            or "pos_embed" in name
            or "register_token" in name
        )
        is_backbone = name.startswith(("looped_vit.", "backbone."))
        if is_backbone and is_norm_or_bias:
            backbone_no_wd.append(p)
        elif is_backbone:
            backbone_wd.append(p)
        elif is_norm_or_bias:
            head_no_wd.append(p)
        else:
            head_wd.append(p)

    groups = []
    if head_wd:
        groups.append({"params": head_wd, "lr": lr, "weight_decay": wd, "lr_mult": 1.0})
    if head_no_wd:
        groups.append(
            {"params": head_no_wd, "lr": lr, "weight_decay": 0.0, "lr_mult": 1.0}
        )
    if backbone_wd:
        groups.append(
            {
                "params": backbone_wd,
                "lr": lr * bb_mult,
                "weight_decay": wd,
                "lr_mult": bb_mult,
            }
        )
    if backbone_no_wd:
        groups.append(
            {
                "params": backbone_no_wd,
                "lr": lr * bb_mult,
                "weight_decay": 0.0,
                "lr_mult": bb_mult,
            }
        )
    return build_optimizer(torch.optim.AdamW, groups, betas=(0.9, 0.999))


class LoopDVRValidationLoss(DETRValidationLoss):
    """Evaluate LoopDVR detection criterion from the validation forward pass."""

    family = "LoopDVR"

    def __init__(self, model: nn.Module, criterion: nn.Module) -> None:
        if type(model) is not LibreLoopDVRModel:
            raise TypeError("LoopDVR validation loss supports the detect model only")
        super().__init__(model, criterion)


def _loopdvr_freeze_groups(model: nn.Module) -> list[tuple[str, nn.Module]]:
    """Return semantic freeze groups so ``freeze='backbone'`` and ``freeze='looped_vit'`` both work."""
    groups: list[tuple[str, nn.Module]] = []
    looped_vit = getattr(model, "looped_vit", None)
    if looped_vit is not None:
        groups.append(("backbone", looped_vit))
    encoder = getattr(model, "encoder", None)
    if encoder is not None:
        groups.append(("encoder", encoder))
    decoder = getattr(model, "decoder", None)
    if decoder is not None:
        groups.append(("decoder", decoder))
    return groups


class LoopDVRTrainer(ECTrainer):
    """Trainer for LoopDVR detection models (LoopDet)."""

    supports_lora = False

    @classmethod
    def _config_class(cls) -> type[TrainConfig]:
        return LoopDVRConfig

    def get_model_family(self) -> str:
        return "loopdvr"

    def get_model_tag(self) -> str:
        return f"LoopDVR-{self.config.size}"

    def get_freeze_groups(self):
        return _loopdvr_freeze_groups(self.model) or super().get_freeze_groups()

    def on_setup(self):
        if getattr(self.config, "lora", False):
            raise ValueError(
                "LoopDVR training does not support lora=True yet. "
                "Use freeze='backbone' or freeze='looped_vit' for "
                "parameter-efficient fine-tuning."
            )
        self.criterion = self.build_criterion()

    def build_validation_loss_adapter(self, model: nn.Module):
        return LoopDVRValidationLoss(
            model, self.build_criterion(distributed_normalize=False)
        )

    def _setup_optimizer(self) -> torch.optim.Optimizer:
        return _build_loopdvr_optimizer(
            self.model,
            lr=self.effective_lr,
            wd=self.config.weight_decay,
            bb_mult=float(getattr(self.config, "backbone_lr_mult", 1.0)),
        )


class LoopDVRSegTrainer(ECSegTrainer):
    """Trainer for LoopDVR instance-segmentation models (LoopSeg)."""

    artifact_model_families = ("loopdvr",)
    supports_lora = False

    @classmethod
    def _config_class(cls) -> type[TrainConfig]:
        return LoopDVRSegConfig

    def get_model_family(self) -> str:
        return "loopdvr"

    def get_model_tag(self) -> str:
        return f"LoopDVR-Seg-{self.config.size}"

    def get_freeze_groups(self):
        return _loopdvr_freeze_groups(self.model) or super().get_freeze_groups()

    def on_setup(self):
        if getattr(self.config, "lora", False):
            raise ValueError(
                "LoopDVR training does not support lora=True yet. "
                "Use freeze='backbone' or freeze='looped_vit' for "
                "parameter-efficient fine-tuning."
            )
        super().on_setup()

    def _setup_optimizer(self) -> torch.optim.Optimizer:
        return _build_loopdvr_optimizer(
            self.model,
            lr=self.effective_lr,
            wd=self.config.weight_decay,
            bb_mult=float(getattr(self.config, "backbone_lr_mult", 1.0)),
        )


class LoopDVRPoseTrainer(ECPoseTrainer):
    """Trainer for LoopDVR pose-estimation models (LoopPose)."""

    artifact_model_families = ("loopdvr",)
    supports_lora = False

    @classmethod
    def _config_class(cls) -> type[TrainConfig]:
        return LoopDVRPoseConfig

    def get_model_family(self) -> str:
        return "loopdvr"

    def get_model_tag(self) -> str:
        return f"LoopDVR-Pose-{self.config.size}"

    def get_freeze_groups(self):
        return _loopdvr_freeze_groups(self.model) or super().get_freeze_groups()

    def on_setup(self):
        if getattr(self.config, "lora", False):
            raise ValueError(
                "LoopDVR training does not support lora=True yet. "
                "Use freeze='backbone' or freeze='looped_vit' for "
                "parameter-efficient fine-tuning."
            )
        super().on_setup()

    def _setup_optimizer(self) -> torch.optim.Optimizer:
        return _build_loopdvr_optimizer(
            self.model,
            lr=self.effective_lr,
            wd=self.config.weight_decay,
            bb_mult=float(getattr(self.config, "backbone_lr_mult", 1.0)),
        )


__all__ = [
    "LoopDVRConfig",
    "LoopDVRPoseConfig",
    "LoopDVRPoseTrainer",
    "LoopDVRSegConfig",
    "LoopDVRSegTrainer",
    "LoopDVRTrainer",
    "LoopDVRValidationLoss",
]
