"""LoopDVR fine-tuning configuration dataclasses (detect / segment / pose).

Kept separate from ``trainer.py`` so importing ``LibreLoopDVR`` does not pull in
training losses or ``torch.jit.script`` helpers at package import time.
"""

from __future__ import annotations

from dataclasses import dataclass

from ...training.config import ECConfig, ECPoseConfig, ECSegConfig


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


__all__ = [
    "LoopDVRConfig",
    "LoopDVRPoseConfig",
    "LoopDVRSegConfig",
]
