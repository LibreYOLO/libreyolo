"""GTR pose fine-tuning on the shared DETRPose trainer.

The matcher, criterion (VFL + keypoint L1 + OKS with GO union matching and
contrastive denoising) and keypoint data pipeline are the EdgeCrafter ECPose
ones, which follow the same DETRPose recipe as upstream GTR. The optimizer and
schedule follow the pinned GTR ``coco_pose_finetune`` configs. Upstream's
PoseMosaic, MixUpCopyPaste and zoom-out augmentations are not reproduced; the
ECPose keypoint-aware flip, color and affine transforms are used instead.
"""

from __future__ import annotations

from dataclasses import dataclass

from ...training.config import ECPoseConfig
from ...training.scheduler import ConstantLRScheduler
from ..ec.pose_trainer import ECPoseTrainer

# Size-specific upstream values: (epochs, backbone LR multiplier, weight decay).
POSE_RECIPES = {
    "s": (92, 0.05, 1e-4),
    "m": (92, 0.05, 1e-4),
    "l": (74, 0.005, 1.25e-4),
    "x": (74, 0.005, 1.25e-4),
}


@dataclass(kw_only=True)
class GTRPoseConfig(ECPoseConfig):
    lr0: float = 5e-4
    amp: bool = False
    scheduler: str = "constant"
    # Upstream LinearWarmup(500 iterations), then a constant learning rate.
    warmup_iters: int = 500
    warmup_lr_start: float = 0.0
    no_aug_epochs: int = 0
    min_lr_ratio: float = 1.0
    name: str = "gtr_pose_exp"

    @classmethod
    def from_kwargs(cls, **kwargs):
        cfg = super().from_kwargs(**kwargs)
        size = str(cfg.size).lower()
        if size not in POSE_RECIPES:
            raise ValueError(f"Unknown GTR size: {cfg.size!r}")
        epochs, backbone_lr_mult, weight_decay = POSE_RECIPES[size]
        if "epochs" not in kwargs:
            cfg.epochs = epochs
        if "backbone_lr_mult" not in kwargs:
            cfg.backbone_lr_mult = backbone_lr_mult
        if "weight_decay" not in kwargs:
            cfg.weight_decay = weight_decay
        if cfg.warmup_iters < 0:
            raise ValueError("GTR warmup_iters must be non-negative")
        return cfg


class GTRPoseTrainer(ECPoseTrainer):
    artifact_model_families = ("gtr",)
    # lora=True freezes the recurrent ViT base and the decoder layer bases and
    # trains adapters on their Linears (libreyolo/training/lora.py).
    supports_lora = True

    @classmethod
    def _config_class(cls):
        return GTRPoseConfig

    def get_model_family(self) -> str:
        return "gtr"

    def get_model_tag(self) -> str:
        return f"GTR-Pose-{self.config.size}"

    def preserve_freeze_param(self, name, param) -> bool:
        if not getattr(self.config, "lora", False):
            return False
        from ...training.lora import is_lora_parameter_name

        return is_lora_parameter_name(name)

    def on_setup(self):
        if getattr(self.config, "lora", False):
            from ...training.lora import apply_lora_to_gtr

            apply_lora_to_gtr(self.model)
        super().on_setup()

    def create_scheduler(self, iters_per_epoch: int):
        scheduler = ConstantLRScheduler(
            lr=self.effective_lr,
            iters_per_epoch=iters_per_epoch,
            total_epochs=self.config.epochs,
            warmup_lr_start=self.config.warmup_lr_start,
        )
        total = iters_per_epoch * self.config.epochs
        scheduler.warmup_iters = min(int(self.config.warmup_iters), max(0, total - 1))
        return scheduler
