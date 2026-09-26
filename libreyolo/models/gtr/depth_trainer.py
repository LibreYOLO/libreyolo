"""GTR depth fine-tuning with the upstream SILog objective.

Recipe from Intellindust-AI-Lab/GTR (MIT), revision
782e737efe2e6437ac537fbdcee089673d3376c1, ``configs/depth/gtrdepth_base.yml``:
AdamW at 2e-4 with the backbone at 0.1x, weight decay 1e-4 except norms and
biases, quadratic warmup, flat phase then cosine decay to 0.1x, EMA 0.999,
gradient clipping at 0.1, FP32. The shared depth dataset supplies ``[0, 1]``
RGB and dense depth targets (invalid pixels are 0); train on metres to keep
the checkpoint's metric scale.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch

from ...training.config import TrainConfig
from ...training.optim import build_optimizer
from ...training.trainer import BaseTrainer
from .scheduler import GTRScheduler


@dataclass(kw_only=True)
class GTRDepthConfig(TrainConfig):
    epochs: int = 60
    batch: int = 8
    imgsz: int = 640
    optimizer: str = "adamw"
    lr0: float = 2e-4
    weight_decay: float = 1e-4
    backbone_lr_mult: float = 0.1
    scheduler: str = "flat_cosine"
    warmup_iters: int = 500
    warmup_epochs: Optional[float] = None
    warmup_lr_start: float = 0.0
    flat_epochs: int = 15
    no_aug_epochs: int = 0
    min_lr_ratio: float = 0.1
    ema: bool = True
    ema_decay: float = 0.999
    amp: bool = False
    clip_max_norm: float = 0.1
    silog_lambda: float = 0.5
    eval_interval: int = 1
    name: str = "gtr_depth_exp"

    def __post_init__(self):
        super().__post_init__()
        if str(self.optimizer).lower() != "adamw":
            raise ValueError("GTR depth currently supports optimizer='adamw' only")
        if self.lora:
            raise ValueError("LoRA is not supported for depth models (ADR 0006).")


def silog_loss(pred, gt, lambd=0.5):
    """Scale-invariant log loss over valid (positive, finite) target pixels."""
    valid = (gt > 0) & torch.isfinite(gt)
    if not valid.any():
        return pred.sum() * 0.0
    diff = torch.log(gt[valid]) - torch.log(pred[valid].clamp_min(1e-6))
    return torch.sqrt(
        (diff.pow(2).mean() - lambd * diff.mean().pow(2)).clamp_min(1e-12)
    )


class GTRDepthTrainer(BaseTrainer):
    best_metric_key = "metrics/delta1"

    @classmethod
    def _config_class(cls):
        return GTRDepthConfig

    def get_model_family(self):
        return "gtr"

    def get_model_tag(self):
        return f"GTR-{self.config.size}-depth"

    def create_transforms(self):
        return None, None

    def create_scheduler(self, iters_per_epoch):
        return GTRScheduler(self.effective_lr, iters_per_epoch, self.config)

    def _scale_lr(self, base_lr, param_group):
        return base_lr * param_group.get("lr_mult", 1.0)

    def _setup_optimizer(self):
        groups = {}
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            backbone = name.startswith("backbone.")
            no_decay = "norm" in name or ".bn." in name or "bias" in name
            groups.setdefault((backbone, no_decay), []).append(param)
        lr, mult = self.effective_lr, float(self.config.backbone_lr_mult)
        param_groups = [
            {
                "params": params,
                "lr": lr * (mult if backbone else 1.0),
                "lr_mult": mult if backbone else 1.0,
                "weight_decay": 0.0 if no_decay else self.config.weight_decay,
            }
            for (backbone, no_decay), params in sorted(groups.items())
        ]
        if not param_groups:
            raise ValueError("No trainable parameters remain after layer freezing")
        return build_optimizer(torch.optim.AdamW, param_groups, betas=(0.9, 0.999))

    def on_forward(self, imgs, targets, polygons=None) -> Dict:
        del polygons
        # The graph emits inverse depth (the LibreYOLO contract); SILog runs on
        # its exact reciprocal, i.e. upstream's metre output.
        depth = self.model(imgs)[:, 0].reciprocal()
        if depth.shape[-2:] != targets.shape[-2:]:
            depth = torch.nn.functional.interpolate(
                depth[:, None],
                size=targets.shape[-2:],
                mode="bilinear",
                align_corners=True,
            )[:, 0]
        loss = silog_loss(depth, targets, float(self.config.silog_lambda))
        return {"total_loss": loss, "loss_silog": loss.detach()}

    def get_loss_components(self, outputs):
        return {"silog": float(outputs.get("loss_silog", outputs["total_loss"]))}
