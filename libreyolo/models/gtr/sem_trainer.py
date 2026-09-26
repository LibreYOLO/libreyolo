"""GTR semantic fine-tuning through the shared semantic BaseTrainer path.

Recipe from the pinned upstream Cityscapes configs (``configs/semseg``):
AdamW at 5e-4 with a size-specific backbone LR and weight decay, no weight
decay on norm/BN/bias, quadratic warmup over 2000 iterations, six flat epochs,
cosine decay to half the LR, EMA 0.9999 with a 1000-update warmup, gradient
clipping at 0.1, FP32, 30 epochs, batch 8 and 1024px square crops.

Augmentation uses the shared ``SemanticDataset``: horizontal flip, torchvision
photometric distortion (p=0.5) and upstream's large-scale jitter, which fits
the long side to ``1024 * [1, 4]`` and random-crops the 1024px window. That is
the dataset's letterbox mode with a 1..4 scale range. Two differences remain:
padding uses the dataset's grey fill (masks still pad with ignore), and the
crop does not re-sample single-class crops (upstream cat_max_ratio 0.75).
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Dict, Type

import torch

from ...training.config import TrainConfig
from ...training.distributed import unwrap_model
from ...training.optim import build_optimizer
from ...training.trainer import BaseTrainer
from ..base.semantic_validation_loss import SemanticValidationLossMixin
from . import sem
from .scheduler import GTRScheduler

# Upstream backbone LR over the 5e-4 base LR, and weight decay, per size.
SEM_BACKBONE_LR_MULT = {"s": 0.3, "m": 0.36, "l": 0.24, "x": 0.24}
SEM_WEIGHT_DECAY = {"s": 1e-4, "m": 1e-4, "l": 1.25e-4, "x": 1.25e-4}


@dataclass(kw_only=True)
class GTRSemConfig(TrainConfig):
    optimizer: str = "adamw"
    scheduler: str = "flat_cosine"
    lr0: float = 5e-4
    weight_decay: float | None = None
    backbone_lr_mult: float | None = None
    epochs: int = 30
    batch: int = 8
    imgsz: int = sem.SEM_WINDOW
    warmup_iters: int = 2000
    warmup_epochs: float | None = None
    warmup_lr_start: float = 0.0
    flat_epochs: int = 6
    no_aug_epochs: int = 0
    min_lr_ratio: float = 0.5
    mosaic_prob: float = 0.0
    mixup_prob: float = 0.0
    ema: bool = True
    ema_decay: float = 0.9999
    ema_tau: int = 1000
    clip_max_norm: float = 0.1
    # Upstream converts BN to SyncBN; single-process runs keep plain BN, whose
    # batch statistics are noisy below upstream's global batch of 8.
    sync_bn: bool = True
    amp: bool = False
    eval_interval: int = 1
    name: str = "gtr_sem_exp"

    def __post_init__(self):
        super().__post_init__()
        if self.size not in SEM_BACKBONE_LR_MULT:
            raise ValueError(f"Unknown GTR size: {self.size!r}")
        if str(self.optimizer).lower() != "adamw":
            raise ValueError("GTR semantic training supports optimizer='adamw' only")
        if self.scheduler != "flat_cosine":
            raise ValueError(
                "GTR semantic training supports scheduler='flat_cosine' only"
            )
        if not isinstance(self.imgsz, int) or self.imgsz < 64 or self.imgsz % 32:
            raise ValueError(
                "GTR semantic training imgsz is the square crop and must be a "
                "multiple of 32 and at least 64"
            )
        if self.weight_decay is None:
            self.weight_decay = SEM_WEIGHT_DECAY[self.size]
        if self.backbone_lr_mult is None:
            self.backbone_lr_mult = SEM_BACKBONE_LR_MULT[self.size]


class GTRSemTrainer(SemanticValidationLossMixin, BaseTrainer):
    best_metric_key: str = "metrics/mIoU"

    @classmethod
    def _config_class(cls) -> Type[TrainConfig]:
        return GTRSemConfig

    def get_model_family(self) -> str:
        return "gtr"

    def get_model_tag(self) -> str:
        return f"GTR-{self.config.size}-sem"

    @property
    def effective_lr(self) -> float:
        # Upstream LRs are absolute for batch 8, not per-image rates.
        return self.config.lr0

    def _setup_optimizer(self) -> torch.optim.Optimizer:
        groups: Dict[tuple, list] = {}
        for name, param in unwrap_model(self.model).named_parameters():
            if not param.requires_grad:
                continue
            is_backbone = name.startswith("backbone.")
            # Upstream's regex: substring match on norm|bn|bias.
            no_decay = any(tag in name for tag in ("norm", "bn", "bias"))
            mult = self.config.backbone_lr_mult if is_backbone else 1.0
            decay = 0.0 if no_decay else self.config.weight_decay
            groups.setdefault((mult, decay), []).append(param)
        if not groups:
            raise ValueError("No trainable parameters remain for the GTR optimizer")
        param_groups = [
            {
                "params": params,
                "lr": self.effective_lr * mult,
                "weight_decay": decay,
                "lr_mult": mult,
            }
            for (mult, decay), params in groups.items()
        ]
        return build_optimizer(torch.optim.AdamW, param_groups, betas=(0.9, 0.999))

    def _scale_lr(self, base_lr: float, param_group: dict) -> float:
        return base_lr * float(param_group.get("lr_mult", 1.0))

    def create_scheduler(self, iters_per_epoch: int):
        return GTRScheduler(self.effective_lr, iters_per_epoch, self.config)

    def create_transforms(self):
        raise NotImplementedError(
            "Semantic training builds SemanticDataset directly "
            "(BaseTrainer._setup_semantic_data)."
        )

    def on_forward(self, imgs, targets, polygons=None) -> Dict:
        return self.model(imgs, targets=targets)

    def get_loss_components(self, outputs: Dict) -> Dict[str, float]:
        value = outputs.get("sem", 0)
        return {
            "sem": float(value.detach()) if torch.is_tensor(value) else float(value)
        }


class _PhotometricDistort:
    """torchvision RandomPhotometricDistort on an RGB HWC uint8 array."""

    def __init__(self, p=0.5):
        from torchvision.transforms import v2

        self.transform = v2.RandomPhotometricDistort(p=p)

    def __call__(self, image):
        tensor = torch.from_numpy(image.copy()).permute(2, 0, 1)
        return self.transform(tensor).permute(1, 2, 0).numpy()


def train_semantic(
    model, *, data=None, resume=False, callbacks=None, loggers=None, **kwargs
):
    """Fine-tune a GTR semantic model on a dense-mask dataset YAML."""
    if not data:
        raise ValueError("GTR semantic training requires data= (a dataset YAML)")
    valid = {field.name for field in fields(GTRSemConfig)}
    unknown = sorted(
        key for key in kwargs if key not in valid and kwargs[key] is not None
    )
    if unknown:
        raise TypeError(f"Unsupported GTR semantic training arguments: {unknown}")
    settings = {key: value for key, value in kwargs.items() if value is not None}
    if settings.get("device") == "":
        settings.pop("device")
    trainer = GTRSemTrainer(
        model=model.model,
        wrapper_model=model,
        data=data,
        size=model.size,
        num_classes=model.nb_classes,
        resume=bool(resume),
        callbacks=callbacks,
        loggers=loggers,
        **settings,
    )
    result = trainer.train()
    for key in ("best_checkpoint", "last_checkpoint"):
        path = result.get(key)
        if path and Path(path).exists():
            model.model_path = str(path)
            model._load_weights(str(path))
            break
    model.model.to(model.device).eval()
    return result


__all__ = ["GTRSemConfig", "GTRSemTrainer", "train_semantic"]
