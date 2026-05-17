"""Training configuration for native LibreRFDETR."""

from dataclasses import dataclass

from libreyolo.training.config import TrainConfig


@dataclass(kw_only=True)
class RFDETRConfig(TrainConfig):
    """CLI-visible RF-DETR fine-tuning defaults."""

    epochs: int = 100
    batch: int = 4
    lr0: float = 1e-4
    device: str = "auto"

    workers: int = 2
    weight_decay: float = 1e-4
    eval_interval: int = 1
    warmup_epochs: int = 0
    warmup_lr_start: float = 1e-6
    no_aug_epochs: int = 0
    min_lr_ratio: float = 0.0

    ema: bool = True
    ema_decay: float = 0.993
    seed: int | None = None

    patience: int = 0
    optimizer: str = "adamw"
    scheduler: str = "flat_cosine"
    mosaic_prob: float = 0.0
    mixup_prob: float = 0.0
    hsv_prob: float = 0.0
    degrees: float = 0.0
    translate: float = 0.0
    shear: float = 0.0
    amp: bool = True
    backbone_lr_mult: float = 0.1
    clip_max_norm: float = 0.1

    name: str = "rfdetr_exp"
