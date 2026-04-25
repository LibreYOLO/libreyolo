"""Training configuration for RT-DETR.

Provides a dataclass-based configuration with RT-DETR-specific defaults.
"""

from dataclasses import dataclass
from typing import Tuple

from libreyolo.training.config import TrainConfig


@dataclass(kw_only=True)
class RTDETRConfig(TrainConfig):
    """Training configuration for RT-DETR models."""

    epochs: int = 72
    batch: int = 4
    optimizer: str = "adamw"
    lr0: float = 1e-4
    scheduler: str = "linear"

    # RT-DETR specific optimizer settings
    lr_backbone: float = (
        0.00001  # Separate backbone learning rate (10x lower than base)
    )
    betas: Tuple[float, float] = (0.9, 0.999)  # AdamW betas

    # Optimizer overrides (RT-DETR uses lighter regularisation than YOLO)
    weight_decay: float = 1e-4

    # Scheduler overrides (RT-DETR uses no warmup / no-aug tail by default)
    warmup_epochs: int = 0
    no_aug_epochs: int = 0

    # Augmentation overrides (RT-DETR uses milder augmentation than YOLOX)
    mosaic_prob: float = 0.5
    mixup_prob: float = 0.0
    hsv_prob: float = 0.1
    degrees: float = 0.0
    shear: float = 0.0

    # Evaluation overrides (RT-DETR evaluates every epoch)
    eval_interval: int = 1

    # Default name for RTDETR experiments
    name: str = "rtdetr_exp"
