"""Training configuration dataclasses for LibreYOLO."""

import logging
import warnings
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Tuple, Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class TrainConfig:
    """Base training configuration. Subclasses override defaults per model family."""

    # Model
    size: str = "s"
    num_classes: int = 80

    # Data
    data: Optional[str] = None
    data_dir: Optional[str] = None
    imgsz: int = 640

    # Training
    epochs: int = 300
    batch: int = 16
    device: str = "auto"

    # Optimizer
    optimizer: str = "sgd"
    lr0: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 5e-4
    nesterov: bool = True

    # Scheduler
    scheduler: str = "yoloxwarmcos"
    warmup_epochs: int = 5
    warmup_lr_start: float = 0.0
    no_aug_epochs: int = 15
    min_lr_ratio: float = 0.05

    # Augmentation
    mosaic_prob: float = 1.0
    mixup_prob: float = 1.0
    hsv_prob: float = 1.0
    flip_prob: float = 0.5
    degrees: float = 10.0
    translate: float = 0.1
    mosaic_scale: Tuple[float, float] = (0.1, 2.0)
    mixup_scale: Tuple[float, float] = (0.5, 1.5)
    shear: float = 2.0

    # Training features
    ema: bool = True
    ema_decay: float = 0.9998
    amp: bool = True

    # Checkpointing / output
    project: str = "runs/train"
    name: str = "exp"
    exist_ok: bool = False
    save_period: int = 10
    eval_interval: int = 10

    # System
    workers: int = 4
    patience: int = 50
    resume: bool = False
    log_interval: int = 10
    seed: int = 0
    allow_download_scripts: bool = False

    @classmethod
    def from_kwargs(cls, **kwargs):
        """Construct config, warning on unknown keys."""
        valid = {f.name for f in fields(cls)}
        unknown = set(kwargs) - valid
        if unknown:
            warnings.warn(
                f"Unknown training config keys (ignored): {sorted(unknown)}",
                stacklevel=2,
            )
        filtered = {k: v for k, v in kwargs.items() if k in valid}
        return cls(**filtered)

    def to_dict(self) -> dict:
        """Convert to dict with tuples converted to lists for YAML/checkpoint."""
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, tuple):
                d[k] = list(v)
        return d

    def to_yaml(self, path) -> None:
        """Serialize config to a YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)


@dataclass(kw_only=True)
class YOLOXConfig(TrainConfig):
    """YOLOX-specific training defaults."""

    momentum: float = 0.9
    warmup_epochs: int = 5
    warmup_lr_start: float = 0.0
    no_aug_epochs: int = 15
    min_lr_ratio: float = 0.05
    degrees: float = 10.0
    shear: float = 2.0
    mosaic_scale: Tuple[float, float] = (0.1, 2.0)
    mixup_prob: float = 1.0
    ema_decay: float = 0.9998
    name: str = "exp"


@dataclass(kw_only=True)
class YOLO9Config(TrainConfig):
    """YOLOv9-specific training defaults."""

    momentum: float = 0.937
    scheduler: str = "linear"
    warmup_epochs: int = 3
    warmup_lr_start: float = 0.0001
    no_aug_epochs: int = 15
    min_lr_ratio: float = 0.01
    degrees: float = 0.0
    shear: float = 0.0
    mosaic_scale: Tuple[float, float] = (0.5, 1.5)
    mixup_prob: float = 0.0
    ema_decay: float = 0.9999
    name: str = "yolo9_exp"
    workers: int = 8
