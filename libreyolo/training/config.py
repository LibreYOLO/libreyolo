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


@dataclass(kw_only=True)
class DFINEConfig(TrainConfig):
    """D-FINE-specific training defaults.

    Inference matches upstream byte-for-byte; training is a v1 cut: AdamW with
    no-wd on norms/biases, flat LR with warmup + cosine tail, hflip-only aug,
    no mosaic/mixup. AMP off by default — D-FINE's decoder clamps activations
    to ±65504 (FP16 max) which strongly suggests FP32 is required.
    """

    optimizer: str = "adamw"
    lr0: float = 2e-4
    weight_decay: float = 1e-4

    scheduler: str = "flat_cosine"
    warmup_epochs: int = 2
    warmup_lr_start: float = 1e-6
    no_aug_epochs: int = 4
    min_lr_ratio: float = 0.05

    # No mosaic / no mixup / no color or geometric aug for v1.
    mosaic_prob: float = 0.0
    mixup_prob: float = 0.0
    hsv_prob: float = 0.0
    flip_prob: float = 0.5
    degrees: float = 0.0
    translate: float = 0.0
    shear: float = 0.0

    ema: bool = True
    ema_decay: float = 0.9999
    ema_restart_decay: float = 0.9999

    # D-FINE-specific training knobs (paper-faithful fine-tune defaults).
    backbone_lr_mult: float = 0.5  # upstream's fine-tune recipe uses 0.5×
    clip_max_norm: float = 0.1  # upstream default; 0 disables clipping
    multi_scale: bool = True  # per-batch random resize via DFINEMultiScaleCollate
    aug_stop_epoch_ratio: float = 0.85  # disable strong augs at epoch * ratio

    amp: bool = False
    epochs: int = 132
    name: str = "dfine_exp"


@dataclass(kw_only=True)
class DEIMConfig(TrainConfig):
    """DEIM-D-FINE fine-tuning defaults.

    DEIM keeps the D-FINE HGNetv2 architecture and replaces the classification
    objective with MAL from the Dense O2O recipe. These defaults are for
    practical LibreYOLO fine-tuning, not reproducing DEIM's full COCO training
    recipe. The upstream Mosaic/MixUp schedule is intentionally left for the
    shared augmentation refactor.
    """

    optimizer: str = "adamw"
    lr0: float = 4e-4
    weight_decay: float = 1e-4

    scheduler: str = "flat_cosine"
    warmup_epochs: int = 2
    warmup_lr_start: float = 1e-6
    no_aug_epochs: int = 12
    min_lr_ratio: float = 1.0

    mosaic_prob: float = 0.0
    mixup_prob: float = 0.0
    hsv_prob: float = 0.0
    flip_prob: float = 0.5
    degrees: float = 10.0
    translate: float = 0.1
    mosaic_scale: Tuple[float, float] = (0.5, 1.5)
    mixup_scale: Tuple[float, float] = (0.5, 1.5)
    shear: float = 0.0

    ema: bool = True
    ema_decay: float = 0.9999
    ema_restart_decay: float = 0.9999

    backbone_lr_mult: Optional[float] = None
    clip_max_norm: float = 0.1
    multi_scale: bool = True
    aug_stop_epoch_ratio: float = 0.91

    amp: bool = False
    epochs: int = 132
    name: str = "deim_exp"


@dataclass(kw_only=True)
class ECDetConfig(TrainConfig):
    """ECDet-specific training defaults (experimental).

    Fine-tune defaults follow upstream EdgeCrafter's published recipe (S/M):
    AdamW with backbone-LR multiplier 0.05 (≈2.5e-5 vs head 5e-4), no-decay
    on norms/biases, FlatCosine schedule with quadratic warmup, EMA 0.9999,
    Mosaic+Mixup until ~mid-training, all strong augs disabled past
    ``stop_epoch``. Loss = MAL + L1 + GIoU + FGL + DDF.

    Training has NOT been validated on a real fine-tune run — ship as
    experimental.
    """

    optimizer: str = "adamw"
    lr0: float = 5e-4
    weight_decay: float = 1e-4

    scheduler: str = "flat_cosine"
    warmup_epochs: int = 2
    warmup_lr_start: float = 1e-6
    no_aug_epochs: int = 4
    min_lr_ratio: float = 0.5  # ECDet's lr_gamma in upstream

    mosaic_prob: float = 0.75
    mixup_prob: float = 0.75
    hsv_prob: float = 0.5
    flip_prob: float = 0.5
    degrees: float = 10.0
    translate: float = 0.1
    mosaic_scale: Tuple[float, float] = (0.5, 1.5)
    mixup_scale: Tuple[float, float] = (0.5, 1.5)
    shear: float = 0.0

    ema: bool = True
    ema_decay: float = 0.9999
    ema_restart_decay: float = 0.9999

    # ECDet-specific knobs.
    backbone_lr_mult: float = 0.05  # 2.5e-5 / 5e-4 ≈ 0.05 for S/M; L/X use 0.01
    clip_max_norm: float = 0.1
    multi_scale: bool = (
        False  # upstream uses fixed 640; multi-scale not in their config
    )
    aug_stop_epoch_ratio: float = 0.97  # stop_epoch=72 with epochs=74 → 72/74

    amp: bool = True
    epochs: int = 74
    name: str = "ecdet_exp"


@dataclass(kw_only=True)
class YOLONASConfig(TrainConfig):
    """YOLO-NAS-specific training defaults."""

    optimizer: str = "adamw"
    lr0: float = 5e-4
    momentum: float = 0.9
    weight_decay: float = 1e-5
    scheduler: str = "cos"
    warmup_epochs: int = 1
    warmup_lr_start: float = 1e-6
    no_aug_epochs: int = 0
    min_lr_ratio: float = 0.1
    mosaic_prob: float = 0.0
    mixup_prob: float = 0.5
    hsv_prob: float = 0.5
    flip_prob: float = 0.5
    degrees: float = 0.0
    translate: float = 0.25
    mosaic_scale: Tuple[float, float] = (0.5, 1.5)
    mixup_scale: Tuple[float, float] = (0.5, 1.5)
    shear: float = 0.0
    ema_decay: float = 0.9997
    amp: bool = False
    name: str = "yolonas_exp"
