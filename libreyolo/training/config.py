"""
Training configuration for YOLOX.

Provides a dataclass-based configuration with YAML support.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Tuple, Union, List
import yaml


@dataclass
class YOLOXTrainConfig:
    """
    Configuration for YOLOX training.

    All YOLOX training parameters with sensible defaults.
    """

    # Model configuration
    size: str = "s"
    num_classes: int = 80
    pretrained: Optional[str] = None

    # Data configuration
    data: Optional[str] = None  # Path to data.yaml
    data_dir: Optional[str] = None  # Direct path to dataset
    imgsz: int = 640

    # Training parameters
    epochs: int = 300
    batch_size: int = 16
    device: str = "auto"  # "auto", "cuda", "mps", "cpu"

    # Optimizer settings (YOLOX defaults)
    optimizer: str = "sgd"  # "sgd" or "adam"
    lr: float = 0.01  # Base learning rate (will be scaled by batch size)
    momentum: float = 0.9
    weight_decay: float = 5e-4
    nesterov: bool = True

    # Learning rate schedule
    scheduler: str = "yoloxwarmcos"  # "cos", "warmcos", "yoloxwarmcos", "multistep"
    warmup_epochs: int = 5
    warmup_lr_start: float = 0.0
    no_aug_epochs: int = 15
    min_lr_ratio: float = 0.05

    # Augmentation settings (YOLOX defaults)
    mosaic_prob: float = 1.0
    mixup_prob: float = 1.0
    hsv_prob: float = 1.0
    flip_prob: float = 0.5
    degrees: float = 10.0
    translate: float = 0.1
    mosaic_scale: Tuple[float, float] = (0.1, 2.0)
    mixup_scale: Tuple[float, float] = (0.5, 1.5)
    shear: float = 2.0

    # Loss weights
    obj_loss_weight: float = 1.0
    cls_loss_weight: float = 1.0
    iou_loss_weight: float = 5.0
    l1_loss_weight: float = 1.0

    # Training features
    ema: bool = True
    ema_decay: float = 0.9998
    amp: bool = True  # Automatic mixed precision

    # Checkpointing
    save_dir: str = "runs/train"
    save_interval: int = 10  # Save checkpoint every N epochs
    eval_interval: int = 10  # Evaluate every N epochs

    # Workers
    num_workers: int = 4

    # Logging
    log_interval: int = 10  # Log every N iterations

    # Reproducibility
    seed: int = 0

    def __post_init__(self):
        """Validate configuration after initialization."""
        valid_sizes = ["nano", "tiny", "s", "m", "l", "x"]
        if self.size not in valid_sizes:
            raise ValueError(f"Invalid size '{self.size}'. Must be one of {valid_sizes}")

        valid_schedulers = ["cos", "warmcos", "yoloxwarmcos", "multistep"]
        if self.scheduler not in valid_schedulers:
            raise ValueError(
                f"Invalid scheduler '{self.scheduler}'. Must be one of {valid_schedulers}"
            )

        valid_optimizers = ["sgd", "adam", "adamw"]
        if self.optimizer not in valid_optimizers:
            raise ValueError(
                f"Invalid optimizer '{self.optimizer}'. Must be one of {valid_optimizers}"
            )

        # Ensure mosaic_scale and mixup_scale are tuples
        if isinstance(self.mosaic_scale, list):
            self.mosaic_scale = tuple(self.mosaic_scale)
        if isinstance(self.mixup_scale, list):
            self.mixup_scale = tuple(self.mixup_scale)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "YOLOXTrainConfig":
        """Load configuration from YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)

    def update(self, **kwargs) -> "YOLOXTrainConfig":
        """Create a new config with updated values."""
        data = asdict(self)
        data.update(kwargs)
        return YOLOXTrainConfig(**data)

    @property
    def input_size(self) -> Tuple[int, int]:
        """Return input size as (height, width) tuple."""
        return (self.imgsz, self.imgsz)

    @property
    def effective_lr(self) -> float:
        """Calculate effective learning rate based on batch size."""
        # YOLOX uses linear scaling: lr = base_lr * batch_size / 64
        return self.lr * self.batch_size / 64

    def __repr__(self) -> str:
        lines = ["YOLOXTrainConfig("]
        for key, value in asdict(self).items():
            lines.append(f"  {key}={value!r},")
        lines.append(")")
        return "\n".join(lines)


# Preset configurations
YOLOX_CONFIGS = {
    "nano": YOLOXTrainConfig(size="nano", imgsz=416),
    "tiny": YOLOXTrainConfig(size="tiny", imgsz=416),
    "s": YOLOXTrainConfig(size="s", imgsz=640),
    "m": YOLOXTrainConfig(size="m", imgsz=640),
    "l": YOLOXTrainConfig(size="l", imgsz=640),
    "x": YOLOXTrainConfig(size="x", imgsz=640),
}


def get_config(size: str = "s", **kwargs) -> YOLOXTrainConfig:
    """
    Get a preset configuration with optional overrides.

    Args:
        size: Model size ("nano", "tiny", "s", "m", "l", "x")
        **kwargs: Override any configuration parameter

    Returns:
        YOLOXTrainConfig instance
    """
    if size not in YOLOX_CONFIGS:
        raise ValueError(f"Unknown size '{size}'. Available: {list(YOLOX_CONFIGS.keys())}")

    config = YOLOX_CONFIGS[size]
    if kwargs:
        config = config.update(**kwargs)
    return config
