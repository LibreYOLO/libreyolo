"""
Training configuration for YOLOv9.

Provides a dataclass-based configuration with v9-specific defaults.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Tuple, Union
import yaml


@dataclass
class V9TrainConfig:
    """
    Configuration for YOLOv9 training.

    All YOLOv9 training parameters with defaults from the official YOLO repo.
    """

    # Model configuration
    size: str = "c"  # "t", "s", "m", "c"
    num_classes: int = 80
    reg_max: int = 16
    pretrained: Optional[str] = None

    # Data configuration
    data: Optional[str] = None  # Path to data.yaml
    data_dir: Optional[str] = None  # Direct path to dataset
    imgsz: int = 640

    # Training parameters
    epochs: int = 300
    batch: int = 16
    device: str = "auto"  # "auto", "cuda", "mps", "cpu"

    # Optimizer settings (YOLOv9 defaults from YOLO repo)
    optimizer: str = "sgd"  # "sgd", "adam", "adamw"
    lr0: float = 0.01  # Base learning rate
    momentum: float = 0.937  # Higher momentum than YOLOX
    weight_decay: float = 0.0005
    nesterov: bool = True

    # Learning rate schedule (LinearLR with warmup - v9 default)
    scheduler: str = "linear"  # "linear", "cos", "warmcos"
    warmup_epochs: int = 3
    warmup_lr_start: float = 0.0001
    no_aug_epochs: int = 15
    min_lr_ratio: float = 0.01  # Final LR = lr0 * min_lr_ratio

    # Loss weights (v9 defaults)
    box_weight: float = 7.5  # CIoU loss weight
    dfl_weight: float = 1.5  # Distribution Focal Loss weight
    cls_weight: float = 0.5  # BCE classification loss weight

    # Task Aligned Assignment (TAL) parameters
    tal_topk: int = 10
    tal_iou_factor: float = 6.0
    tal_cls_factor: float = 0.5

    # Augmentation settings (v9 defaults - more conservative than YOLOX)
    mosaic_prob: float = 1.0
    mixup_prob: float = 0.0  # Disabled by default in v9
    hsv_prob: float = 1.0
    flip_prob: float = 0.5
    degrees: float = 0.0  # No rotation by default
    translate: float = 0.1
    mosaic_scale: Tuple[float, float] = (0.5, 1.5)
    mixup_scale: Tuple[float, float] = (0.5, 1.5)
    shear: float = 0.0  # No shear by default

    # Training features
    ema: bool = True
    ema_decay: float = 0.9999  # Higher than YOLOX
    amp: bool = True  # Automatic mixed precision

    # Checkpointing
    project: str = "runs/train"
    name: str = "v9_exp"
    exist_ok: bool = False
    save_period: int = 10  # Save checkpoint every N epochs
    eval_interval: int = 10  # Evaluate every N epochs

    # Workers
    workers: int = 8

    # Early stopping
    patience: int = 50

    # Resume
    resume: bool = False

    # Logging
    log_interval: int = 10  # Log every N iterations

    # Reproducibility
    seed: int = 0

    def __post_init__(self):
        """Validate configuration after initialization."""
        valid_sizes = ["t", "s", "m", "c"]
        if self.size not in valid_sizes:
            raise ValueError(f"Invalid size '{self.size}'. Must be one of {valid_sizes}")

        valid_schedulers = ["linear", "cos", "warmcos"]
        if self.scheduler not in valid_schedulers:
            raise ValueError(
                f"Invalid scheduler '{self.scheduler}'. Must be one of {valid_schedulers}"
            )

        valid_optimizers = ["sgd", "adam", "adamw"]
        if self.optimizer not in valid_optimizers:
            raise ValueError(
                f"Invalid optimizer '{self.optimizer}'. Must be one of {valid_optimizers}"
            )

        # Ensure scale tuples are tuples
        if isinstance(self.mosaic_scale, list):
            self.mosaic_scale = tuple(self.mosaic_scale)
        if isinstance(self.mixup_scale, list):
            self.mixup_scale = tuple(self.mixup_scale)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "V9TrainConfig":
        """Load configuration from YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        # Handle old field names for backward compatibility
        renames = {
            "batch_size": "batch",
            "num_workers": "workers",
            "save_dir": "project",
            "lr": "lr0",
            "save_interval": "save_period",
        }
        for old, new in renames.items():
            if old in data and new not in data:
                data[new] = data.pop(old)
        return cls(**data)

    def to_yaml(self, path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)

    def update(self, **kwargs) -> "V9TrainConfig":
        """Create a new config with updated values."""
        data = asdict(self)
        data.update(kwargs)
        return V9TrainConfig(**data)

    @property
    def input_size(self) -> Tuple[int, int]:
        """Return input size as (height, width) tuple."""
        return (self.imgsz, self.imgsz)

    @property
    def effective_lr(self) -> float:
        """Calculate effective learning rate based on batch size."""
        # Linear scaling: lr = base_lr * batch / 64
        return self.lr0 * self.batch / 64

    def __repr__(self) -> str:
        lines = ["V9TrainConfig("]
        for key, value in asdict(self).items():
            lines.append(f"  {key}={value!r},")
        lines.append(")")
        return "\n".join(lines)


# Preset configurations for different model sizes
V9_CONFIGS = {
    "t": V9TrainConfig(size="t", imgsz=640),
    "s": V9TrainConfig(size="s", imgsz=640),
    "m": V9TrainConfig(size="m", imgsz=640),
    "c": V9TrainConfig(size="c", imgsz=640),
}


def get_v9_config(size: str = "c", **kwargs) -> V9TrainConfig:
    """
    Get a preset V9 configuration with optional overrides.

    Args:
        size: Model size ("t", "s", "m", "c")
        **kwargs: Override any configuration parameter

    Returns:
        V9TrainConfig instance
    """
    if size not in V9_CONFIGS:
        raise ValueError(f"Unknown size '{size}'. Available: {list(V9_CONFIGS.keys())}")

    config = V9_CONFIGS[size]
    if kwargs:
        config = config.update(**kwargs)
    return config
