"""
RT-DETR Training Configuration.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Union, Tuple
import yaml


@dataclass
class RTDETRTrainConfig:
    """Configuration for RT-DETR training."""

    # === Required ===
    data: str  # Path to data.yaml

    # === Model ===
    size: str = "s"  # s, ms, m, l, x
    num_classes: int = 80
    pretrained: Optional[str] = None  # Path to pretrained weights

    # === Training ===
    epochs: int = 72
    batch_size: int = 16
    imgsz: int = 640

    # === Device ===
    device: str = "auto"  # auto, cuda, cpu, mps
    workers: int = 4

    # === Optimizer (RT-DETR specific) ===
    optimizer: str = "adamw"
    lr: float = 0.0001  # 1e-4 for encoder/decoder
    backbone_lr_mult: float = 0.1  # Backbone uses lr * 0.1
    weight_decay: float = 0.0001
    betas: Tuple[float, float] = (0.9, 0.999)

    # === LR Schedule ===
    lr_drop_epochs: int = 60  # Drop LR at this epoch

    # === Loss weights ===
    loss_vfl_weight: float = 1.0
    loss_bbox_weight: float = 5.0
    loss_giou_weight: float = 2.0

    # === Matcher costs ===
    matcher_cost_class: float = 2.0
    matcher_cost_bbox: float = 5.0
    matcher_cost_giou: float = 2.0

    # === Training features ===
    amp: bool = True  # Automatic mixed precision
    ema: bool = True  # Exponential moving average
    ema_decay: float = 0.9999
    clip_grad: float = 0.1  # Gradient clipping

    # === Output ===
    project: str = "runs/train"
    name: str = "rtdetr"
    save_period: int = 10  # Save checkpoint every N epochs

    # === Resume ===
    resume: Union[bool, str] = False

    # === Reproducibility ===
    seed: int = 0

    def __post_init__(self):
        """Validate configuration."""
        valid_sizes = ["s", "ms", "m", "l", "x"]
        if self.size not in valid_sizes:
            raise ValueError(f"Invalid size '{self.size}'. Must be one of {valid_sizes}")

        valid_optimizers = ["adamw", "adam", "sgd"]
        if self.optimizer.lower() not in valid_optimizers:
            raise ValueError(
                f"Invalid optimizer '{self.optimizer}'. Must be one of {valid_optimizers}"
            )

        # Convert betas to tuple if list
        if isinstance(self.betas, list):
            self.betas = tuple(self.betas)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "RTDETRTrainConfig":
        """Load configuration from YAML file."""
        with open(path) as f:
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

    def update(self, **kwargs) -> "RTDETRTrainConfig":
        """Create a new config with updated values."""
        data = asdict(self)
        data.update(kwargs)
        return RTDETRTrainConfig(**data)

    def __repr__(self) -> str:
        lines = ["RTDETRTrainConfig("]
        for key, value in asdict(self).items():
            lines.append(f"  {key}={value!r},")
        lines.append(")")
        return "\n".join(lines)
