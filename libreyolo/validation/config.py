"""
Validation configuration for LibreYOLO.

Provides a dataclass for configuring model validation runs.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Tuple, Union

import yaml


@dataclass
class ValidationConfig:
    """
    Configuration for model validation.

    Attributes:
        data: Path to data.yaml file containing dataset configuration.
        data_dir: Direct path to dataset directory (alternative to data).
        split: Dataset split to validate on ("val" or "test").
        batch_size: Batch size for validation.
        imgsz: Image size for validation (assumes square input).
        conf_thres: Confidence threshold. Use low value (0.001) for mAP calculation.
        iou_thres: IoU threshold for NMS.
        max_det: Maximum detections per image.
        iou_thresholds: IoU thresholds for mAP calculation (default: 0.50 to 0.95).
        device: Device to use ("auto", "cuda", "mps", "cpu").
        save_dir: Directory to save results.
        save_json: Whether to save predictions in COCO JSON format.
        plots: Whether to generate confusion matrix and other plots.
        verbose: Whether to print detailed metrics.
        num_workers: Number of dataloader workers.
        half: Whether to use FP16 inference.
    """

    # Data configuration
    data: Optional[str] = None
    data_dir: Optional[str] = None
    split: str = "val"

    # Inference settings
    batch_size: int = 16
    imgsz: int = 640
    conf_thres: float = 0.001
    iou_thres: float = 0.6
    max_det: int = 300

    # Metrics settings
    iou_thresholds: Tuple[float, ...] = (
        0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95
    )

    # Device
    device: str = "auto"

    # Output
    save_dir: Optional[str] = None
    save_json: bool = False
    plots: bool = True
    verbose: bool = True

    # Workers
    num_workers: int = 4

    # Precision
    half: bool = False

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.data is None and self.data_dir is None:
            raise ValueError("Either 'data' or 'data_dir' must be specified")

        if self.split not in ("val", "test", "train"):
            raise ValueError(f"Invalid split: {self.split}. Must be 'val', 'test', or 'train'")

        if not 0 < self.conf_thres < 1:
            raise ValueError(f"conf_thres must be in (0, 1), got {self.conf_thres}")

        if not 0 < self.iou_thres < 1:
            raise ValueError(f"iou_thres must be in (0, 1), got {self.iou_thres}")

        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "ValidationConfig":
        """
        Load configuration from YAML file.

        Args:
            path: Path to YAML configuration file.

        Returns:
            ValidationConfig instance.
        """
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)

        # Convert iou_thresholds list to tuple if present
        if "iou_thresholds" in config_dict and isinstance(config_dict["iou_thresholds"], list):
            config_dict["iou_thresholds"] = tuple(config_dict["iou_thresholds"])

        return cls(**config_dict)

    def to_yaml(self, path: Union[str, Path]) -> None:
        """
        Save configuration to YAML file.

        Args:
            path: Path to save YAML configuration.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        config_dict = self.to_dict()
        # Convert tuple to list for YAML serialization
        config_dict["iou_thresholds"] = list(config_dict["iou_thresholds"])

        with open(path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    def to_dict(self) -> dict:
        """
        Convert configuration to dictionary.

        Returns:
            Configuration as dictionary.
        """
        return asdict(self)

    def update(self, **kwargs) -> "ValidationConfig":
        """
        Create new configuration with updated values.

        Args:
            **kwargs: Configuration values to update.

        Returns:
            New ValidationConfig with updated values.
        """
        current = self.to_dict()
        current.update(kwargs)

        # Convert iou_thresholds to tuple if it's a list
        if isinstance(current.get("iou_thresholds"), list):
            current["iou_thresholds"] = tuple(current["iou_thresholds"])

        return ValidationConfig(**current)
