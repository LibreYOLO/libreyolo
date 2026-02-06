"""
Export configuration handling.

Provides loading and validation of export configuration from YAML files.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union
import yaml


@dataclass
class DynamicBatchConfig:
    """Dynamic batching configuration."""
    enabled: bool = False
    min_batch: int = 1
    opt_batch: int = 1
    max_batch: int = 8


@dataclass
class Int8CalibrationConfig:
    """INT8 calibration configuration."""
    dataset: str = "coco5000.yaml"
    fraction: float = 0.1
    cache: bool = True


@dataclass
class OutputConfig:
    """Output settings."""
    add_precision_suffix: bool = True
    overwrite: bool = True


@dataclass
class TensorRTExportConfig:
    """
    TensorRT export configuration.

    This class holds all configuration options for TensorRT engine export.
    It can be loaded from a YAML file or created programmatically.

    Attributes:
        precision: Precision mode ('fp32', 'fp16', 'int8')
        workspace: GPU workspace size in GiB
        verbose: Enable verbose TensorRT logging
        hardware_compatibility: Hardware compatibility level
            ('none', 'ampere_plus', 'same_compute_capability')
        device: GPU device ID for multi-GPU systems
        dynamic: Dynamic batching configuration
        int8_calibration: INT8 calibration settings
        output: Output file settings
    """
    precision: str = "fp16"
    workspace: float = 4.0
    verbose: bool = False
    hardware_compatibility: str = "none"
    device: int = 0
    dynamic: DynamicBatchConfig = field(default_factory=DynamicBatchConfig)
    int8_calibration: Int8CalibrationConfig = field(default_factory=Int8CalibrationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate()

    def _validate(self):
        """Validate configuration values."""
        valid_precisions = ("fp32", "fp16", "int8")
        if self.precision not in valid_precisions:
            raise ValueError(
                f"Invalid precision '{self.precision}'. "
                f"Must be one of: {valid_precisions}"
            )

        valid_hw_compat = ("none", "ampere_plus", "same_compute_capability")
        if self.hardware_compatibility not in valid_hw_compat:
            raise ValueError(
                f"Invalid hardware_compatibility '{self.hardware_compatibility}'. "
                f"Must be one of: {valid_hw_compat}"
            )

        if self.workspace <= 0:
            raise ValueError(f"workspace must be positive, got {self.workspace}")

        if self.device < 0:
            raise ValueError(f"device must be non-negative, got {self.device}")

        if not 0 < self.int8_calibration.fraction <= 1:
            raise ValueError(
                f"int8_calibration.fraction must be in (0, 1], "
                f"got {self.int8_calibration.fraction}"
            )

    @property
    def half(self) -> bool:
        """Whether to use FP16 precision."""
        return self.precision in ("fp16", "int8")

    @property
    def int8(self) -> bool:
        """Whether to use INT8 precision."""
        return self.precision == "int8"

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "TensorRTExportConfig":
        """
        Load configuration from a YAML file.

        Args:
            path: Path to YAML configuration file.

        Returns:
            TensorRTExportConfig instance.

        Raises:
            FileNotFoundError: If config file doesn't exist.
            ValueError: If config is invalid.
        """
        path = Path(path)
        if not path.exists():
            # Check in default config directory
            default_path = Path(__file__).parent.parent / "cfg" / "export" / path.name
            if default_path.exists():
                path = default_path
            else:
                raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as f:
            data = yaml.safe_load(f)

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "TensorRTExportConfig":
        """
        Create configuration from a dictionary.

        Args:
            data: Configuration dictionary.

        Returns:
            TensorRTExportConfig instance.
        """
        # Parse nested configs
        dynamic_data = data.get("dynamic", {})
        dynamic = DynamicBatchConfig(
            enabled=dynamic_data.get("enabled", False),
            min_batch=dynamic_data.get("min_batch", 1),
            opt_batch=dynamic_data.get("opt_batch", 1),
            max_batch=dynamic_data.get("max_batch", 8),
        )

        int8_data = data.get("int8_calibration", {})
        int8_calibration = Int8CalibrationConfig(
            dataset=int8_data.get("dataset", "coco5000.yaml"),
            fraction=int8_data.get("fraction", 0.1),
            cache=int8_data.get("cache", True),
        )

        output_data = data.get("output", {})
        output = OutputConfig(
            add_precision_suffix=output_data.get("add_precision_suffix", True),
            overwrite=output_data.get("overwrite", True),
        )

        return cls(
            precision=data.get("precision", "fp16"),
            workspace=data.get("workspace", 4.0),
            verbose=data.get("verbose", False),
            hardware_compatibility=data.get("hardware_compatibility", "none"),
            device=data.get("device", 0),
            dynamic=dynamic,
            int8_calibration=int8_calibration,
            output=output,
        )

    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            "precision": self.precision,
            "workspace": self.workspace,
            "verbose": self.verbose,
            "hardware_compatibility": self.hardware_compatibility,
            "device": self.device,
            "dynamic": {
                "enabled": self.dynamic.enabled,
                "min_batch": self.dynamic.min_batch,
                "opt_batch": self.dynamic.opt_batch,
                "max_batch": self.dynamic.max_batch,
            },
            "int8_calibration": {
                "dataset": self.int8_calibration.dataset,
                "fraction": self.int8_calibration.fraction,
                "cache": self.int8_calibration.cache,
            },
            "output": {
                "add_precision_suffix": self.output.add_precision_suffix,
                "overwrite": self.output.overwrite,
            },
        }


def load_export_config(
    config: Optional[Union[str, Path, dict, TensorRTExportConfig]] = None
) -> TensorRTExportConfig:
    """
    Load export configuration from various sources.

    Args:
        config: Configuration source. Can be:
            - None: Use default configuration
            - str/Path: Path to YAML config file
            - dict: Configuration dictionary
            - TensorRTExportConfig: Use as-is

    Returns:
        TensorRTExportConfig instance.

    Examples:
        >>> config = load_export_config()  # Default config
        >>> config = load_export_config("my_config.yaml")  # From file
        >>> config = load_export_config({"precision": "int8"})  # From dict
    """
    if config is None:
        return TensorRTExportConfig()

    if isinstance(config, TensorRTExportConfig):
        return config

    if isinstance(config, dict):
        return TensorRTExportConfig.from_dict(config)

    if isinstance(config, (str, Path)):
        return TensorRTExportConfig.from_yaml(config)

    raise TypeError(
        f"config must be None, str, Path, dict, or TensorRTExportConfig, "
        f"got {type(config)}"
    )


# Default configuration instance
DEFAULT_CONFIG = TensorRTExportConfig()
