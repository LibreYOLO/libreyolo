"""
Libre YOLO - An open source YOLO library with MIT license.
"""
from importlib.metadata import version, PackageNotFoundError

from .v8.model import LIBREYOLO8
from .v11.model import LIBREYOLO11
from .v9.model import LIBREYOLO9
from .yolox.model import LIBREYOLOX
from .factory import LIBREYOLO, create_model
from .common.onnx import LIBREYOLOOnnx
from .common.cam import (
    CAM_METHODS,
    BaseCAM,
    EigenCAM,
    GradCAM,
    HiResCAM,
    XGradCAM,
    LayerCAM,
    EigenGradCAM,
    GradCAMPlusPlus,
)
from .validation import (
    ValidationConfig,
    BaseValidator,
    DetectionValidator,
    DetMetrics,
    ConfusionMatrix,
)
from .data import (
    DATASETS_DIR,
    load_data_config,
    check_dataset,
)

try:
    __version__ = version("libreyolo")
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"  # Fallback for editable installs without metadata

__all__ = [
    # Main API
    "LIBREYOLO",
    "LIBREYOLO8",
    "LIBREYOLO9",
    "LIBREYOLO11",
    "LIBREYOLOX",
    "LIBREYOLOOnnx",
    "create_model",
    # CAM methods
    "CAM_METHODS",
    "BaseCAM",
    "EigenCAM",
    "GradCAM",
    "HiResCAM",
    "XGradCAM",
    "LayerCAM",
    "EigenGradCAM",
    "GradCAMPlusPlus",
    # Validation
    "ValidationConfig",
    "BaseValidator",
    "DetectionValidator",
    "DetMetrics",
    "ConfusionMatrix",
    # Data utilities
    "DATASETS_DIR",
    "load_data_config",
    "check_dataset",
]
