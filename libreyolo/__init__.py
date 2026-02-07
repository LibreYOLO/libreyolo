"""
Libre YOLO - An open source YOLO library with MIT license.
"""
from importlib.metadata import version, PackageNotFoundError

from .v9.model import LIBREYOLO9
from .yolox.model import LIBREYOLOX
from .factory import LIBREYOLO, create_model

# Lazy import for RF-DETR to avoid dependency issues
def __getattr__(name):
    if name == "LIBREYOLORFDETR":
        import importlib.util
        if importlib.util.find_spec("rfdetr") is None:
            raise ModuleNotFoundError(
                "RF-DETR support requires extra dependencies.\n"
                "Install with: pip install libreyolo[rfdetr]"
            )
        from .rfdetr.model import LIBREYOLORFDETR
        return LIBREYOLORFDETR
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
from .export import Exporter
from .common.onnx import LIBREYOLOOnnx
from .common.results import Results, Boxes
# CAM/GradCAM removed
from .validation import (
    ValidationConfig,
    BaseValidator,
    DetectionValidator,
    DetMetrics,
)
from .data import (
    DATASETS_DIR,
    load_data_config,
    check_dataset,
)

from pathlib import Path as _Path
SAMPLE_IMAGE = str(_Path(__file__).parent / "assets" / "parkour.jpg")

try:
    __version__ = version("libreyolo")
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"  # Fallback for editable installs without metadata

__all__ = [
    # Export
    "Exporter",
    # Main API
    "LIBREYOLO",
    "LIBREYOLO9",
    "LIBREYOLOX",
    "LIBREYOLORFDETR",
    "LIBREYOLOOnnx",
    "create_model",
    # Results
    "Results",
    "Boxes",
    # Validation
    "ValidationConfig",
    "BaseValidator",
    "DetectionValidator",
    "DetMetrics",
    # Data utilities
    "DATASETS_DIR",
    "load_data_config",
    "check_dataset",
    # Assets
    "SAMPLE_IMAGE",
]
