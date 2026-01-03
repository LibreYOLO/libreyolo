"""
Libre YOLO - An open source YOLO library with MIT license.
"""
from importlib.metadata import version, PackageNotFoundError

from .v8.model import LIBREYOLO8
from .v11.model import LIBREYOLO11
from .v9.model import LIBREYOLO9
from .v7.model import LIBREYOLO7
from .rd.model import LIBREYOLORD
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

try:
    __version__ = version("libreyolo")
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"  # Fallback for editable installs without metadata

__all__ = [
    # Main API
    "LIBREYOLO",
    "LIBREYOLO7",
    "LIBREYOLO8",
    "LIBREYOLO9",
    "LIBREYOLO11",
    "LIBREYOLORD",
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
]
