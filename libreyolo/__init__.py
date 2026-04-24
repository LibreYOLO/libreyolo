"""Libre YOLO — open source YOLO library with MIT license."""

from importlib.metadata import version, PackageNotFoundError
from pathlib import Path as _Path

# Core API — always available
from .models import LibreYOLO, LibreYOLOX, LibreYOLO9, LibreYOLONAS, LibreDFINE
from .utils.results import Results, Boxes, Masks

SAMPLE_IMAGE = str(_Path(__file__).parent / "assets" / "parkour.jpg")

try:
    __version__ = version("libreyolo")
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"


# Lazy imports for optional/heavy modules
def __getattr__(name):
    _lazy = {
        "LibreYOLORFDETR": (".models.rfdetr.model", "LibreYOLORFDETR"),
        "OnnxBackend": (".backends.onnx", "OnnxBackend"),
        "OpenVINOBackend": (".backends.openvino", "OpenVINOBackend"),
        "TensorRTBackend": (".backends.tensorrt", "TensorRTBackend"),
        "NcnnBackend": (".backends.ncnn", "NcnnBackend"),
        "BaseExporter": (".export", "BaseExporter"),
        "DetectionValidator": (".validation", "DetectionValidator"),
        "ValidationConfig": (".validation", "ValidationConfig"),
        "DetMetrics": (".validation", "DetMetrics"),
        "ByteTracker": (".tracking", "ByteTracker"),
        "TrackConfig": (".tracking", "TrackConfig"),
        "DATASETS_DIR": (".data", "DATASETS_DIR"),
        "load_data_config": (".data", "load_data_config"),
        "check_dataset": (".data", "check_dataset"),
    }
    if name == "LibreYOLORFDETR":
        # RF-DETR needs dependency check before import
        from .models import _ensure_rfdetr

        _ensure_rfdetr()
    if name in _lazy:
        import importlib

        module_path, attr = _lazy[name]
        mod = importlib.import_module(module_path, package=__name__)
        return getattr(mod, attr)
    raise AttributeError(f"module 'libreyolo' has no attribute '{name}'")


__all__ = [
    # Main API
    "LibreYOLO",
    "LibreYOLO9",
    "LibreYOLONAS",
    "LibreYOLOX",
    "LibreYOLORFDETR",
    "LibreDFINE",
    # Results
    "Results",
    "Boxes",
    "Masks",
    # Assets
    "SAMPLE_IMAGE",
    # Tracking
    "ByteTracker",
    "TrackConfig",
    # Lazy-loaded
    "OnnxBackend",
    "OpenVINOBackend",
    "TensorRTBackend",
    "NcnnBackend",
    "BaseExporter",
    "DetectionValidator",
    "ValidationConfig",
    "DetMetrics",
    "DATASETS_DIR",
    "load_data_config",
    "check_dataset",
]
