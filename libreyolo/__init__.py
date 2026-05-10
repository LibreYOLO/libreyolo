"""Libre YOLO — open source YOLO library with MIT license."""

from importlib.metadata import version, PackageNotFoundError
from pathlib import Path as _Path

# Core API — always available
from .models import (
    LibreYOLO,
    LibreYOLOX,
    LibreYOLO9,
    LibreYOLO9E2E,
    LibreYOLONAS,
    LibreDFINE,
    LibreDEIM,
    LibreDEIMv2,
    LibreEC,
    LibrePICODET,
    LibreRTDETR,
    LibreRTDETRv2,
    LibreRTDETRv4,
    LibreRTMDet,
)
from .utils.results import Results, Boxes, Masks, Keypoints, Probs, OBB

SAMPLE_IMAGE = str(_Path(__file__).parent / "assets" / "parkour.jpg")

try:
    __version__ = version("libreyolo")
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"


# Old class names that were renamed for nomenclature consistency. Resolved
# via __getattr__ with a DeprecationWarning so existing imports keep working.
_DEPRECATED_ALIASES = {
    "LibreYOLORTDETR": "LibreRTDETR",
    "LibreYOLORFDETR": "LibreRFDETR",
}


# Lazy imports for optional/heavy modules
def __getattr__(name):
    if name in _DEPRECATED_ALIASES:
        new_name = _DEPRECATED_ALIASES[name]
        import sys
        import warnings

        warnings.warn(
            f"{name} has been renamed to {new_name}. Update your imports — "
            "the old name will be removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
        # ``getattr`` on the module object resolves both eager imports
        # (``LibreRTDETR`` in globals) and the lazy ``__getattr__`` path
        # (``LibreRFDETR``); recursing into ``__getattr__`` directly would
        # skip the eager case.
        return getattr(sys.modules[__name__], new_name)

    _lazy = {
        "LibreRFDETR": (".models.rfdetr.model", "LibreRFDETR"),
        "OnnxBackend": (".backends.onnx", "OnnxBackend"),
        "OpenVINOBackend": (".backends.openvino", "OpenVINOBackend"),
        "TensorRTBackend": (".backends.tensorrt", "TensorRTBackend"),
        "NcnnBackend": (".backends.ncnn", "NcnnBackend"),
        "BaseExporter": (".export", "BaseExporter"),
        "DetectionValidator": (".validation", "DetectionValidator"),
        "SegmentationValidator": (".validation", "SegmentationValidator"),
        "PoseValidator": (".validation", "PoseValidator"),
        "ValidationConfig": (".validation", "ValidationConfig"),
        "ByteTracker": (".tracking", "ByteTracker"),
        "TrackConfig": (".tracking", "TrackConfig"),
        "DATASETS_DIR": (".data", "DATASETS_DIR"),
        "load_data_config": (".data", "load_data_config"),
        "check_dataset": (".data", "check_dataset"),
    }
    if name == "LibreRFDETR":
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
    "LibreYOLO9E2E",
    "LibreYOLONAS",
    "LibreYOLOX",
    "LibreRTDETR",
    "LibreRTDETRv2",
    "LibreRTDETRv4",
    "LibreRFDETR",
    "LibreDFINE",
    "LibreDEIM",
    "LibreDEIMv2",
    "LibreEC",
    "LibrePICODET",
    "LibreRTMDet",
    # Results
    "Results",
    "Boxes",
    "Masks",
    "Keypoints",
    "Probs",
    "OBB",
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
    "SegmentationValidator",
    "PoseValidator",
    "ValidationConfig",
    "DATASETS_DIR",
    "load_data_config",
    "check_dataset",
]
