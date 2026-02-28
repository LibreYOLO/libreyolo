"""Inference backends for LibreYOLO."""

from .onnx import LIBREYOLOOnnx
from .openvino import LIBREYOLOOpenVINO

__all__ = [
    "LIBREYOLOOnnx",
    "LIBREYOLOOpenVINO",
]

# Lazy imports for backends with heavy optional dependencies
def __getattr__(name):
    if name == "LIBREYOLOTensorRT":
        from .tensorrt import LIBREYOLOTensorRT
        return LIBREYOLOTensorRT
    if name == "LIBREYOLONCNN":
        from .ncnn import LIBREYOLONCNN
        return LIBREYOLONCNN
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
