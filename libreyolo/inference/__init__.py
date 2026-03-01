"""Inference backends for LibreYOLO."""

from .onnx import OnnxBackend
from .openvino import OpenVINOBackend

__all__ = [
    "OnnxBackend",
    "OpenVINOBackend",
]

# Lazy imports for backends with heavy optional dependencies
def __getattr__(name):
    if name == "TensorRTBackend":
        from .tensorrt import TensorRTBackend
        return TensorRTBackend
    if name == "NcnnBackend":
        from .ncnn import NcnnBackend
        return NcnnBackend
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
