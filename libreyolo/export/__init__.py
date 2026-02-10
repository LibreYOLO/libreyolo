"""
Model export utilities for LibreYOLO.

Supports exporting models to various deployment formats:
- ONNX: Universal interchange format
- TorchScript: PyTorch deployment format
- TensorRT: NVIDIA GPU acceleration (requires tensorrt package)
- OpenVINO: Intel CPU/GPU/VPU acceleration (requires openvino package)
- ncnn: Mobile ARM / embedded CPU deployment (requires pnnx package)

Example::

    from libreyolo import LIBREYOLO
    from libreyolo.export import Exporter

    model = LIBREYOLO("yolov9c.pt")

    # ONNX export
    model.export(format="onnx")

    # TensorRT with FP16
    model.export(format="tensorrt", half=True)

    # TensorRT with INT8
    model.export(format="tensorrt", int8=True, data="coco8.yaml")

    # TensorRT with config file
    model.export(format="tensorrt", trt_config="tensorrt_default.yaml")
"""

from .exporter import Exporter
from .config import (
    TensorRTExportConfig,
    DynamicBatchConfig,
    Int8CalibrationConfig,
    load_export_config,
)

__all__ = [
    "Exporter",
    "TensorRTExportConfig",
    "DynamicBatchConfig",
    "Int8CalibrationConfig",
    "load_export_config",
]
