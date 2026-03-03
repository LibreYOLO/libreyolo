"""
Model export utilities for LibreYOLO.

Supports exporting models to various deployment formats:
- ONNX: Universal interchange format
- TorchScript: PyTorch deployment format
- TensorRT: NVIDIA GPU acceleration (requires tensorrt package)
- OpenVINO: Intel CPU/GPU/VPU acceleration (requires openvino package)
- ncnn: Mobile ARM / embedded CPU deployment (requires pnnx package)

Example::

    from libreyolo import LibreYOLO
    from libreyolo.export import BaseExporter, OnnxExporter

    model = LibreYOLO("LibreYOLO9c.pt")

    # Via factory
    BaseExporter.create("onnx", model)(simplify=True)

    # Or direct subclass
    OnnxExporter(model)(dynamic=True)

    # Or the model facade
    model.export(format="tensorrt", half=True)
"""

from .exporter import BaseExporter

__all__ = ["BaseExporter"]
