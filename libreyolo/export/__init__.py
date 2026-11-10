"""
Model export utilities for LibreYOLO.

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
from .gguf import GgufExporter
from .rknn import (
    compare_rknn_outputs,
    export_rknn,
    run_rknn_simulator,
    verify_rknn_simulator_parity,
)
from .support import SupportEntry, get_support

__all__ = [
    "BaseExporter",
    "GgufExporter",
    "SupportEntry",
    "compare_rknn_outputs",
    "export_rknn",
    "get_support",
    "run_rknn_simulator",
    "verify_rknn_simulator_parity",
]
