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

__all__ = ["BaseExporter"]
