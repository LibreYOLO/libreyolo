"""
LibreYOLO RF-DETR - Detection Transformer with DINOv2 backbone.

A LibreYOLO wrapper for RF-DETR with 100% weight compatibility.

Example usage:
    >>> from libreyolo import LibreYOLORFDETR
    >>>
    >>> # Use pretrained COCO weights (auto-downloads)
    >>> model = LibreYOLORFDETR(size="s")  # or "n", "m", "l"
    >>> detections = model.predict("path/to/image.jpg")
    >>> print(detections["boxes"], detections["scores"], detections["classes"])
    >>>
    >>> # With custom weights
    >>> model = LibreYOLORFDETR(model_path="custom_weights.pth", size="s")
    >>>
    >>> # Training (Ultralytics-style API)
    >>> model = LibreYOLORFDETR(size="s")
    >>> model.train(data="coco128", epochs=10, batch_size=4)

Available model sizes:
    - "n" (nano): Fastest, smallest
    - "s" (small): Fast, lightweight
    - "m" (medium): Better accuracy
    - "l" (large): Best accuracy, slowest
"""

from .model import LibreYOLORFDETR
from .nn import LibreRFDETRModel, create_rfdetr_model, RFDETR_CONFIGS
from .utils import postprocess, cxcywh_to_xyxy
from .trainer import train_rfdetr, RFDETR_TRAINERS

__all__ = [
    # Main model wrapper
    "LibreYOLORFDETR",
    # Neural network
    "LibreRFDETRModel",
    "create_rfdetr_model",
    "RFDETR_CONFIGS",
    # Postprocessing
    "postprocess",
    "cxcywh_to_xyxy",
    # Training
    "train_rfdetr",
    "RFDETR_TRAINERS",
]

__version__ = "0.1.0"
