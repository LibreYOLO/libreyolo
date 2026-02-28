"""
LibreYOLO RF-DETR - Detection Transformer with DINOv2 backbone.

A LibreYOLO wrapper for RF-DETR with 100% weight compatibility.

Example usage:
    >>> from libreyolo import LIBREYOLORFDETR
    >>>
    >>> # Use pretrained COCO weights (auto-downloads)
    >>> model = LIBREYOLORFDETR(size="s")  # or "n", "m", "l"
    >>> detections = model.predict("path/to/image.jpg")
    >>> print(detections["boxes"], detections["scores"], detections["classes"])
    >>>
    >>> # With custom weights
    >>> model = LIBREYOLORFDETR(model_path="custom_weights.pth", size="s")
    >>>
    >>> # Training (Ultralytics-style API)
    >>> model = LIBREYOLORFDETR(size="s")
    >>> model.train(data="coco128", epochs=10, batch_size=4)

Available model sizes:
    - "n" (nano): Fastest, smallest
    - "s" (small): Fast, lightweight
    - "m" (medium): Better accuracy
    - "l" (large): Best accuracy, slowest
"""

from .model import LIBREYOLORFDETR
from .nn import RFDETRModel, create_rfdetr_model, RFDETR_CONFIGS
from .utils import postprocess, box_cxcywh_to_xyxy
from .train import train_rfdetr, RFDETR_TRAINERS

__all__ = [
    # Main model wrapper
    "LIBREYOLORFDETR",
    # Neural network
    "RFDETRModel",
    "create_rfdetr_model",
    "RFDETR_CONFIGS",
    # Postprocessing
    "postprocess",
    "box_cxcywh_to_xyxy",
    # Training
    "train_rfdetr",
    "RFDETR_TRAINERS",
]

__version__ = "0.1.0"
