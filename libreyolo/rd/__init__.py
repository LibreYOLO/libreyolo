"""
YOLO-RD module for LibreYOLO.

This module provides the YOLO-RD (Regional Diversity) model implementation.
YOLO-RD extends YOLOv9-c with DConv (Dynamic Convolution with PONO) for
enhanced regional feature diversity.
"""

from .model import LIBREYOLORD
from .nn import LibreYOLORDModel

__all__ = ["LIBREYOLORD", "LibreYOLORDModel"]
