"""
RT-DETR implementation for LibreYOLO.

RT-DETR (Real-Time DEtection TRansformer) is a transformer-based object detector
that achieves real-time performance while maintaining high accuracy.

This module provides:
- RTDETRModel: The complete RT-DETR neural network architecture
- LIBREYOLORTDETR: High-level wrapper with LibreYOLO-compatible API
"""

from .model import LIBREYOLORTDETR
from .nn import RTDETRModel

__all__ = [
    "LIBREYOLORTDETR",
    "RTDETRModel",
]
