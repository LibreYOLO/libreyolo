"""
YOLOv7 module for LibreYOLO.

This module provides YOLOv7 model implementation with anchor-based detection.
Uses IDetection head with implicit layers (ImplicitA, ImplicitM).
"""

from .model import LIBREYOLO7
from .nn import LibreYOLO7Model

__all__ = ["LIBREYOLO7", "LibreYOLO7Model"]
