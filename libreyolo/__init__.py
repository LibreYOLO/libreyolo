"""
Libre YOLO - An open source YOLO library with MIT license.
"""

from .v8.model import LIBREYOLO8
from .v11.model import LIBREYOLO11
from .factory import LIBREYOLO, create_model

__version__ = "0.1.0"
__all__ = ["LIBREYOLO", "LIBREYOLO8", "LIBREYOLO11", "create_model"]
