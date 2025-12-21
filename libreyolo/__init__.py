"""
Libre YOLO - An open source YOLO library with MIT license.
"""
from importlib.metadata import version, PackageNotFoundError

from .v8.model import LIBREYOLO8
from .v11.model import LIBREYOLO11
from .factory import LIBREYOLO, create_model

try:
    __version__ = version("libreyolo")
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"  # Fallback for editable installs without metadata

__all__ = ["LIBREYOLO", "LIBREYOLO8", "LIBREYOLO11", "create_model"]
