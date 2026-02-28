"""Re-export shim — real code lives in libreyolo.models.yolox."""
from ..models.yolox.model import LIBREYOLOX
from ..models.yolox.nn import YOLOXModel

__all__ = ["LIBREYOLOX", "YOLOXModel"]
