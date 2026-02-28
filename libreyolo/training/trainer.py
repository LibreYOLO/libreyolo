"""Re-export shim — real code lives in libreyolo.models.yolox.trainer."""
from ..models.yolox.trainer import YOLOXTrainer

__all__ = ["YOLOXTrainer"]
