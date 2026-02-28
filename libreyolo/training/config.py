"""Re-export shim — real code lives in libreyolo.models.yolox.config."""
from ..models.yolox.config import YOLOXTrainConfig, get_config, YOLOX_CONFIGS

__all__ = ["YOLOXTrainConfig", "get_config", "YOLOX_CONFIGS"]
