"""YOLOv9 E2E trainer."""

from .config import YOLO9E2EConfig
from ..yolo9.trainer import YOLO9Trainer


class YOLO9E2ETrainer(YOLO9Trainer):
    """Thin trainer subclass for yolo9_e2e family metadata and defaults."""

    @classmethod
    def _config_class(cls):
        return YOLO9E2EConfig

    def get_model_family(self) -> str:
        return "yolo9_e2e"

    def get_model_tag(self) -> str:
        return f"YOLOv9-E2E-{self.config.size}"
