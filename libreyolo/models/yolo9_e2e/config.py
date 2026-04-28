"""Training config for YOLOv9 end-to-end (NMS-free)."""

from dataclasses import dataclass

from ...training.config import YOLO9Config


@dataclass(kw_only=True)
class YOLO9E2EConfig(YOLO9Config):
    """YOLOv9 E2E uses the same training defaults as YOLOv9."""

    name: str = "yolo9_e2e_exp"
