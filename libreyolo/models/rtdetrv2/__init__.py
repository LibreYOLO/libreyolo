"""RT-DETRv2 family — improved baseline of RT-DETR with bag-of-freebies training."""

from .model import LibreRTDETRv2
from .trainer import RTDETRv2Trainer

__all__ = ["LibreRTDETRv2", "RTDETRv2Trainer"]
