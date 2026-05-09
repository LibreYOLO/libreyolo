"""RT-DETRv4 family — HGNetv2 student detectors distilled from a DINOv3 ViT teacher."""

from .model import LibreRTDETRv4
from .trainer import RTDETRv4Trainer

__all__ = ["LibreRTDETRv4", "RTDETRv4Trainer"]
