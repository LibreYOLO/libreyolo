"""LibreYOLOWorld — open-vocabulary YOLO (text-prompted detection)."""

from .model import LibreYOLOWorld
from .nn import (
    LibreYOLOWorldModel,
    TextEncoder,
    YOLOv8CSPDarknet,
    YOLOWorldPAFPN,
    YOLOWorldHeadModule,
    MaxSigmoidAttnBlock,
    MaxSigmoidCSPLayerWithTwoConv,
    BNContrastiveHead,
    CLIP_EMBED_DIM,
)

__all__ = [
    "LibreYOLOWorld",
    "LibreYOLOWorldModel",
    "TextEncoder",
    "YOLOv8CSPDarknet",
    "YOLOWorldPAFPN",
    "YOLOWorldHeadModule",
    "MaxSigmoidAttnBlock",
    "MaxSigmoidCSPLayerWithTwoConv",
    "BNContrastiveHead",
    "CLIP_EMBED_DIM",
]
