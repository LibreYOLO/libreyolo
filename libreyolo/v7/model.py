"""
LibreYOLO7 inference wrapper.

Provides a high-level API for YOLOv7 object detection inference.
YOLOv7 uses anchor-based detection with implicit layers.
"""

from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from PIL import Image

from ..common.base_model import LibreYOLOBase
from ..common.image_loader import ImageInput
from ..common.utils import preprocess_image
from .nn import LibreYOLO7Model
from .utils import postprocess


class LIBREYOLO7(LibreYOLOBase):
    """
    LibreYOLO7 model for object detection.

    Uses anchor-based detection with ImplicitA/ImplicitM layers.

    Args:
        model_path: Model weights source. Can be:
            - str: Path to a .pt/.pth weights file
            - dict: Pre-loaded state_dict (e.g., from torch.load())
        size: Model size variant (required). Must be one of: "base", "tiny"
        nb_classes: Number of classes (default: 80 for COCO)
        device: Device for inference

    Example:
        >>> model = LIBREYOLO7(model_path="path/to/weights.pt", size="base")
        >>> detections = model(image=image_path, save=True)
    """

    def __init__(
        self,
        model_path,
        size: str,
        nb_classes: int = 80,
        device: str = "auto",
        **kwargs,
    ):
        super().__init__(
            model_path=model_path,
            size=size,
            nb_classes=nb_classes,
            device=device,
            **kwargs,
        )

    def _strict_loading(self) -> bool:
        """YOLOv7 uses non-strict loading as anchors/stride buffers are defined in model."""
        return False

    def _get_valid_sizes(self) -> List[str]:
        return ["base", "tiny"]

    def _get_model_name(self) -> str:
        return "LIBREYOLO7"

    def _get_input_size(self) -> int:
        return 640

    def _init_model(self) -> nn.Module:
        return LibreYOLO7Model(config=self.size, nb_classes=self.nb_classes)

    def _get_available_layers(self) -> Dict[str, nn.Module]:
        return {
            # Backbone layers
            "backbone_elan1": self.model.backbone.elan1,
            "backbone_elan2": self.model.backbone.elan2,
            "backbone_elan3": self.model.backbone.elan3,
            "backbone_elan4": self.model.backbone.elan4,
            # Neck layers
            "neck_sppcspc": self.model.neck.sppcspc,
            "neck_elan_up1": self.model.neck.elan_up1,
            "neck_elan_up2": self.model.neck.elan_up2,
            "neck_elan_down1": self.model.neck.elan_down1,
            "neck_elan_down2": self.model.neck.elan_down2,
            # Rep layers
            "rep_p3": self.model.rep_p3,
            "rep_p4": self.model.rep_p4,
            "rep_p5": self.model.rep_p5,
        }

    def _preprocess(
        self, image: ImageInput, color_format: str = "auto"
    ) -> Tuple[torch.Tensor, Image.Image, Tuple[int, int]]:
        return preprocess_image(image, input_size=640, color_format=color_format)

    def _forward(self, input_tensor: torch.Tensor) -> Any:
        return self.model(input_tensor)

    def _postprocess(
        self,
        output: Any,
        conf_thres: float,
        iou_thres: float,
        original_size: Tuple[int, int],
        **kwargs,
    ) -> Dict:
        return postprocess(
            output,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            input_size=640,
            original_size=original_size,
        )
