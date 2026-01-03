"""
LibreYOLO9 inference wrapper.

Provides a high-level API for YOLOv9 object detection inference.
"""

from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from PIL import Image

from ..common.base_model import LibreYOLOBase
from ..common.image_loader import ImageInput
from ..common.utils import preprocess_image
from .nn import LibreYOLO9Model
from .utils import postprocess


class LIBREYOLO9(LibreYOLOBase):
    """
    LibreYOLO9 model for object detection.

    Args:
        model_path: Model weights source. Can be:
            - str: Path to a .pt/.pth weights file
            - dict: Pre-loaded state_dict (e.g., from torch.load())
        size: Model size variant (required). Must be one of: "t", "s", "m", "c"
        reg_max: Regression max value for DFL (default: 16)
        nb_classes: Number of classes (default: 80 for COCO)
        device: Device for inference. "auto" uses CUDA if available, else MPS, else CPU.

    Example:
        >>> model = LIBREYOLO9(model_path="path/to/weights.pt", size="s")
        >>> detections = model(image=image_path, save=True)
        >>> # Use tiling for large images
        >>> detections = model(image=large_image_path, save=True, tiling=True)
    """

    def __init__(
        self,
        model_path,
        size: str,
        reg_max: int = 16,
        nb_classes: int = 80,
        device: str = "auto",
        **kwargs,
    ):
        self.reg_max = reg_max
        super().__init__(
            model_path=model_path,
            size=size,
            nb_classes=nb_classes,
            device=device,
            **kwargs,
        )

    def _get_valid_sizes(self) -> List[str]:
        return ["t", "s", "m", "c"]

    def _get_model_name(self) -> str:
        return "LIBREYOLO9"

    def _get_input_size(self) -> int:
        return 640

    def _init_model(self) -> nn.Module:
        return LibreYOLO9Model(
            config=self.size, reg_max=self.reg_max, nb_classes=self.nb_classes
        )

    def _get_available_layers(self) -> Dict[str, nn.Module]:
        return {
            # Backbone layers
            "backbone_conv0": self.model.backbone.conv0,
            "backbone_conv1": self.model.backbone.conv1,
            "backbone_elan1": self.model.backbone.elan1,
            "backbone_down2": self.model.backbone.down2,
            "backbone_elan2": self.model.backbone.elan2,
            "backbone_down3": self.model.backbone.down3,
            "backbone_elan3": self.model.backbone.elan3,
            "backbone_down4": self.model.backbone.down4,
            "backbone_elan4": self.model.backbone.elan4,
            "backbone_spp": self.model.backbone.spp,
            # Neck layers
            "neck_elan_up1": self.model.neck.elan_up1,
            "neck_elan_up2": self.model.neck.elan_up2,
            "neck_elan_down1": self.model.neck.elan_down1,
            "neck_elan_down2": self.model.neck.elan_down2,
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
