"""
LibreYOLO-RD (Regional Diversity) inference wrapper.

Provides a high-level API for YOLO-RD object detection inference.
YOLO-RD extends YOLOv9-c with DConv for enhanced regional feature diversity.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import torch
import torch.nn as nn
from PIL import Image

from ..common.base_model import LibreYOLOBase
from ..common.image_loader import ImageInput
from ..common.utils import preprocess_image
from .nn import LibreYOLORDModel
from .utils import postprocess


class LIBREYOLORD(LibreYOLOBase):
    """
    LibreYOLO-RD (Regional Diversity) model for object detection.

    Based on YOLOv9-c architecture with DConv at B3 position for
    enhanced regional feature diversity.

    Args:
        model_path: Model weights source. Can be:
            - str: Path to a .pt/.pth weights file
            - dict: Pre-loaded state_dict (e.g., from torch.load())
        size: Model size variant (required). Must be "c" (only variant supported)
        atoms: DConv atoms parameter (512 for rd-9c, 4096 for rd-9c-4096).
            If "auto", attempts to detect from weights.
        reg_max: Regression max value for DFL (default: 16)
        nb_classes: Number of classes (default: 80 for COCO)
        device: Device for inference

    Example:
        >>> model = LIBREYOLORD(model_path="path/to/weights.pt", size="c")
        >>> detections = model(image=image_path, save=True)
        >>> # For 4096 variant:
        >>> model_4096 = LIBREYOLORD("path/to/rd_4096.pt", size="c", atoms=4096)
    """

    def __init__(
        self,
        model_path: Union[str, dict],
        size: str = "c",
        atoms: Union[int, str] = "auto",
        reg_max: int = 16,
        nb_classes: int = 80,
        device: str = "auto",
        **kwargs,
    ):
        self.reg_max = reg_max
        self._atoms_arg = atoms
        self._model_path_arg = model_path

        # We need to detect atoms before calling super().__init__
        # Load state_dict first if atoms="auto"
        if isinstance(model_path, dict):
            self._state_dict = model_path
        else:
            if not Path(model_path).exists():
                raise FileNotFoundError(f"Model weights file not found: {model_path}")
            self._state_dict = torch.load(model_path, map_location="cpu", weights_only=False)

        # Detect atoms from weights if auto
        if atoms == "auto":
            atoms = self._detect_atoms(self._state_dict)
        self.atoms = atoms

        super().__init__(
            model_path=model_path,
            size=size,
            nb_classes=nb_classes,
            device=device,
            **kwargs,
        )

    def _detect_atoms(self, state_dict: dict) -> int:
        """Detect atoms from weights by checking DConv layer shapes."""
        cg_key = "backbone.elan2.dconv.CG.conv.weight"
        if cg_key in state_dict:
            return state_dict[cg_key].shape[0]
        return 512

    def _load_weights(self, model_path: str):
        """Override to use pre-loaded state_dict."""
        try:
            self.model.load_state_dict(self._state_dict, strict=True)
        except Exception as e:
            raise RuntimeError(f"Failed to load model weights: {e}") from e

    def _get_valid_sizes(self) -> List[str]:
        return ["c"]

    def _get_model_name(self) -> str:
        return "LIBREYOLORD"

    def _get_input_size(self) -> int:
        return 640

    def _init_model(self) -> nn.Module:
        return LibreYOLORDModel(
            config=self.size,
            reg_max=self.reg_max,
            nb_classes=self.nb_classes,
            atoms=self.atoms,
        )

    def _get_available_layers(self) -> Dict[str, nn.Module]:
        return {
            # Backbone layers
            "backbone_conv0": self.model.backbone.conv0,
            "backbone_conv1": self.model.backbone.conv1,
            "backbone_elan1": self.model.backbone.elan1,
            "backbone_down2": self.model.backbone.down2,
            "backbone_elan2": self.model.backbone.elan2,  # Contains DConv
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
