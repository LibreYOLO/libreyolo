"""
RT-DETR Model wrapper for LibreYOLO.

Provides the LIBREYOLORTDETR class with the same API as other LibreYOLO models.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from PIL import Image

from ..common.base_model import LibreYOLOBase
from ..common.image_loader import ImageInput
from .nn import RTDETRModel
from .utils import preprocess_image


class LIBREYOLORTDETR(LibreYOLOBase):
    """
    RT-DETR model for object detection.

    Provides the same API as other LibreYOLO models (LIBREYOLO8, LIBREYOLO11, etc.)
    but uses the RT-DETR transformer-based architecture.

    Args:
        model_path: Model weights source. Can be:
            - str: Path to a .pt/.pth weights file
            - dict: Pre-loaded state_dict (e.g., from torch.load())
        size: Model size variant. Must be one of: "s", "ms", "m", "l", "x"
        nb_classes: Number of classes (default: 80 for COCO)
        device: Device for inference.

    Example:
        >>> model = LIBREYOLORTDETR("rtdetrv2_r50vd.pth", size="l")
        >>> results = model("image.jpg", save=True)
        >>> print(f"Found {results['num_detections']} objects")
    """

    def __init__(
        self,
        model_path: Union[str, dict],
        size: str,
        nb_classes: int = 80,
        device: str = "auto",
        **kwargs,
    ):
        self._original_size = None  # Store for postprocessing
        super().__init__(
            model_path=model_path,
            size=size,
            nb_classes=nb_classes,
            device=device,
            **kwargs,
        )
        # Put model in deploy mode after loading
        self.model.deploy()

    def _get_valid_sizes(self) -> List[str]:
        return ["s", "ms", "m", "l", "x"]

    def _get_model_name(self) -> str:
        return "LIBREYOLORTDETR"

    def _get_input_size(self) -> int:
        return 640

    def _init_model(self) -> nn.Module:
        return RTDETRModel(config=self.size, nb_classes=self.nb_classes)

    def _get_available_layers(self) -> Dict[str, nn.Module]:
        return {
            "backbone": self.model.backbone,
            "encoder": self.model.encoder,
            "decoder": self.model.decoder,
        }

    def _load_weights(self, model_path: str):
        """Override to handle RT-DETR checkpoint formats."""
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model weights file not found: {model_path}")

        try:
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
            self._load_state_dict(checkpoint)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load model weights from {model_path}: {e}"
            ) from e

    def _load_state_dict(self, checkpoint: dict):
        """Load state dict handling various RT-DETR checkpoint formats."""
        if "ema" in checkpoint and isinstance(checkpoint["ema"], dict):
            if "module" in checkpoint["ema"]:
                state_dict = checkpoint["ema"]["module"]
            else:
                state_dict = checkpoint["ema"]
        elif "model" in checkpoint and isinstance(checkpoint["model"], dict):
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        try:
            self.model.load_state_dict(state_dict, strict=True)
        except RuntimeError:
            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"Warning: Missing keys in state_dict: {len(missing)} keys")
            if unexpected:
                print(f"Warning: Unexpected keys in state_dict: {len(unexpected)} keys")

    def _preprocess(
        self, image: ImageInput, color_format: str = "auto"
    ) -> Tuple[torch.Tensor, Image.Image, Tuple[int, int]]:
        """RT-DETR uses ImageNet normalization."""
        tensor, orig_img, orig_size = preprocess_image(
            image, input_size=640, color_format=color_format
        )
        # Store original size for forward pass
        self._original_size = orig_size
        return tensor, orig_img, orig_size

    def _forward(self, input_tensor: torch.Tensor) -> Any:
        """RT-DETR forward pass needs original size tensor."""
        orig_size_tensor = torch.tensor(
            [[self._original_size[0], self._original_size[1]]],
            dtype=torch.float32,
            device=self.device,
        )
        return self.model(input_tensor, orig_size_tensor)

    def _postprocess(
        self,
        output: Any,
        conf_thres: float,
        iou_thres: float,
        original_size: Tuple[int, int],
        **kwargs,
    ) -> Dict:
        """RT-DETR doesn't need NMS - output is already filtered."""
        labels, boxes, scores = output

        # Filter by confidence threshold
        scores_np = scores[0].cpu()
        mask = scores_np > conf_thres

        filtered_boxes = boxes[0][mask].cpu().tolist()
        filtered_scores = scores_np[mask].tolist()
        filtered_classes = labels[0][mask].cpu().tolist()

        return {
            "boxes": filtered_boxes,
            "scores": filtered_scores,
            "classes": [int(c) for c in filtered_classes],
            "num_detections": len(filtered_boxes),
        }

    @staticmethod
    def get_available_sizes() -> List[str]:
        """
        Get list of available model sizes.

        Returns:
            List of size codes matching official RT-DETRv2 naming:
            - 's':  RT-DETRv2-S  (r18vd,  48.1 AP)
            - 'ms': RT-DETRv2-M* (r34vd,  49.9 AP)
            - 'm':  RT-DETRv2-M  (r50vd_m, 51.9 AP)
            - 'l':  RT-DETRv2-L  (r50vd,  53.4 AP)
            - 'x':  RT-DETRv2-X  (r101vd, 54.3 AP)
        """
        return ["s", "ms", "m", "l", "x"]
