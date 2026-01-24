"""
LIBREYOLOX implementation for LibreYOLO.

Supports both inference and training.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from PIL import Image

from ..common.base_model import LibreYOLOBase
from ..common.image_loader import ImageInput
from .nn import YOLOXModel
from .utils import preprocess_image, postprocess


class LIBREYOLOX(LibreYOLOBase):
    """
    YOLOX model for object detection.

    Args:
        model_path: Model weights source. Can be:
            - str: Path to a .pt/.pth weights file
            - dict: Pre-loaded state_dict (e.g., from torch.load())
        size: Model size variant (required). Must be one of: "nano", "tiny", "s", "m", "l", "x"
        nb_classes: Number of classes (default: 80 for COCO)
        device: Device for inference.

    Example:
        >>> model = LIBREYOLOX(model_path="libreyoloXs.pt", size="s")
        >>> detections = model(image="image.jpg", save=True)
    """

    # Default input sizes for different model variants
    DEFAULT_INPUT_SIZES = {
        "nano": 416,
        "tiny": 416,
        "s": 640,
        "m": 640,
        "l": 640,
        "x": 640,
    }

    def __init__(
        self,
        model_path,
        size: str,
        nb_classes: int = 80,
        device: str = "auto",
        **kwargs,
    ):
        self.input_size = self.DEFAULT_INPUT_SIZES.get(size, 640)
        super().__init__(
            model_path=model_path,
            size=size,
            nb_classes=nb_classes,
            device=device,
            **kwargs,
        )

    def _get_valid_sizes(self) -> List[str]:
        return ["nano", "tiny", "s", "m", "l", "x"]

    def _get_model_name(self) -> str:
        return "LIBREYOLOX"

    def _get_input_size(self) -> int:
        return self.input_size

    def _init_model(self) -> nn.Module:
        return YOLOXModel(config=self.size, nb_classes=self.nb_classes)

    def _get_available_layers(self) -> Dict[str, nn.Module]:
        return {
            "backbone_stem": self.model.backbone.stem,
            "backbone_dark2": self.model.backbone.dark2,
            "backbone_dark3": self.model.backbone.dark3,
            "backbone_dark4": self.model.backbone.dark4,
            "backbone_dark5": self.model.backbone.dark5,
        }

    def _get_val_preprocessor(self, img_size: int = None):
        """YOLOX uses letterbox + no normalization (0-255 range)."""
        from libreyolo.validation.preprocessors import YOLOXValPreprocessor
        if img_size is None:
            img_size = self.input_size
        return YOLOXValPreprocessor(img_size=(img_size, img_size))

    def _load_weights(self, model_path: str):
        """Override to handle different checkpoint formats."""
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model weights file not found: {model_path}")

        try:
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

            if isinstance(checkpoint, dict):
                if "model" in checkpoint:
                    state_dict = checkpoint["model"]
                elif "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint

            self.model.load_state_dict(state_dict, strict=False)

            # Apply nano-specific BN initialization (matching official YOLOX)
            if self.size == 'nano':
                for m in self.model.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.eps = 1e-3
                        m.momentum = 0.03

        except Exception as e:
            raise RuntimeError(
                f"Failed to load model weights from {model_path}: {e}"
            ) from e

    def _preprocess(
        self, image: ImageInput, color_format: str = "auto"
    ) -> Tuple[torch.Tensor, Image.Image, Tuple[int, int]]:
        """YOLOX preprocessing with letterbox - stores ratio for postprocessing."""
        tensor, orig_img, orig_size, ratio = preprocess_image(
            image, input_size=self.input_size, color_format=color_format
        )
        # Store ratio for use in _postprocess
        self._current_ratio = ratio
        return tensor, orig_img, orig_size

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
        # Compute ratio from original_size if not set during _preprocess
        # This handles batch validation where _preprocess is not called per-image
        ratio = getattr(self, "_current_ratio", None)

        # Use passed input_size if available (from validator), otherwise use model's default
        # This is important when validation uses a different size than model's native size
        actual_input_size = kwargs.get('input_size', self.input_size)

        if ratio is None and original_size is not None:
            orig_w, orig_h = original_size
            # Use actual input size for ratio calculation
            ratio = min(actual_input_size / orig_h, actual_input_size / orig_w)
        elif ratio is None:
            ratio = 1.0
        return postprocess(
            output,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            input_size=actual_input_size,
            original_size=original_size,
            ratio=ratio,
        )

    @classmethod
    def new(
        cls,
        size: str = "s",
        num_classes: int = 80,
        device: str = "auto",
    ) -> "LIBREYOLOX":
        """
        Create a new untrained YOLOX model.

        Args:
            size: Model size variant ("nano", "tiny", "s", "m", "l", "x")
            num_classes: Number of classes (default: 80)
            device: Device for the model

        Returns:
            LIBREYOLOX instance with randomly initialized weights
        """
        if size not in ["nano", "tiny", "s", "m", "l", "x"]:
            raise ValueError(
                f"Invalid size: {size}. Must be one of: 'nano', 'tiny', 's', 'm', 'l', 'x'"
            )

        instance = cls.__new__(cls)

        if device == "auto":
            if torch.cuda.is_available():
                instance.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                instance.device = torch.device("mps")
            else:
                instance.device = torch.device("cpu")
        else:
            instance.device = torch.device(device)

        instance.size = size
        instance.nb_classes = num_classes
        instance.input_size = cls.DEFAULT_INPUT_SIZES[size]
        instance.model_path = None

        instance.model = YOLOXModel(config=size, nb_classes=num_classes)
        instance.model.to(instance.device)
        instance.model.train()

        return instance

    def export(
        self,
        format: str = "onnx",
        output_path: Optional[str] = None,
        opset: int = 11,
        simplify: bool = True,
        dynamic: bool = False,
    ) -> str:
        """
        Export the model to a different format.

        Args:
            format: Export format ("onnx", "torchscript")
            output_path: Output file path (auto-generated if None)
            opset: ONNX opset version (default: 11)
            simplify: Simplify ONNX model (default: True)
            dynamic: Use dynamic input shapes (default: False)

        Returns:
            Path to the exported model file
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"yolox_{self.size}_{timestamp}.{format}"

        self.model.eval()

        if format == "onnx":
            return self._export_onnx(output_path, opset, simplify, dynamic)
        elif format == "torchscript":
            return self._export_torchscript(output_path)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def _export_onnx(
        self,
        output_path: str,
        opset: int = 11,
        simplify: bool = True,
        dynamic: bool = False,
    ) -> str:
        """Export to ONNX format."""
        dummy_input = torch.randn(1, 3, self.input_size, self.input_size).to(
            self.device
        )

        dynamic_axes = None
        if dynamic:
            dynamic_axes = {
                "images": {0: "batch", 2: "height", 3: "width"},
                "outputs": {0: "batch"},
            }

        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            input_names=["images"],
            output_names=["outputs"],
            opset_version=opset,
            dynamic_axes=dynamic_axes,
        )

        if simplify:
            try:
                import onnx
                from onnxsim import simplify as onnx_simplify

                model = onnx.load(output_path)
                model_simplified, check = onnx_simplify(model)
                if check:
                    onnx.save(model_simplified, output_path)
            except ImportError:
                pass

        return output_path

    def _export_torchscript(self, output_path: str) -> str:
        """Export to TorchScript format."""
        dummy_input = torch.randn(1, 3, self.input_size, self.input_size).to(
            self.device
        )
        traced_model = torch.jit.trace(self.model, dummy_input)
        traced_model.save(output_path)
        return output_path
