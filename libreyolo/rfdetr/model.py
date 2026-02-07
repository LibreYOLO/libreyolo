"""
LIBREYOLORFDETR implementation for LibreYOLO.

Supports both inference and training with RF-DETR (Detection Transformer).
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from PIL import Image

from ..common.base_model import LibreYOLOBase
from ..common.image_loader import ImageInput, ImageLoader
from .nn import RFDETRModel, RFDETR_CONFIGS
from .utils import postprocess
from .train import train_rfdetr, RFDETR_TRAINERS


# ImageNet normalization constants (same as original rfdetr)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# COCO 91-class to 80-class mapping.
# RF-DETR pretrained models output 91 COCO category IDs (1-90),
# but YOLO-format labels use a contiguous 80-class scheme (0-79).
# This table maps COCO category ID → YOLO class index.
_COCO91_TO_COCO80 = {
    1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9,
    11: 10, 13: 11, 14: 12, 15: 13, 16: 14, 17: 15, 18: 16, 19: 17,
    20: 18, 21: 19, 22: 20, 23: 21, 24: 22, 25: 23, 27: 24, 28: 25,
    31: 26, 32: 27, 33: 28, 34: 29, 35: 30, 36: 31, 37: 32, 38: 33,
    39: 34, 40: 35, 41: 36, 42: 37, 43: 38, 44: 39, 46: 40, 47: 41,
    48: 42, 49: 43, 50: 44, 51: 45, 52: 46, 53: 47, 54: 48, 55: 49,
    56: 50, 57: 51, 58: 52, 59: 53, 60: 54, 61: 55, 62: 56, 63: 57,
    64: 58, 65: 59, 67: 60, 70: 61, 72: 62, 73: 63, 74: 64, 75: 65,
    76: 66, 77: 67, 78: 68, 79: 69, 80: 70, 81: 71, 82: 72, 84: 73,
    85: 74, 86: 75, 87: 76, 88: 77, 89: 78, 90: 79,
}


class LIBREYOLORFDETR(LibreYOLOBase):
    """
    LibreYOLO RF-DETR model for object detection.

    RF-DETR is a Detection Transformer using DINOv2 backbone with
    multi-scale deformable attention for high-quality object detection.

    This implementation is 100% compatible with original rfdetr checkpoints
    and produces identical outputs.

    Args:
        model_path: Model weights source. Can be:
            - str: Path to a .pt/.pth weights file
            - dict: Pre-loaded state_dict (e.g., from torch.load())
            - None: Use default pretrained weights for the size
        size: Model size variant. Must be one of: "n", "s", "b", "m", "l"
        nb_classes: Number of classes (default: 80 for COCO)
        device: Device for inference. "auto" uses CUDA if available.

    Example:
        >>> # Use pretrained COCO weights
        >>> model = LIBREYOLORFDETR(size="b")
        >>> detections = model.predict("path/to/image.jpg")

        >>> # Use custom weights
        >>> model = LIBREYOLORFDETR(model_path="custom_weights.pth", size="b")
        >>> detections = model.predict("path/to/image.jpg", conf_thres=0.5)
    """

    def __init__(
        self,
        model_path: str = None,
        size: str = "b",
        nb_classes: int = 80,
        device: str = "auto",
        **kwargs,
    ):
        # Get resolution from config before calling super().__init__
        config_cls = RFDETR_CONFIGS[size]
        self.resolution = config_cls().resolution

        # Store model_path for later use
        # Convert empty dict (from factory) to None for RF-DETR config compatibility
        if isinstance(model_path, dict) and not model_path:
            self._pretrain_weights = None
        else:
            self._pretrain_weights = model_path

        # Pass special marker for LibreYOLOBase to skip weight loading
        super().__init__(
            model_path="__skip_loading__",  # Special marker to skip _load_weights
            size=size,
            nb_classes=nb_classes,
            device=device,
            **kwargs,
        )

    def _get_valid_sizes(self) -> List[str]:
        return ["n", "s", "b", "m", "l"]

    def _get_model_name(self) -> str:
        return "LIBREYOLORFDETR"

    def _get_input_size(self) -> int:
        return self.resolution

    def _init_model(self) -> nn.Module:
        """Initialize RF-DETR model."""
        return RFDETRModel(
            config=self.size,
            nb_classes=self.nb_classes,
            pretrain_weights=self._pretrain_weights,
            device=str(self.device),
        )

    def _get_available_layers(self) -> Dict[str, nn.Module]:
        """Return available layers from RF-DETR model."""
        # RF-DETR has backbone, encoder, decoder structure
        layers = {}
        if hasattr(self.model, 'model'):
            actual_model = self.model.model
            if hasattr(actual_model, 'backbone'):
                layers['backbone'] = actual_model.backbone
            if hasattr(actual_model, 'encoder'):
                layers['encoder'] = actual_model.encoder
            if hasattr(actual_model, 'decoder'):
                layers['decoder'] = actual_model.decoder
        return layers

    def _get_val_preprocessor(self, img_size: int = None):
        """
        Return RF-DETR specific validation preprocessor.

        RF-DETR requires ImageNet normalization, unlike YOLO models
        which use simple 0-1 normalization.

        Args:
            img_size: Target image size. Defaults to model's native input size.

        Returns:
            RFDETRValPreprocessor instance.
        """
        from libreyolo.validation.preprocessors import RFDETRValPreprocessor
        if img_size is None:
            img_size = self._get_input_size()
        return RFDETRValPreprocessor(img_size=(img_size, img_size))

    def _load_weights(self, model_path: str):
        """Override to handle RF-DETR checkpoint format."""
        # Skip loading if special marker (weights loaded in _init_model via rfdetr)
        if model_path == "__skip_loading__":
            return

        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model weights file not found: {model_path}")

        try:
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

            # Handle different checkpoint formats
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

        except Exception as e:
            raise RuntimeError(
                f"Failed to load model weights from {model_path}: {e}"
            ) from e

    def _preprocess(
        self, image: ImageInput, color_format: str = "auto", input_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Image.Image, Tuple[int, int]]:
        """
        Preprocess image for inference.

        Matches original rfdetr preprocessing exactly:
        1. Convert to tensor [0, 1]
        2. Normalize with ImageNet mean/std
        3. Resize to model resolution (direct resize, no letterbox)

        Args:
            image: Input image in various formats
            color_format: Color format hint (unused, kept for compatibility)
            input_size: Override input resolution (None = model default)

        Returns:
            Tuple of (input_tensor, original_image, original_size)
            - input_tensor: (1, 3, H, W) normalized tensor
            - original_image: Original PIL image
            - original_size: (width, height) of original image
        """
        effective_res = input_size if input_size is not None else self.resolution

        # Load image to PIL
        img = ImageLoader.load(image, color_format=color_format)

        # Get original size (width, height) — normalized convention across all models
        orig_w, orig_h = img.size  # PIL size is (W, H)
        orig_size = (orig_w, orig_h)

        # Convert to tensor [0, 1]
        img_tensor = F.to_tensor(img)

        # Normalize with ImageNet mean/std
        img_tensor = F.normalize(img_tensor, IMAGENET_MEAN, IMAGENET_STD)

        # Resize to model resolution (direct resize, no letterbox)
        img_tensor = F.resize(img_tensor, (effective_res, effective_res))

        # Add batch dimension
        img_tensor = img_tensor.unsqueeze(0)

        return img_tensor, img, orig_size

    def _forward(self, input_tensor: torch.Tensor) -> Any:
        """Run model forward pass."""
        return self.model(input_tensor)

    def _postprocess(
        self,
        output: Any,
        conf_thres: float,
        iou_thres: float,
        original_size: Tuple[int, int],
        max_det: int = 300,
        **kwargs,
    ) -> Dict:
        """
        Postprocess RF-DETR output to detections.

        RF-DETR uses top-K selection (no NMS) unlike YOLO models.

        Args:
            output: Model output dictionary with 'pred_logits' and 'pred_boxes'
            conf_thres: Confidence threshold for filtering detections
            iou_thres: IoU threshold (unused for RF-DETR, kept for compatibility)
            original_size: (width, height) of original image
            max_det: Maximum number of detections
            **kwargs: Additional arguments (e.g., num_select)

        Returns:
            Dictionary with:
                - boxes: List of [x1, y1, x2, y2] in original image coordinates
                - scores: List of confidence scores
                - classes: List of class IDs (0-indexed)
                - num_detections: Number of detections
        """
        num_select = kwargs.get('num_select', max_det)

        # original_size is now (width, height); rfdetr postprocess expects (height, width)
        orig_w, orig_h = original_size
        target_sizes = torch.tensor([(orig_h, orig_w)], device=self.device)

        # Postprocess (matches original rfdetr exactly)
        results = postprocess(output, target_sizes, num_select=num_select)

        # Extract first (and only) result
        result = results[0]
        scores = result['scores']
        labels = result['labels']
        boxes = result['boxes']

        # Filter by confidence threshold
        keep = scores > conf_thres
        scores = scores[keep]
        labels = labels[keep]
        boxes = boxes[keep]

        # Map COCO 91-class IDs to YOLO 80-class indices.
        # RF-DETR pretrained COCO models output 91 category IDs (1-90),
        # but LibreYOLO uses contiguous 0-79 class indices (YOLO convention).
        # Check the actual output dimension to decide if mapping is needed.
        num_output_classes = output['pred_logits'].shape[-1]
        if num_output_classes == 91 and self.nb_classes == 80:
            mapped = torch.tensor(
                [_COCO91_TO_COCO80.get(int(c), -1) for c in labels.cpu()],
                dtype=labels.dtype,
            )
            valid = mapped >= 0
            boxes = boxes[valid]
            scores = scores[valid]
            labels = mapped[valid]

        return {
            "boxes": boxes.cpu().tolist(),
            "scores": scores.cpu().tolist(),
            "classes": labels.cpu().tolist(),
            "num_detections": len(boxes),
        }

    def train(
        self,
        data: str,
        epochs: int = 100,
        batch_size: int = 4,
        lr: float = 1e-4,
        output_dir: str = "runs/train",
        resume: str = None,
        **kwargs
    ) -> Dict:
        """
        Train the model using the original RF-DETR training implementation.

        This wraps the official rfdetr training API which includes:
        - EMA (Exponential Moving Average)
        - Proper warmup and cosine LR schedule
        - Hungarian matching loss
        - Distributed training support

        Args:
            data: Path to dataset in Roboflow format (COCO annotations).
                  Structure should be:
                    dataset/
                        train/
                            _annotations.coco.json
                            images...
                        valid/
                            _annotations.coco.json
                            images...
                        test/  (optional)
                            _annotations.coco.json
                            images...
            epochs: Number of training epochs (default: 100)
            batch_size: Batch size (default: 4)
            lr: Learning rate (default: 1e-4)
            output_dir: Directory to save outputs (default: "runs/train")
            resume: Path to checkpoint to resume from (default: None)
            **kwargs: Additional args passed to rfdetr train()
                     See rfdetr.config.TrainConfig for all options.

        Returns:
            Dictionary with training results including output_dir and model

        Example:
            >>> model = LIBREYOLORFDETR(size="b")
            >>> # Download a dataset from Roboflow in COCO format
            >>> results = model.train(data="path/to/dataset", epochs=50)
        """
        return train_rfdetr(
            data=data,
            size=self.size,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            output_dir=output_dir,
            resume=resume,
            **kwargs
        )
