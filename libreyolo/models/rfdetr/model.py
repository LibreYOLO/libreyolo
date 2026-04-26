"""LibreYOLORFDETR implementation for LibreYOLO."""

from pathlib import Path
from typing import Any, ClassVar, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from PIL import Image

from ..base import BaseModel
from ...utils.image_loader import ImageInput, ImageLoader
from ...utils.serialization import load_trusted_torch_file
from .nn import LibreRFDETRModel
from .config import RFDETRConfig
from .utils import postprocess, IMAGENET_MEAN, IMAGENET_STD
from .trainer import train_rfdetr
from ...validation.preprocessors import RFDETRValPreprocessor

# COCO 91-class to 80-class mapping.
# RF-DETR pretrained models output 91 COCO category IDs (1-90),
# but YOLO-format labels use a contiguous 80-class scheme (0-79).
_COCO91_TO_COCO80 = {
    1: 0,
    2: 1,
    3: 2,
    4: 3,
    5: 4,
    6: 5,
    7: 6,
    8: 7,
    9: 8,
    10: 9,
    11: 10,
    13: 11,
    14: 12,
    15: 13,
    16: 14,
    17: 15,
    18: 16,
    19: 17,
    20: 18,
    21: 19,
    22: 20,
    23: 21,
    24: 22,
    25: 23,
    27: 24,
    28: 25,
    31: 26,
    32: 27,
    33: 28,
    34: 29,
    35: 30,
    36: 31,
    37: 32,
    38: 33,
    39: 34,
    40: 35,
    41: 36,
    42: 37,
    43: 38,
    44: 39,
    46: 40,
    47: 41,
    48: 42,
    49: 43,
    50: 44,
    51: 45,
    52: 46,
    53: 47,
    54: 48,
    55: 49,
    56: 50,
    57: 51,
    58: 52,
    59: 53,
    60: 54,
    61: 55,
    62: 56,
    63: 57,
    64: 58,
    65: 59,
    67: 60,
    70: 61,
    72: 62,
    73: 63,
    74: 64,
    75: 65,
    76: 66,
    77: 67,
    78: 68,
    79: 69,
    80: 70,
    81: 71,
    82: 72,
    84: 73,
    85: 74,
    86: 75,
    87: 76,
    88: 77,
    89: 78,
    90: 79,
}


class LibreYOLORFDETR(BaseModel):
    """RF-DETR model for object detection and instance segmentation.

    RF-DETR is a Detection Transformer using DINOv2 backbone with
    multi-scale deformable attention. Segmentation variants add a
    lightweight mask head for instance segmentation.

    Args:
        model_path: Path to weights, pre-loaded state_dict, or None for pretrained.
        size: Model size variant ("n", "s", "m", "l").
        nb_classes: Number of classes (default: 80 for COCO).
        device: Device for inference.

    Example::

        >>> model = LibreYOLORFDETR(size="s")
        >>> detections = model.predict("path/to/image.jpg")
    """

    # Class-level metadata
    FAMILY = "rfdetr"
    FILENAME_PREFIX = "LibreRFDETR"
    INPUT_SIZES = {"n": 384, "s": 512, "m": 576, "l": 704}
    SEG_INPUT_SIZES = {"n": 312, "s": 384, "m": 432, "l": 504}
    TRAIN_CONFIG = RFDETRConfig
    TRAIN_CONFIG = RFDETRConfig
    val_preprocessor_class = RFDETRValPreprocessor

    # CLI parameters not supported by RF-DETR's training API
    UNSUPPORTED_TRAIN_PARAMS: ClassVar[set[str]] = {
        "imgsz",
        "mosaic",
        "mixup",
        "degrees",
        "shear",
        "scheduler",
        "warmup_lr_start",
        "min_lr_ratio",
        "mosaic_scale",
        "mixup_scale",
        "no_aug_epochs",
        "optimizer",
        "momentum",
        "nesterov",
        "hsv_prob",
        "flip_prob",
        "translate",
        "amp",
        "pretrained",
        "log_interval",
    }

    # =========================================================================
    # Registry classmethods
    # =========================================================================

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        keys_lower = [k.lower() for k in weights_dict]
        return any(
            "detr" in k
            or "dinov2" in k
            or "transformer" in k
            or ("encoder" in k and "decoder" in k)
            or "query_embed" in k
            or "class_embed" in k
            or "bbox_embed" in k
            for k in keys_lower
        )

    @classmethod
    def detect_size(
        cls, weights_dict: dict, state_dict: dict | None = None
    ) -> Optional[str]:
        full_ckpt = state_dict if state_dict is not None else weights_dict
        is_seg = any(k.startswith("segmentation_head") for k in weights_dict)

        RESOLUTION_TO_SIZE = {384: "n", 512: "s", 576: "m", 704: "l"}
        SEG_RESOLUTION_TO_SIZE = {312: "n", 384: "s", 432: "m", 504: "l"}
        res_map = SEG_RESOLUTION_TO_SIZE if is_seg else RESOLUTION_TO_SIZE

        args = full_ckpt.get("args")
        if args is not None:
            resolution = (
                getattr(args, "resolution", None)
                if hasattr(args, "resolution")
                else args.get("resolution")
                if isinstance(args, dict)
                else None
            )
            if resolution in res_map:
                return res_map[resolution]

        # Fallback: infer from backbone position_embeddings shape
        pos_key = "backbone.0.encoder.encoder.embeddings.position_embeddings"
        if pos_key in weights_dict:
            pos_tokens = weights_dict[pos_key].shape[1]
            if pos_tokens == 577:
                return "n"
            if pos_tokens == 1025:
                return "s"
            if pos_tokens == 1297:
                return "m"
            return "l"

        return None

    @classmethod
    def detect_nb_classes(cls, weights_dict: dict) -> Optional[int]:
        # RF-DETR class_embed has (num_classes + 1) outputs (includes background)
        if "class_embed.bias" in weights_dict:
            return weights_dict["class_embed.bias"].shape[0] - 1
        return None

    # =========================================================================
    # Initialization
    # =========================================================================

    def __init__(
        self,
        model_path: str | None = None,
        size: str = "s",
        nb_classes: int = 80,
        device: str = "auto",
        segmentation: bool = False,
        **kwargs,
    ):
        # Convert empty dict (from factory) to None for RF-DETR config compatibility
        if isinstance(model_path, dict) and not model_path:
            self._pretrain_weights = None
        elif isinstance(model_path, str):
            self._pretrain_weights = self._resolve_weights_path(model_path)
        else:
            self._pretrain_weights = model_path

        self._is_segmentation = segmentation

        # Auto-detect segmentation from filename first (avoids loading weights twice)
        if not segmentation and self._pretrain_weights is not None:
            task = self.detect_task_from_filename(str(self._pretrain_weights))
            if task == "seg":
                self._is_segmentation = True
            else:
                self._is_segmentation = self._detect_segmentation(
                    self._pretrain_weights
                )

        if self._is_segmentation:
            self.INPUT_SIZES = self.SEG_INPUT_SIZES

        # RF-DETR COCO checkpoints have 90 arch-classes (91 outputs incl.
        # background), but libreyolo uses 80 YOLO-contiguous classes with a
        # 91→80 mapping in _postprocess.  Store the arch count for rfdetr
        # config and present 80 to the rest of the framework.
        self._model_num_classes = nb_classes
        user_nb_classes = 80 if nb_classes == 90 else nb_classes

        super().__init__(
            model_path=None,
            size=size,
            nb_classes=user_nb_classes,
            device=device,
            **kwargs,
        )

        # RF-DETR loads its own weights in _init_model() via pretrain_weights
        if self._pretrain_weights is not None:
            self.model.eval()

    @staticmethod
    def _detect_segmentation(model_path) -> bool:
        """Check if weights contain a segmentation head."""
        if not isinstance(model_path, str):
            return False
        try:
            ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
            state = ckpt.get("model", ckpt)
            return any(k.startswith("segmentation_head") for k in state)
        except Exception:
            return False

    # =========================================================================
    # Model lifecycle
    # =========================================================================

    def _init_model(self) -> nn.Module:
        return LibreRFDETRModel(
            config=self.size,
            nb_classes=self._model_num_classes,
            pretrain_weights=self._pretrain_weights,
            device=str(self.device),
            segmentation=self._is_segmentation,
        )

    def _get_available_layers(self) -> Dict[str, nn.Module]:
        layers = {}
        if hasattr(self.model, "model"):
            actual_model = self.model.model
            if hasattr(actual_model, "backbone"):
                layers["backbone"] = actual_model.backbone
            if hasattr(actual_model, "encoder"):
                layers["encoder"] = actual_model.encoder
            if hasattr(actual_model, "decoder"):
                layers["decoder"] = actual_model.decoder
        return layers

    def _strict_loading(self) -> bool:
        return False

    # =========================================================================
    # Inference pipeline
    # =========================================================================

    @staticmethod
    def _get_preprocess_numpy():
        from .utils import preprocess_numpy

        return preprocess_numpy

    def _preprocess(
        self,
        image: ImageInput,
        color_format: str = "auto",
        input_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Image.Image, Tuple[int, int], float]:
        """Preprocess: resize + ImageNet normalization (no letterbox)."""
        effective_res = input_size if input_size is not None else self.input_size

        img = ImageLoader.load(image, color_format=color_format)
        orig_w, orig_h = img.size
        orig_size = (orig_w, orig_h)

        img_tensor = F.to_tensor(img)
        img_tensor = F.normalize(img_tensor, IMAGENET_MEAN, IMAGENET_STD)
        img_tensor = F.resize(img_tensor, (effective_res, effective_res))
        img_tensor = img_tensor.unsqueeze(0)

        return img_tensor, img, orig_size, 1.0

    def _forward(self, input_tensor: torch.Tensor) -> Any:
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
        num_select = kwargs.get("num_select", max_det)

        # original_size is (width, height); rfdetr postprocess expects (height, width)
        orig_w, orig_h = original_size
        target_sizes = torch.tensor([(orig_h, orig_w)], device=self.device)

        results = postprocess(output, target_sizes, num_select=num_select)

        result = results[0]
        scores = result["scores"]
        labels = result["labels"]
        boxes = result["boxes"]
        masks = result.get("masks")  # (K, H, W) bool or None

        keep = scores > conf_thres
        scores = scores[keep]
        labels = labels[keep]
        boxes = boxes[keep]
        if masks is not None:
            masks = masks[keep]

        # Map COCO 91-class IDs to YOLO 80-class indices if needed
        num_output_classes = output["pred_logits"].shape[-1]
        if num_output_classes == 91 and self.nb_classes == 80:
            mapped = torch.tensor(
                [_COCO91_TO_COCO80.get(int(c), -1) for c in labels.cpu()],
                dtype=labels.dtype,
            )
            valid = mapped >= 0
            boxes = boxes[valid]
            scores = scores[valid]
            labels = mapped[valid]
            if masks is not None:
                masks = masks[valid]

        det = {
            "boxes": boxes.cpu().tolist(),
            "scores": scores.cpu().tolist(),
            "classes": labels.cpu().tolist(),
            "num_detections": len(boxes),
        }
        if masks is not None:
            det["masks"] = masks.cpu()
        return det

    # =========================================================================
    # Public API
    # =========================================================================

    def export(self, format: str = "onnx", *, opset: int = 17, **kwargs) -> str:
        """Export model. RF-DETR requires opset >= 17 for LayerNormalization."""
        return super().export(format, opset=opset, **kwargs)

    def train(
        self,
        data: str,
        epochs: int = 100,
        batch_size: int = 4,
        lr: float = 1e-4,
        output_dir: str = "runs/train",
        resume: str | None = None,
        **kwargs,
    ) -> Dict:
        """Train using the original RF-DETR training implementation.

        Args:
            data: Path to dataset in Roboflow/COCO format.
            epochs: Number of training epochs.
            batch_size: Batch size.
            lr: Learning rate.
            output_dir: Directory to save outputs.
            resume: Path to checkpoint to resume from.
            **kwargs: Additional args passed to rfdetr train().

        Returns:
            Dictionary with training results including output_dir.
        """
        result = train_rfdetr(
            data=data,
            size=self.size,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            output_dir=output_dir,
            resume=resume,
            pretrain_weights=self._pretrain_weights,
            segmentation=self._is_segmentation,
            **kwargs,
        )

        best_ckpt = Path(result["output_dir"]) / "checkpoint_best_total.pth"
        if best_ckpt.exists():
            checkpoint = load_trusted_torch_file(
                best_ckpt,
                map_location="cpu",
                context="RF-DETR best checkpoint reload",
            )
            state_dict = checkpoint["model"]

            # RF-DETR uses num_classes + 1 internally (background class)
            num_classes_internal = state_dict["class_embed.bias"].shape[0]
            num_classes = num_classes_internal - 1

            if num_classes_internal != self.model.model.class_embed.bias.shape[0]:
                self.model.model.reinitialize_detection_head(num_classes_internal)

            self.model.model.load_state_dict(state_dict, strict=False)
            self.model.model.eval()
            self.model.model.to(self.device)

            self.nb_classes = num_classes
            self.model.nb_classes = num_classes
            self.names = {i: f"class_{i}" for i in range(num_classes)}

        return result
