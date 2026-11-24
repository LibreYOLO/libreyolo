"""LibreFasterRCNN: wire native Faster R-CNN into the LibreYOLO factory."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn

from ...postprocess.faster_rcnn import postprocess
from ...utils.coco import COCO91_TO_COCO80
from ...utils.image_loader import ImageInput
from ...validation.preprocessors import FasterRCNNValPreprocessor
from ..base import BaseModel
from .nn import LibreFasterRCNNModel
from .utils import preprocess_image
from .validator import FasterRCNNValidator


class LibreFasterRCNN(BaseModel):
    """Modernized torchvision Faster R-CNN variants for object detection."""

    FAMILY = "faster_rcnn"
    FILENAME_PREFIX = "LibreFasterRCNN"
    INPUT_SIZES = {"n": 320, "s": 800, "m": 800, "l": 800}
    SUPPORTED_TASKS = ("detect",)
    DEFAULT_TASK = "detect"
    TRAIN_CONFIG = None
    val_preprocessor_class = FasterRCNNValPreprocessor
    validator_class = FasterRCNNValidator
    SUPPORTS_BATCHED_PREDICT = False

    def __init__(
        self,
        model_path=None,
        size: str = "n",
        nb_classes: int = 80,
        device: str = "auto",
        **kwargs,
    ) -> None:
        super().__init__(
            model_path=model_path,
            size=size,
            nb_classes=nb_classes,
            device=device,
            **kwargs,
        )
        if isinstance(model_path, str):
            self._load_weights(model_path)
        self._arch_num_classes = self.model.num_classes
        self.model.eval()

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        """Claim Faster R-CNN boxes only, excluding mask/keypoint siblings."""
        keys = tuple(weights_dict)
        if any(
            key.startswith(("roi_heads.mask_", "roi_heads.keypoint_"))
            for key in keys
        ):
            return False
        return (
            any(key.startswith("rpn.head.") for key in keys)
            and "roi_heads.box_predictor.cls_score.weight" in weights_dict
            and "roi_heads.box_predictor.bbox_pred.weight" in weights_dict
        )

    @classmethod
    def detect_size(cls, weights_dict: dict) -> Optional[str]:
        """Infer ResNet variants; MobileNet n/s require a filename hint."""
        if not cls.can_load(weights_dict):
            return None
        if "backbone.body.conv1.weight" in weights_dict:
            return "l" if "rpn.head.conv.1.0.weight" in weights_dict else "m"
        return None

    @classmethod
    def detect_size_from_filename(cls, filename: str) -> Optional[str]:
        detected = super().detect_size_from_filename(filename)
        if detected is not None:
            return detected
        lower = Path(filename).name.lower()
        aliases = {
            "fasterrcnn_mobilenet_v3_large_320_fpn": "n",
            "fasterrcnn_mobilenet_v3_large_fpn": "s",
            "fasterrcnn_resnet50_fpn_v2": "l",
            "fasterrcnn_resnet50_fpn": "m",
        }
        for prefix in sorted(aliases, key=len, reverse=True):
            if lower.startswith(prefix):
                return aliases[prefix]
        return None

    @classmethod
    def detect_nb_classes(cls, weights_dict: dict) -> Optional[int]:
        key = "roi_heads.box_predictor.cls_score.weight"
        if key not in weights_dict:
            return None
        head_width = int(weights_dict[key].shape[0])
        return 80 if head_width == 91 else head_width - 1

    def _init_model(self) -> nn.Module:
        head_width = 91 if self.nb_classes == 80 else self.nb_classes + 1
        return LibreFasterRCNNModel(size=self.size, num_classes=head_width)

    def _get_available_layers(self) -> dict[str, nn.Module]:
        return {
            "backbone": self.model.backbone,
            "rpn": self.model.rpn,
            "roi_heads": self.model.roi_heads,
        }

    @staticmethod
    def _get_preprocess_numpy():
        from .utils import preprocess_numpy

        return preprocess_numpy

    def _preprocess(
        self,
        image: ImageInput,
        color_format: str = "auto",
        input_size: Optional[int] = None,
    ):
        del input_size
        return preprocess_image(image, color_format=color_format)

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
    ) -> dict:
        class_map = (
            COCO91_TO_COCO80
            if self._arch_num_classes == 91 and self.nb_classes == 80
            else None
        )
        return postprocess(
            output,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            original_size=original_size,
            max_det=max_det,
            class_map=class_map,
            **kwargs,
        )

    def train(self, *args, **kwargs):
        raise NotImplementedError(
            "Faster R-CNN is currently inference-only; RPN and sampled-RoI "
            "training are not implemented."
        )

    def export(
        self,
        format: str = "onnx",
        *,
        opset: int = 18,
        **kwargs,
    ) -> str:
        if format.lower() == "onnx":
            if int(kwargs.get("batch", 1)) != 1:
                raise NotImplementedError(
                    "Faster R-CNN ONNX export supports batch=1 only."
                )
            if kwargs.get("dynamic") is False:
                warnings.warn(
                    "Faster R-CNN ONNX keeps its upstream resize inside the "
                    "graph; forcing dynamic=True for non-square source parity.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            kwargs["dynamic"] = True
        return super().export(format=format, opset=opset, **kwargs)

    def _strict_loading(self) -> bool:
        return True
