"""LibreFCOS: wire the FCOS family into the LibreYOLO factory."""

from __future__ import annotations

import warnings
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn

from ...postprocess.fcos import postprocess
from ...utils.coco import COCO91_TO_COCO80
from ...utils.image_loader import ImageInput
from ...validation.preprocessors import FCOSValPreprocessor
from ..base import BaseModel
from .nn import LibreFCOSModel
from .utils import preprocess_image, preprocess_numpy
from .validator import FCOSValidator


class LibreFCOS(BaseModel):
    """FCOS ResNet-50/FPN, the landmark anchor-free per-pixel detector."""

    FAMILY = "fcos"
    FILENAME_PREFIX = "LibreFCOS"
    INPUT_SIZES = {"r50": 800}
    SUPPORTED_TASKS = ("detect",)
    DEFAULT_TASK = "detect"
    TRAIN_CONFIG = None
    val_preprocessor_class = FCOSValPreprocessor
    validator_class = FCOSValidator
    SUPPORTS_BATCHED_PREDICT = False

    def __init__(
        self,
        model_path=None,
        size: str = "r50",
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

    def __call__(self, source=None, **kwargs):
        """Use the published FCOS thresholds unless the caller overrides them."""
        kwargs.setdefault("conf", 0.2)
        kwargs.setdefault("iou", 0.6)
        kwargs.setdefault("max_det", 100)
        return super().__call__(source, **kwargs)

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        """Claim FCOS only through its centerness branch and P6/P7 FPN."""
        return (
            "head.regression_head.bbox_ctrness.weight" in weights_dict
            and "head.classification_head.cls_logits.weight" in weights_dict
            and "backbone.fpn.extra_blocks.p6.weight" in weights_dict
            and "backbone.fpn.extra_blocks.p7.weight" in weights_dict
        )

    @classmethod
    def detect_size(cls, weights_dict: dict) -> Optional[str]:
        """Recognize the only permissive pretrained variant, ResNet-50/FPN."""
        if not cls.can_load(weights_dict):
            return None
        stem = weights_dict.get("backbone.body.conv1.weight")
        if stem is not None and tuple(stem.shape) == (64, 3, 7, 7):
            return "r50"
        return None

    @classmethod
    def detect_nb_classes(cls, weights_dict: dict) -> Optional[int]:
        key = "head.classification_head.cls_logits.weight"
        if key not in weights_dict:
            return None
        width = int(weights_dict[key].shape[0])
        return 80 if width == 91 else width

    def _init_model(self) -> nn.Module:
        head_width = 91 if self.nb_classes == 80 else self.nb_classes
        return LibreFCOSModel(num_classes=head_width)

    def _get_available_layers(self) -> dict[str, nn.Module]:
        return {"backbone": self.model.backbone, "head": self.model.head}

    @staticmethod
    def _get_preprocess_numpy():
        return preprocess_numpy

    def _preprocess(
        self,
        image: ImageInput,
        color_format: str = "auto",
        input_size: Optional[int] = None,
    ):
        return preprocess_image(
            image,
            color_format=color_format,
            input_size=int(input_size or self.input_size),
        )

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
            "FCOS is currently inference-only; dense assignment and loss "
            "training are not implemented."
        )

    def export(
        self,
        format: str = "onnx",
        *,
        opset: int | None = None,
        **kwargs,
    ) -> str:
        export_format = format.lower()
        if opset is None and export_format != "axelera":
            # Every validated FCOS format was measured at opset 18; Axelera
            # takes opset 14-17 and chooses its own.
            opset = 18
        if export_format in {"onnx", "openvino"}:
            if kwargs.get("dynamic") is False:
                warnings.warn(
                    "FCOS preserves aspect ratio before the graph; forcing "
                    "dynamic=True so padded source shapes remain valid.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            kwargs["dynamic"] = True
        return super().export(format=format, opset=opset, **kwargs)

    def _strict_loading(self) -> bool:
        return True


__all__ = ["LibreFCOS"]
