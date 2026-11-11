"""LibreRetinaNet: wire RetinaNet into the LibreYOLO factory."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Optional, Tuple

import torch
from torch import nn

from ...postprocess.retinanet import postprocess
from ...utils.image_loader import ImageInput
from ...validation.preprocessors import RetinaNetValPreprocessor
from ..base import BaseModel
from .nn import LibreRetinaNetModel
from .utils import preprocess_image
from .validator import RetinaNetValidator


class LibreRetinaNet(BaseModel):
    """RetinaNet ResNet-50-FPN v1 and v2 detection variants.

    RetinaNet made one-stage detection competitive with two-stage detectors by
    introducing focal loss for extreme foreground/background imbalance.
    """

    FAMILY = "retinanet"
    FILENAME_PREFIX = "LibreRetinaNet"
    INPUT_SIZES = {"r50": 800, "r50v2": 800}
    SUPPORTED_TASKS = ("detect",)
    DEFAULT_TASK = "detect"
    TRAIN_CONFIG = None
    val_preprocessor_class = RetinaNetValPreprocessor
    validator_class = RetinaNetValidator
    SUPPORTS_BATCHED_PREDICT = False
    TTA_ENABLED = False

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

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        """Claim RetinaNet dense heads while rejecting FCOS-like siblings."""
        keys = tuple(weights_dict)
        return (
            "head.classification_head.cls_logits.weight" in weights_dict
            and "head.regression_head.bbox_reg.weight" in weights_dict
            and not any("bbox_ctrness" in key for key in keys)
        )

    @classmethod
    def detect_size(cls, weights_dict: dict) -> Optional[str]:
        """Distinguish v1 and v2 from the P6 input width and GroupNorm head."""
        if not cls.can_load(weights_dict):
            return None
        p6 = weights_dict.get("backbone.fpn.extra_blocks.p6.weight")
        if p6 is not None:
            in_channels = int(p6.shape[1])
            if in_channels == 2048:
                return "r50v2"
            if in_channels == 256:
                return "r50"
        has_group_norm = any(
            key.startswith("head.classification_head.conv.") and ".1.weight" in key
            for key in weights_dict
        )
        return "r50v2" if has_group_norm else "r50"

    @classmethod
    def detect_size_from_filename(cls, filename: str) -> Optional[str]:
        """Resolve multi-character canonical and torchvision variant names."""
        detected = super().detect_size_from_filename(filename)
        if detected is not None:
            return detected
        lower = Path(filename).name.lower()
        aliases = {
            "retinanet_resnet50_fpn_v2": "r50v2",
            "retinanet_resnet50_fpn": "r50",
        }
        for prefix in sorted(aliases, key=len, reverse=True):
            if lower.startswith(prefix):
                return aliases[prefix]
        return None

    @classmethod
    def detect_nb_classes(cls, weights_dict: dict) -> Optional[int]:
        key = "head.classification_head.cls_logits.weight"
        if key not in weights_dict:
            return None
        out_channels = int(weights_dict[key].shape[0])
        if out_channels <= 0 or out_channels % 9:
            return None
        classes = out_channels // 9
        return 80 if classes == 91 else classes

    def _init_model(self) -> nn.Module:
        head_width = 91 if self.nb_classes == 80 else self.nb_classes
        return LibreRetinaNetModel(size=self.size, num_classes=head_width)

    def _get_available_layers(self) -> dict[str, nn.Module]:
        return {"backbone": self.model.backbone, "head": self.model.head}

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
        return preprocess_image(
            image,
            input_size=input_size or self.input_size,
            color_format=color_format,
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
        ratio: float = 1.0,
        **kwargs,
    ) -> dict:
        return postprocess(
            output,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            original_size=original_size,
            max_det=max_det,
            ratio=ratio,
            **kwargs,
        )

    def train(self, *args, **kwargs):
        raise NotImplementedError(
            "RetinaNet is currently inference-only; focal-loss training is "
            "not implemented."
        )

    def export(
        self,
        format: str = "onnx",
        *,
        opset: int | None = None,
        **kwargs,
    ) -> str:
        if format.lower() == "onnx":
            # The ONNX parity evidence is at opset 13; other formats choose their own.
            opset = 13 if opset is None else opset
            if int(kwargs.get("batch", 1)) != 1:
                raise NotImplementedError(
                    "RetinaNet ONNX export supports batch=1 only."
                )
            if kwargs.get("dynamic") is False:
                warnings.warn(
                    "RetinaNet uses variable aspect-preserved inputs; forcing "
                    "dynamic=True for source-shape parity.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            kwargs["dynamic"] = True
        return super().export(format=format, opset=opset, **kwargs)

    def _strict_loading(self) -> bool:
        return True


__all__ = ["LibreRetinaNet"]
