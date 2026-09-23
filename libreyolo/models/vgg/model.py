"""LibreVGG: VGG classification family wiring for the LibreYOLO factory."""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from ...postprocess.vgg import postprocess as _vgg_postprocess
from ..base import BaseModel
from ..base.classify_preprocess import ClassifyPreprocessMixin
from .nn import VGG


class LibreVGG(ClassifyPreprocessMixin, BaseModel):
    """Historic VGG-16/VGG-19 ImageNet classifier family.

    VGG established uniform deep stacks of small 3x3 convolutions and became
    a standard feature extractor for early detection and segmentation systems.
    The shipped weights are torchvision's later from-scratch ImageNet recipe,
    not conversions of the Oxford 2014 Caffe release. Batch-normalized variants
    are later extensions rather than configurations from the original report.
    """

    FAMILY = "vgg"
    FILENAME_PREFIX = "LibreVGG"
    INPUT_SIZES = {"16": 224, "19": 224, "16bn": 224, "19bn": 224}
    SUPPORTED_TASKS = ("classify",)
    DEFAULT_TASK = "classify"
    REQUIRE_TASK_SUFFIX = True
    TRAIN_CONFIG = None

    CROP_PCT = {size: 0.875 for size in INPUT_SIZES}

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        stem = weights_dict.get("features.0.weight")
        first_fc = weights_dict.get("classifier.0.weight")
        final_fc = weights_dict.get("classifier.6.weight")
        return (
            isinstance(stem, torch.Tensor)
            and tuple(stem.shape) == (64, 3, 3, 3)
            and isinstance(first_fc, torch.Tensor)
            and tuple(first_fc.shape) == (4096, 512 * 7 * 7)
            and isinstance(final_fc, torch.Tensor)
            and final_fc.ndim == 2
            and final_fc.shape[1] == 4096
            and cls.detect_size(weights_dict) is not None
        )

    @staticmethod
    def _conv_count(weights_dict: dict) -> int:
        pattern = re.compile(r"^features\.\d+\.weight$")
        return sum(
            1
            for key, value in weights_dict.items()
            if pattern.match(key)
            and isinstance(value, torch.Tensor)
            and value.ndim == 4
        )

    @classmethod
    def detect_size(cls, weights_dict: dict) -> Optional[str]:
        stem = weights_dict.get("features.0.weight")
        first_fc = weights_dict.get("classifier.0.weight")
        if (
            not isinstance(stem, torch.Tensor)
            or tuple(stem.shape) != (64, 3, 3, 3)
            or not isinstance(first_fc, torch.Tensor)
            or tuple(first_fc.shape) != (4096, 512 * 7 * 7)
        ):
            return None
        conv_count = cls._conv_count(weights_dict)
        has_batch_norm = any(
            re.match(r"^features\.\d+\.running_var$", key) for key in weights_dict
        )
        return {
            (13, False): "16",
            (16, False): "19",
            (13, True): "16bn",
            (16, True): "19bn",
        }.get((conv_count, has_batch_norm))

    @classmethod
    def detect_nb_classes(cls, weights_dict: dict) -> Optional[int]:
        final_fc = weights_dict.get("classifier.6.weight")
        if not isinstance(final_fc, torch.Tensor) or final_fc.ndim != 2:
            return None
        return int(final_fc.shape[0])

    def __init__(
        self,
        model_path=None,
        size: str = "16",
        nb_classes: int = 1000,
        device: str = "auto",
        task: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            model_path=model_path,
            size=size,
            nb_classes=nb_classes,
            device=device,
            task=task,
            **kwargs,
        )
        self.crop_pct = self.CROP_PCT[self.size]
        self.interpolation = "bilinear"
        if isinstance(model_path, str):
            self._load_weights(model_path)

    def _init_model(self) -> nn.Module:
        return VGG(size=self.size, num_classes=self.nb_classes)

    def _get_available_layers(self) -> Dict[str, nn.Module]:
        return {
            "features": self.model.features,
            "avgpool": self.model.avgpool,
            "classifier": self.model.classifier,
        }

    def _check_eval_imgsz(self, imgsz: int) -> None:
        if int(imgsz) != int(self.input_size):
            raise ValueError(
                "LibreVGG runs at its fixed native resolution "
                f"{self.input_size}x{self.input_size}; got imgsz={imgsz}."
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
    ) -> Dict:
        return _vgg_postprocess(
            output,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            original_size=original_size,
            max_det=max_det,
            ratio=ratio,
            **kwargs,
        )

    def train(self, *args, **kwargs):
        del args, kwargs
        raise NotImplementedError(
            "LibreVGG is shipped as an inference-only classification family. "
            "Classification fine-tuning is not implemented for this family yet."
        )

    def _strict_loading(self) -> bool:
        return True


__all__ = ["LibreVGG"]
