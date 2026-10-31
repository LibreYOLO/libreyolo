"""LibreAlexNet: factory integration for AlexNet image classification."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from PIL import Image

from ...postprocess.alexnet import postprocess as _alexnet_postprocess
from ...utils.image_loader import ImageInput
from ..base import BaseModel
from .nn import AlexNet
from .utils import preprocess_image as _alexnet_preprocess
from .utils import preprocess_numpy


class LibreAlexNet(BaseModel):
    """AlexNet museum classifier using the single-tower 64-channel stem.

    This is the later "one weird trick" graph released by torchvision, not the
    two-GPU 2012 graph: it has 64 conv1 filters, no local response
    normalization, and no grouped convolutions. The official ImageNet-1K
    checkpoint reports 56.522 percent top-1 accuracy.
    """

    FAMILY = "alexnet"
    FILENAME_PREFIX = "LibreAlexNet"
    INPUT_SIZES = {"b": 224}
    SUPPORTED_TASKS = ("classify",)
    DEFAULT_TASK = "classify"
    REQUIRE_TASK_SUFFIX = True
    TRAIN_CONFIG = None
    CROP_PCT = {"b": 0.875}

    @staticmethod
    def _shape(weights_dict: dict, key: str) -> tuple[int, ...] | None:
        value = weights_dict.get(key)
        shape = getattr(value, "shape", None)
        return tuple(shape) if shape is not None else None

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        """Recognize only the shipped single-tower AlexNet tensor layout."""
        return (
            cls._shape(weights_dict, "features.0.weight") == (64, 3, 11, 11)
            and "classifier.6.weight" in weights_dict
            and cls.detect_size(weights_dict) is not None
        )

    @classmethod
    def detect_size(cls, weights_dict: dict) -> Optional[str]:
        signatures = (
            cls._shape(weights_dict, "features.0.weight") == (64, 3, 11, 11),
            cls._shape(weights_dict, "classifier.1.weight") == (4096, 256 * 6 * 6),
            cls._shape(weights_dict, "classifier.4.weight") == (4096, 4096),
        )
        final_shape = cls._shape(weights_dict, "classifier.6.weight")
        if all(signatures) and final_shape is not None and len(final_shape) == 2:
            if final_shape[1] == 4096 and final_shape[0] > 0:
                return "b"
        return None

    @classmethod
    def detect_nb_classes(cls, weights_dict: dict) -> Optional[int]:
        shape = cls._shape(weights_dict, "classifier.6.weight")
        if shape is None or len(shape) != 2 or shape[0] <= 0:
            return None
        return int(shape[0])

    def __init__(
        self,
        model_path=None,
        size: str = "b",
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
        if isinstance(model_path, (str, Path)):
            self._load_weights(model_path)

    def _init_model(self) -> nn.Module:
        return AlexNet(num_classes=self.nb_classes)

    def _get_available_layers(self) -> Dict[str, nn.Module]:
        return {
            "features": self.model.features,
            "avgpool": self.model.avgpool,
            "classifier": self.model.classifier,
        }

    def _get_preprocess_numpy(self):
        from functools import partial

        return partial(
            preprocess_numpy,
            crop_pct=self.crop_pct,
            interpolation=self.interpolation,
        )

    def _preprocess(
        self,
        image: ImageInput,
        color_format: str = "auto",
        input_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Image.Image, Tuple[int, int], float]:
        effective_size = input_size if input_size is not None else self.input_size
        return _alexnet_preprocess(
            image,
            input_size=effective_size,
            crop_pct=self.crop_pct,
            color_format=color_format,
            interpolation=self.interpolation,
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
        return _alexnet_postprocess(
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
            "AlexNet is shipped as an inference-only museum classifier. "
            "Fine-tuning support is not implemented in LibreYOLO."
        )


__all__ = ["LibreAlexNet"]
