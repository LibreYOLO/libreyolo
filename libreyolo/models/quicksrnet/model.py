"""LibreQuickSRNet compact real-time super-resolution family.

The initial ``m2`` checkpoint is QuickSRNet Medium 2x: a 32-channel CNN with
five intermediate convolutions and pixel-shuffle upsampling. Prediction runs
at native image resolution and returns a canvas twice the input height and
width. Training is outside the initial family scope; paired PSNR/SSIM
validation is provided by LibreYOLO's restore validator.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Dict, Optional, Tuple

import torch
import torch.nn as nn

from ...postprocess.quicksrnet import postprocess as _quicksrnet_postprocess
from ...utils.image_loader import ImageInput
from ..base import BaseModel
from .nn import QuickSRNet
from .utils import preprocess_image, preprocess_numpy


QUICKSRNET_SIZE_CONFIGS: dict[str, dict[str, int]] = {
    "m2": {
        "scale": 2,
        "num_channels": 32,
        "num_intermediate_layers": 5,
    }
}


def _is_quicksrnet_medium_2x_state_dict(state_dict: dict) -> bool:
    """Return whether ``state_dict`` has the exact Medium 2x architecture."""

    expected_shapes = {
        "cnn.0.weight": (32, 3, 3, 3),
        "cnn.0.bias": (32,),
        "cnn.2.weight": (32, 32, 3, 3),
        "cnn.2.bias": (32,),
        "cnn.4.weight": (32, 32, 3, 3),
        "cnn.4.bias": (32,),
        "cnn.6.weight": (32, 32, 3, 3),
        "cnn.6.bias": (32,),
        "cnn.8.weight": (32, 32, 3, 3),
        "cnn.8.bias": (32,),
        "cnn.10.weight": (32, 32, 3, 3),
        "cnn.10.bias": (32,),
        "conv_last.weight": (12, 32, 3, 3),
        "conv_last.bias": (12,),
    }
    if set(state_dict) != set(expected_shapes):
        return False
    return all(
        getattr(state_dict[key], "shape", None) == torch.Size(shape)
        for key, shape in expected_shapes.items()
    )


class LibreQuickSRNet(BaseModel):
    """QuickSRNet Medium 2x RGB super-resolution."""

    FAMILY = "quicksrnet"
    FILENAME_PREFIX = "LibreQuickSRNet"
    INPUT_SIZES: ClassVar[Dict[str, int]] = {"m2": 64}
    SUPPORTED_TASKS = ("restore",)
    DEFAULT_TASK = "restore"
    REQUIRE_TASK_SUFFIX = True
    TRAIN_CONFIG = None
    SUPPORTS_BATCHED_PREDICT = False
    TTA_ENABLED = False

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        return cls.detect_size(weights_dict) is not None

    @classmethod
    def detect_size(cls, weights_dict: dict) -> Optional[str]:
        return "m2" if _is_quicksrnet_medium_2x_state_dict(weights_dict) else None

    @classmethod
    def detect_nb_classes(cls, weights_dict: dict) -> Optional[int]:
        return 1 if cls.can_load(weights_dict) else None

    @classmethod
    def detect_checkpoint_task(cls, state_dict: dict) -> Optional[str]:
        return "restore" if cls.can_load(state_dict) else None

    def __init__(
        self,
        model_path=None,
        size: str = "m2",
        nb_classes: int = 1,
        device: str = "auto",
        task: str | None = None,
        **kwargs,
    ) -> None:
        del nb_classes
        super().__init__(
            model_path=model_path,
            size=size,
            nb_classes=1,
            device=device,
            task=task,
            **kwargs,
        )
        if model_path is not None and isinstance(model_path, (str, Path)):
            self._load_weights(str(model_path))
        self.nb_classes = 1
        self.names = {0: "image"}

    @property
    def restore_scale(self) -> int:
        return int(QUICKSRNET_SIZE_CONFIGS[self.size]["scale"])

    def _init_model(self) -> nn.Module:
        config = QUICKSRNET_SIZE_CONFIGS[self.size]
        return QuickSRNet(
            scale=int(config["scale"]),
            num_channels=int(config["num_channels"]),
            num_intermediate_layers=int(config["num_intermediate_layers"]),
        )

    def _get_available_layers(self) -> Dict[str, nn.Module]:
        return {name: module for name, module in self.model.named_children()}

    @staticmethod
    def _get_preprocess_numpy():
        return preprocess_numpy

    def _preprocess(
        self,
        image: ImageInput,
        color_format: str = "auto",
        input_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Any, Tuple[int, int], float]:
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
        ratio: float = 1.0,
        **kwargs,
    ) -> Dict:
        del conf_thres, iou_thres, max_det, ratio, kwargs
        return {
            "restored": _quicksrnet_postprocess(
                output,
                original_size,
                scale=self.restore_scale,
            )
        }

    def train(self, *args: Any, **kwargs: Any):
        raise NotImplementedError(
            "LibreQuickSRNet currently ships inference and paired PSNR/SSIM "
            "validation only. Training is not implemented for this family."
        )


__all__ = ["LibreQuickSRNet", "QUICKSRNET_SIZE_CONFIGS"]
