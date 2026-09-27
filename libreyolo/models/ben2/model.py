"""LibreBEN2: efficient background removal using the BEN2 Base network."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Dict, Optional, Tuple

import torch
import torch.nn as nn

from ...postprocess.ben2 import postprocess as _ben2_postprocess
from ...utils.image_loader import ImageInput
from ..base import BaseModel
from .nn import LibreBEN2Model
from .utils import preprocess_image, preprocess_numpy


class LibreBEN2(BaseModel):
    """BEN2 Base background removal: image to soft alpha matte."""

    FAMILY = "ben2"
    FILENAME_PREFIX = "LibreBEN2"
    WEIGHT_EXT = ".pt"
    INPUT_SIZES: ClassVar[Dict[str, int]] = {"b": 1024}
    SUPPORTED_TASKS = ("matte",)
    DEFAULT_TASK = "matte"
    REQUIRE_TASK_SUFFIX = True
    TRAIN_CONFIG = None
    SUPPORTS_BATCHED_PREDICT = True
    TTA_ENABLED = False

    _UPSTREAM_URL = "https://github.com/PramaLLC/BEN2"
    _MARKERS = (
        "multifieldcrossatt.attention.4.out_proj.weight",
        "dec_blk4.sal_conv.weight",
        "insmask_head.6.weight",
    )

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        return cls.detect_size(weights_dict) == "b"

    @classmethod
    def detect_size(cls, weights_dict: dict) -> Optional[str]:
        projection = weights_dict.get("backbone.patch_embed.proj.weight")
        if projection is None or getattr(projection, "ndim", 0) != 4:
            return None
        if tuple(projection.shape[:2]) != (128, 3):
            return None
        return "b" if all(marker in weights_dict for marker in cls._MARKERS) else None

    @classmethod
    def detect_nb_classes(cls, weights_dict: dict) -> Optional[int]:
        return 1 if cls.can_load(weights_dict) else None

    @classmethod
    def detect_checkpoint_task(cls, state_dict: dict) -> Optional[str]:
        return "matte" if cls.can_load(state_dict) else None

    def __init__(
        self,
        model_path=None,
        size: str = "b",
        nb_classes: int = 1,
        device: str = "auto",
        task: str | None = None,
        **kwargs,
    ) -> None:
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
        self.names = {0: "matte"}
        self.model.eval()

    def _init_model(self) -> nn.Module:
        return LibreBEN2Model()

    def _get_available_layers(self) -> Dict[str, nn.Module]:
        return {
            "backbone": self.model.backbone,
            "cross_attention": self.model.multifieldcrossatt,
            "decoder4": self.model.dec_blk4,
            "decoder3": self.model.dec_blk3,
            "decoder2": self.model.dec_blk2,
            "decoder1": self.model.dec_blk1,
            "head": self.model.output,
        }

    @staticmethod
    def _get_preprocess_numpy():
        return preprocess_numpy

    def _preprocess(
        self,
        image: ImageInput,
        color_format: str = "auto",
        input_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Any, Tuple[int, int], float]:
        resolution = self.input_size if input_size is None else input_size
        return preprocess_image(image, input_size=resolution, color_format=color_format)

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
        return _ben2_postprocess(output, original_size)

    def train(self, *args, **kwargs):
        raise NotImplementedError(
            "Training/fine-tuning LibreBEN2 is not wired in this release. "
            "The public BEN2 repository exposes the Base inference graph but "
            "not a complete training recipe. Use the upstream project at "
            f"{self._UPSTREAM_URL} for upstream-supported workflows."
        )


__all__ = ["LibreBEN2"]
