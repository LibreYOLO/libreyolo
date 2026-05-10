"""LibreRTMDet: BaseModel subclass wiring RTMDet into the LibreYOLO factory.

Cleanroom port of RTMDet (Lyu et al., 2022) from open-mmlab/mmdetection
(Apache-2.0). Sizes: t / s / m / l / x. Detection-only in the first PR;
RTMDet-Ins (segmentation) and trainer land as follow-ups.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from PIL import Image

from ...utils.image_loader import ImageInput
from ...validation.preprocessors import RTMDetValPreprocessor
from ..base import BaseModel
from .nn import LibreRTMDetModel
from .utils import postprocess as _postprocess
from .utils import preprocess_image as _rtmdet_preprocess
from .utils import preprocess_numpy as _preprocess_numpy


class LibreRTMDet(BaseModel):
    """RTMDet detector (CSPNeXt backbone + CSPNeXtPAFPN neck + decoupled SepBN head).

    Args:
        model_path: path to a LibreRTMDet weight file, or None for a fresh model.
        size: one of {"t", "s", "m", "l", "x"}.
        nb_classes: number of classes (default 80 for COCO).
        device: inference device.

    Examples::

        >>> model = LibreYOLO("LibreRTMDett.pt")
        >>> result = model("image.jpg", save=True)
    """

    FAMILY = "rtmdet"
    FILENAME_PREFIX = "LibreRTMDet"
    INPUT_SIZES = {"t": 640, "s": 640, "m": 640, "l": 640, "x": 640}
    SUPPORTED_TASKS = ("detect",)
    DEFAULT_TASK = "detect"
    TRAIN_CONFIG = None  # inference-only first PR; trainer follow-up
    val_preprocessor_class = RTMDetValPreprocessor

    # =========================================================================
    # Registry classmethods
    # =========================================================================

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        # `rtm_cls` / `rtm_reg` are unique to RTMDet (no other family in the
        # registry uses these prefixes). Both `bbox_head.rtm_cls` (upstream)
        # and `head.rtm_cls` (LibreRTMDet checkpoints) match.
        return any(
            "rtm_cls" in k or "rtm_reg" in k for k in weights_dict
        )

    @classmethod
    def detect_size(cls, weights_dict: dict) -> Optional[str]:
        # Stem first conv has channels = 64 * widen_factor // 2.
        # tiny=12, s=16, m=24, l=32, x=40.
        for key in ("backbone.stem.0.conv.weight",):
            if key in weights_dict:
                ch = int(weights_dict[key].shape[0])
                return {12: "t", 16: "s", 24: "m", 32: "l", 40: "x"}.get(ch)
        return None

    @classmethod
    def detect_nb_classes(cls, weights_dict: dict) -> Optional[int]:
        for key in ("head.rtm_cls.0.weight", "bbox_head.rtm_cls.0.weight"):
            if key in weights_dict:
                return int(weights_dict[key].shape[0])
        return None

    # =========================================================================
    # Initialization
    # =========================================================================

    def __init__(
        self,
        model_path=None,
        size: str = "s",
        nb_classes: int = 80,
        device: str = "auto",
        **kwargs,
    ):
        super().__init__(
            model_path=model_path,
            size=size,
            nb_classes=nb_classes,
            device=device,
            **kwargs,
        )
        if isinstance(model_path, str):
            self._load_weights(model_path)

    # =========================================================================
    # Model lifecycle
    # =========================================================================

    def _init_model(self) -> nn.Module:
        return LibreRTMDetModel(size=self.size, nc=self.nb_classes)

    def _get_available_layers(self) -> Dict[str, nn.Module]:
        return {
            "backbone": self.model.backbone,
            "neck": self.model.neck,
            "head": self.model.head,
        }

    def _strict_loading(self) -> bool:
        # share_conv aliasing means the saved state_dict has fewer keys than the
        # model exposes (cls_convs[0] / reg_convs[0] only). Strict loading would
        # complain about the missing aliased keys.
        return False

    # =========================================================================
    # Inference pipeline
    # =========================================================================

    @staticmethod
    def _get_preprocess_numpy():
        return _preprocess_numpy

    def _preprocess(
        self,
        image: ImageInput,
        color_format: str = "auto",
        input_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Image.Image, Tuple[int, int], float]:
        effective_size = input_size if input_size is not None else self.input_size
        return _rtmdet_preprocess(
            image, input_size=effective_size, color_format=color_format
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
        actual_input_size = kwargs.get("input_size", self.input_size)

        # Validation path passes ratio=1.0; recompute from original_size if so.
        if ratio == 1.0 and original_size is not None:
            orig_w, orig_h = original_size
            ratio = min(actual_input_size / orig_h, actual_input_size / orig_w)

        return _postprocess(
            output,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            input_size=actual_input_size,
            original_size=original_size,
            ratio=ratio,
            max_det=max_det,
        )

    # =========================================================================
    # Training surface (inference-only for now)
    # =========================================================================

    def train(self, *args, **kwargs):  # noqa: D401
        raise NotImplementedError(
            "RTMDet training is not yet implemented in LibreYOLO. "
            "The first PR ships inference-only across t/s/m/l/x sizes; "
            "trainer (loss + assigner + 2-stage pipeline switch) is a follow-up."
        )
