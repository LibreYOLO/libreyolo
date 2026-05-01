"""LibrePicoDet: BaseModel subclass wiring PicoDet into the LibreYOLO factory."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from PIL import Image

from ...utils.image_loader import ImageInput
from ...validation.preprocessors import PicoDetValPreprocessor
from ..base import BaseModel
from .nn import LibrePicoDetModel
from .utils import postprocess as _picodet_postprocess
from .utils import preprocess_image as _picodet_preprocess


class LibrePicoDet(BaseModel):
    """PicoDet object detector (s/m/l).

    Examples::

        >>> model = LibreYOLO("LibrePicoDets.pt")
        >>> dets = model(image="image.jpg")

        >>> # Inference-only init for now (training wired in a later commit):
        >>> model = LibrePicoDet(size="s")
    """

    FAMILY = "picodet"
    FILENAME_PREFIX = "LibrePicoDet"
    INPUT_SIZES = {"s": 320, "m": 416, "l": 640}
    # Trainer is added in a follow-up; setting to None marks this as
    # inference-only until then (see skill §4: ``TRAIN_CONFIG = None`` is the
    # supported sentinel for inference-only ports).
    TRAIN_CONFIG = None
    val_preprocessor_class = PicoDetValPreprocessor

    # ---- registry --------------------------------------------------------

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        # Tokens unique to PicoDet: shared GFL head + ESNet block list.
        # Avoids matching YOLOX (``head.stems``), YOLOv9, DETR families, etc.
        has_gfl = any("head.gfl_cls" in k for k in weights_dict)
        has_esnet = any("backbone.blocks" in k for k in weights_dict)
        return has_gfl and has_esnet

    @classmethod
    def detect_size(cls, weights_dict: dict) -> Optional[str]:
        # Distinguish via the first stage's output channels: s=24->96, m=24->128, l=24->160.
        # ``backbone.blocks.0`` is an ESBlockDS; its ``conv_pw_2.conv`` has
        # ``out_channels = mid_channels // 2`` and ``in_channels=24`` (stem out).
        # The unambiguous tell is the neck transformer: ``neck.trans.0.conv.weight``
        # has shape (neck_ch, backbone_c3_ch, 1, 1).
        key = "neck.trans.0.conv.weight"
        if key not in weights_dict:
            return None
        neck_ch = weights_dict[key].shape[0]
        return {96: "s", 128: "m", 160: "l"}.get(int(neck_ch))

    @classmethod
    def detect_nb_classes(cls, weights_dict: dict) -> Optional[int]:
        key = "head.gfl_cls.0.weight"
        if key not in weights_dict:
            return None
        # Shared cls/reg head: out_channels = num_classes + 4 * (reg_max + 1).
        # PicoDet uses reg_max=7 -> 32 reg channels. Subtract.
        out_ch = int(weights_dict[key].shape[0])
        nc = out_ch - 32
        return nc if nc > 0 else None

    # ---- init ------------------------------------------------------------

    def __init__(
        self,
        model_path=None,
        size: str = "s",
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

    def _init_model(self) -> nn.Module:
        return LibrePicoDetModel(size=self.size, nb_classes=self.nb_classes)

    def _get_available_layers(self) -> Dict[str, nn.Module]:
        return {
            "backbone_conv1": self.model.backbone.conv1,
            "backbone_blocks": self.model.backbone.blocks,
            "neck": self.model.neck,
            "head": self.model.head,
        }

    def _strict_loading(self) -> bool:
        # Converted Paddle/Bo checkpoints may carry init_cfg state, EMA buffers,
        # or auxiliary keys we drop. Strict loading would refuse them.
        return False

    # ---- inference -------------------------------------------------------

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
        eff = input_size if input_size is not None else self.input_size
        return _picodet_preprocess(image, input_size=eff, color_format=color_format)

    def _forward(self, input_tensor: torch.Tensor) -> Any:
        return self.model(input_tensor)

    def _postprocess(
        self,
        output: Any,
        conf_thres: float,
        iou_thres: float,
        original_size: Tuple[int, int],
        max_det: int = 100,
        ratio: float = 1.0,
        **kwargs,
    ) -> Dict:
        actual_input_size = kwargs.get("input_size", self.input_size)
        return _picodet_postprocess(
            output,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            input_size=actual_input_size,
            original_size=original_size,
            max_det=max_det,
        )

    # ---- training (placeholder) ------------------------------------------

    def train(self, *args: Any, **kwargs: Any) -> dict:
        raise NotImplementedError(
            "PicoDet training is not yet wired into LibreYOLO. "
            "Inference and weight loading are supported; see issue #161."
        )
