"""LibreDDColor restoration-family integration.

The tensor network remains the upstream two-channel Lab ``ab`` predictor.
OpenCV Lab conversion belongs to the prediction wrapper, where the original
resolution ``L`` plane is still available. See the family ``NOTICE`` for the
pinned Apache-2.0 source and its permissively licensed subcomponents.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Dict, Optional, Tuple

import torch
import torch.nn as nn

from ...postprocess.ddcolor import postprocess as _ddcolor_postprocess
from ...utils.image_loader import ImageInput
from ..base import BaseModel
from .nn import DDColor
from .utils import DDCOLOR_ORIGINAL_L_KEY, preprocess_image, preprocess_numpy
from .validator import DDColorValidator


DDCOLOR_SIZE_CONFIGS: dict[str, dict[str, Any]] = {
    "t": {
        "encoder_name": "convnext-t",
        "depths": (3, 3, 9, 3),
        "dims": (96, 192, 384, 768),
        "checkpoint": "piddnad/ddcolor_paper_tiny",
    },
    "l": {
        "encoder_name": "convnext-l",
        "depths": (3, 3, 27, 3),
        "dims": (192, 384, 768, 1536),
        "checkpoint": "piddnad/ddcolor_modelscope",
    },
}

_INPUT_SIZE = 512
_QUERY_SHAPE = torch.Size((100, 256))
_REFINE_SHAPE = torch.Size((2, 103, 1, 1))


def _has_ddcolor_signature(state_dict: dict) -> bool:
    """Recognize the complete, architecture-specific DDColor key signature."""

    query = state_dict.get("decoder.color_decoder.query_feat.weight")
    refine = state_dict.get("refine_net.0.0.weight_orig")
    stem = state_dict.get("encoder.arch.downsample_layers.0.0.weight")
    return (
        getattr(query, "shape", None) == _QUERY_SHAPE
        and getattr(refine, "shape", None) == _REFINE_SHAPE
        and getattr(stem, "ndim", 0) == 4
        and tuple(stem.shape[1:]) == (3, 4, 4)
        and "decoder.color_decoder.transformer_cross_attention_layers.8.multihead_attn.in_proj_weight"
        in state_dict
        and "encoder.arch.norm3.weight" in state_dict
    )


class LibreDDColor(BaseModel):
    """DDColor automatic colorization with exact OpenCV Lab reconstruction.

    Prediction resizes a neutral-L RGB image to 512 square, predicts Lab
    chroma, resizes chroma back with nearest interpolation, and combines it
    with the source image's original-resolution luminance plane.
    """

    FAMILY = "ddcolor"
    FILENAME_PREFIX = "LibreDDColor"
    INPUT_SIZES: ClassVar[Dict[str, int]] = {"t": _INPUT_SIZE, "l": _INPUT_SIZE}
    SUPPORTED_TASKS = ("restore",)
    DEFAULT_TASK = "restore"
    REQUIRE_TASK_SUFFIX = True
    TRAIN_CONFIG = None
    validator_class = DDColorValidator
    SUPPORTS_BATCHED_PREDICT = True
    TTA_ENABLED = False

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        return cls.detect_size(weights_dict) is not None

    @classmethod
    def detect_size(cls, weights_dict: dict) -> Optional[str]:
        if not _has_ddcolor_signature(weights_dict):
            return None
        stem = weights_dict["encoder.arch.downsample_layers.0.0.weight"]
        return {96: "t", 192: "l"}.get(int(stem.shape[0]))

    @classmethod
    def detect_nb_classes(cls, weights_dict: dict) -> Optional[int]:
        return 1 if cls.can_load(weights_dict) else None

    @classmethod
    def detect_checkpoint_task(cls, state_dict: dict) -> Optional[str]:
        return "restore" if cls.can_load(state_dict) else None

    @classmethod
    def get_download_notice(cls, filename: str, url: str) -> str:
        del url
        return (
            f"{Path(filename).name} is published under Apache-2.0 by DDColor's "
            "authors. The checkpoint was trained on ImageNet and initialized "
            "from ImageNet-22K weights; ImageNet's data access terms are "
            "non-commercial research/education terms. No ImageNet data is "
            "bundled by LibreYOLO. The Artistic checkpoint, which also uses "
            "undisclosed private data, is not distributed."
        )

    def __init__(
        self,
        model_path=None,
        size: str = "t",
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
        return 1

    def _init_model(self) -> nn.Module:
        config = DDCOLOR_SIZE_CONFIGS[self.size]
        return DDColor(
            encoder_name=str(config["encoder_name"]),
            input_size=(_INPUT_SIZE, _INPUT_SIZE),
            num_output_channels=2,
            last_norm="Spectral",
            do_normalize=False,
            num_queries=100,
            num_scales=3,
            dec_layers=9,
        )

    def _get_available_layers(self) -> Dict[str, nn.Module]:
        return {
            "encoder": self.model.encoder,
            "decoder": self.model.decoder,
            "refine_net": self.model.refine_net,
        }

    @staticmethod
    def _get_preprocess_numpy():
        return preprocess_numpy

    def _preprocess(
        self,
        image: ImageInput,
        color_format: str = "auto",
        input_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Any, Tuple[int, int], Any]:
        size = _INPUT_SIZE if input_size is None else int(input_size)
        return preprocess_image(
            image,
            input_size=size,
            color_format=color_format,
        )

    def _forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.model(input_tensor)

    def _postprocess(
        self,
        output: Any,
        conf_thres: float,
        iou_thres: float,
        original_size: Tuple[int, int],
        max_det: int = 300,
        ratio: Any = None,
        **kwargs,
    ) -> Dict:
        del conf_thres, iou_thres, max_det, kwargs
        if not isinstance(ratio, dict) or DDCOLOR_ORIGINAL_L_KEY not in ratio:
            raise ValueError(
                "DDColor postprocessing requires the original Lab L plane from "
                "LibreDDColor preprocessing. Call predict() with an image instead "
                "of invoking the two-channel network output directly."
            )
        restored = _ddcolor_postprocess(
            output,
            original_size,
            original_l=ratio[DDCOLOR_ORIGINAL_L_KEY],
        )
        return {"restored": restored}

    def train(self, *args: Any, **kwargs: Any):
        raise NotImplementedError(
            "LibreDDColor is inference-only. Training is not implemented for "
            "this family."
        )


__all__ = ["DDCOLOR_SIZE_CONFIGS", "LibreDDColor"]
