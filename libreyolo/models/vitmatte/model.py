"""LibreViTMatte: trimap-guided image matting.

The family-local network is an inference-only port of the Apache-2.0
Transformers ViTMatte implementation. The published Composition-1k checkpoint
is a separate, NON-COMMERCIAL surface; see ``NOTICE`` in this directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Dict, Optional, Tuple

import torch
import torch.nn as nn

from ...postprocess.vitmatte import postprocess as _vitmatte_postprocess
from ...utils.image_loader import ImageInput
from ..base import BaseModel
from .nn import LibreViTMatteModel
from .utils import preprocess_guided_image, preprocess_numpy
from .validator import (
    DEFAULT_TRIMAP_RADIUS,
    ViTMatteValidator,
    validation_trimap_options,
)


ADOBE_DIM_LICENSE_URL = "https://sites.google.com/view/deepimagematting/homepage"
WEIGHT_LICENSE = "Adobe Deep Image Matting dataset terms, non-commercial"


class LibreViTMatte(BaseModel):
    """ViTMatte-S with an explicit three-level ``trimap=`` guide.

    RGB and trimap are combined into four channels at native resolution and
    padded on the bottom/right to a multiple of 32. The returned matte is a
    probability alpha on the exact source canvas; known trimap background and
    foreground pixels are forced to zero and one respectively.
    """

    FAMILY = "vitmatte"
    FILENAME_PREFIX = "LibreViTMatte"
    WEIGHT_EXT = ".pt"
    INPUT_SIZES: ClassVar[Dict[str, int]] = {"s": 512}
    SUPPORTED_TASKS = ("matte",)
    DEFAULT_TASK = "matte"
    REQUIRE_TASK_SUFFIX = True
    TRAIN_CONFIG = None
    validator_class = ViTMatteValidator
    SUPPORTS_BATCHED_PREDICT = False
    TTA_ENABLED = False
    PREDICT_INPUT_KWARGS = ("trimap",)
    REQUIRED_PREDICT_INPUT_KWARGS = ("trimap",)

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        return cls.detect_size(weights_dict) is not None

    @classmethod
    def detect_size(cls, weights_dict: dict) -> Optional[str]:
        projection = weights_dict.get("backbone.embeddings.projection.weight")
        positions = weights_dict.get("backbone.embeddings.position_embeddings")
        global_relative = weights_dict.get(
            "backbone.encoder.layer.2.attention.rel_pos_h"
        )
        residual = weights_dict.get("backbone.encoder.layer.11.residual.conv3.weight")
        detail_input = weights_dict.get("decoder.convstream.convs.0.conv.weight")
        head = weights_dict.get("decoder.matting_head.matting_convs.3.weight")
        if (
            getattr(projection, "shape", None) == torch.Size((384, 4, 16, 16))
            and getattr(positions, "shape", None) == torch.Size((1, 197, 384))
            and getattr(global_relative, "shape", None) == torch.Size((63, 64))
            and getattr(residual, "shape", None) == torch.Size((384, 192, 1, 1))
            and getattr(detail_input, "shape", None) == torch.Size((48, 4, 3, 3))
            and getattr(head, "shape", None) == torch.Size((1, 16, 1, 1))
        ):
            return "s"
        return None

    @classmethod
    def detect_nb_classes(cls, weights_dict: dict) -> Optional[int]:
        return 1 if cls.can_load(weights_dict) else None

    @classmethod
    def detect_checkpoint_task(cls, state_dict: dict) -> Optional[str]:
        return "matte" if cls.can_load(state_dict) else None

    @classmethod
    def default_checkpoint_names(
        cls, nc: int, task: Optional[str] = None
    ) -> Optional[Dict[int, str]]:
        return {0: "matte"} if nc == 1 else None

    @classmethod
    def get_download_notice(cls, filename: str, url: str) -> str:
        del url
        return (
            f"{Path(filename).name} was trained on Adobe Composition-1k. "
            "LibreYOLO treats this pretrained checkpoint as NON-COMMERCIAL "
            "under the Adobe Deep Image Matting Dataset License Agreement "
            f"({ADOBE_DIM_LICENSE_URL}). This restriction applies to the "
            "pretrained weights, not to LibreYOLO's MIT code or the ViTMatte "
            "architecture. Retain the family NOTICE and required attribution."
        )

    def __init__(
        self,
        model_path=None,
        size: str = "s",
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
        self.names = {0: "matte"}
        self.model.eval()

    def _init_model(self) -> nn.Module:
        return LibreViTMatteModel()

    def _get_available_layers(self) -> Dict[str, nn.Module]:
        return {
            "backbone": self.model.backbone,
            "decoder": self.model.decoder,
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
        del image, color_format, input_size
        raise ValueError(
            "LibreViTMatte requires trimap=. Use model.predict(image, trimap=guide)."
        )

    def _preprocess_predict(
        self,
        image: ImageInput,
        color_format: str = "auto",
        input_size: Optional[int] = None,
        *,
        trimap: ImageInput | None = None,
    ) -> Tuple[torch.Tensor, Any, Tuple[int, int], float]:
        del input_size
        if trimap is None:
            raise ValueError(
                "LibreViTMatte requires trimap=. Use model.predict(image, trimap=guide)."
            )
        return preprocess_guided_image(
            image,
            trimap,
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
    ) -> Dict:
        del conf_thres, iou_thres, max_det, ratio, kwargs
        return _vitmatte_postprocess(output, original_size)

    def val(
        self,
        *args: Any,
        trimap_dir: str | Path | None = None,
        trimap_radius: int = DEFAULT_TRIMAP_RADIUS,
        **kwargs: Any,
    ) -> Dict:
        """Validate with supplied guides or deterministic GT-derived trimaps.

        ``trimap_dir`` must contain one three-level guide per image stem. When
        omitted, known foreground/background are eroded by the fixed
        ``trimap_radius`` (15 pixels by default) and the remaining band is
        marked unknown before prediction. Metrics remain the shared matte MAE
        and S-measure.
        """
        with validation_trimap_options(trimap_dir, trimap_radius):
            return super().val(*args, **kwargs)

    def train(self, *args: Any, **kwargs: Any):
        raise NotImplementedError(
            "LibreViTMatte is inference-only. Training is not implemented for "
            "this family."
        )


__all__ = [
    "ADOBE_DIM_LICENSE_URL",
    "LibreViTMatte",
    "WEIGHT_LICENSE",
]
