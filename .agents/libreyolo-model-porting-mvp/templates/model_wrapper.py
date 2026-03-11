from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from PIL import Image

from ..base import BaseModel
from ...utils.image_loader import ImageInput
from ...validation.preprocessors import StandardValPreprocessor
from .nn import LibreYourModel
from .utils import preprocess_image, postprocess


class LibreYourFamily(BaseModel):
    FAMILY = "yourfamily"
    FILENAME_PREFIX = "LibreYourFamily"
    INPUT_SIZES = {"s": 640}
    val_preprocessor_class = StandardValPreprocessor

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        keys = [k.lower() for k in weights_dict]
        return any("replace_me_unique_key" in k for k in keys)

    @classmethod
    def detect_size(cls, weights_dict: dict) -> Optional[str]:
        # Return one of INPUT_SIZES.keys(), or None if unknown.
        return "s"

    @classmethod
    def detect_nb_classes(cls, weights_dict: dict) -> Optional[int]:
        # Return class count if discoverable from checkpoint tensors.
        return None

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

    def _init_model(self) -> nn.Module:
        return LibreYourModel(config=self.size, nb_classes=self.nb_classes)

    def _get_available_layers(self) -> Dict[str, nn.Module]:
        return {
            "backbone": self.model.backbone,
            "head": self.model.head,
        }

    def _strict_loading(self) -> bool:
        return False

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
        effective_size = input_size if input_size is not None else self.input_size
        return preprocess_image(
            image,
            input_size=effective_size,
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
        return postprocess(
            output,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            original_size=original_size,
            max_det=max_det,
            ratio=ratio,
            input_size=kwargs.get("input_size", self.input_size),
            letterbox=kwargs.get("letterbox", False),
        )
