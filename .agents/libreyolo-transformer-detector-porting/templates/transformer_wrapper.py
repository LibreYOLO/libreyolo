from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from PIL import Image

from ..base import BaseModel
from ...utils.image_loader import ImageInput, ImageLoader
from ...validation.preprocessors import RFDETRValPreprocessor
from .nn import LibreYourTransformerModel
from .utils import postprocess, IMAGENET_MEAN, IMAGENET_STD


class LibreYourTransformer(BaseModel):
    FAMILY = "yourtransformer"
    FILENAME_PREFIX = "LibreYourTransformer"
    INPUT_SIZES = {"s": 512}
    val_preprocessor_class = RFDETRValPreprocessor

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        keys = [k.lower() for k in weights_dict]
        return any(
            "query_embed" in k
            or "class_embed" in k
            or "bbox_embed" in k
            or "transformer" in k
            for k in keys
        )

    @classmethod
    def detect_size(cls, weights_dict: dict, state_dict: dict | None = None) -> Optional[str]:
        # Use checkpoint args/config here if tensor-shape detection is unreliable.
        return "s"

    @classmethod
    def detect_nb_classes(cls, weights_dict: dict) -> Optional[int]:
        # Account for possible background-inclusive heads.
        return None

    def __init__(
        self,
        model_path=None,
        size: str = "s",
        nb_classes: int = 80,
        device: str = "auto",
        **kwargs,
    ):
        self._pretrain_weights = model_path
        super().__init__(
            model_path=None,
            size=size,
            nb_classes=nb_classes,
            device=device,
            **kwargs,
        )

    def _init_model(self) -> nn.Module:
        return LibreYourTransformerModel(
            config=self.size,
            nb_classes=self.nb_classes,
            pretrain_weights=self._pretrain_weights,
            device=str(self.device),
        )

    def _get_available_layers(self) -> Dict[str, nn.Module]:
        layers = {}
        if hasattr(self.model, "backbone"):
            layers["backbone"] = self.model.backbone
        if hasattr(self.model, "encoder"):
            layers["encoder"] = self.model.encoder
        if hasattr(self.model, "decoder"):
            layers["decoder"] = self.model.decoder
        return layers

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
        effective_res = input_size if input_size is not None else self.input_size

        img = ImageLoader.load(image, color_format=color_format)
        orig_w, orig_h = img.size
        orig_size = (orig_w, orig_h)

        img_tensor = F.to_tensor(img)
        img_tensor = F.normalize(img_tensor, IMAGENET_MEAN, IMAGENET_STD)
        img_tensor = F.resize(img_tensor, (effective_res, effective_res))
        img_tensor = img_tensor.unsqueeze(0)

        return img_tensor, img, orig_size, 1.0

    def _forward(self, input_tensor: torch.Tensor) -> Any:
        return self.model(input_tensor)

    def _postprocess(
        self,
        output: Any,
        conf_thres: float,
        iou_thres: float,
        original_size: Tuple[int, int],
        max_det: int = 300,
        **kwargs,
    ) -> Dict:
        # Implement conversion from transformer outputs (often pred_logits/pred_boxes)
        # into boxes/scores/classes in original image coordinates.
        return postprocess(
            output,
            original_size=original_size,
            conf_thres=conf_thres,
            max_det=max_det,
        )
