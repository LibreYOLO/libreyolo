"""LibreCenterNet model-family wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from ...postprocess.centernet import postprocess
from ...utils.image_loader import ImageInput
from ...validation.preprocessors import CenterNetValPreprocessor
from ..base import BaseModel
from .nn import build_centernet
from .utils import preprocess_image


class LibreCenterNet(BaseModel):
    """CenterNet Objects-as-Points detectors with ResDCN-18 or DLA-34."""

    FAMILY = "centernet"
    FILENAME_PREFIX = "LibreCenterNet"
    INPUT_SIZES = {"resdcn18": 512, "dla34": 512}
    SUPPORTED_TASKS = ("detect",)
    DEFAULT_TASK = "detect"
    TRAIN_CONFIG = None
    val_preprocessor_class = CenterNetValPreprocessor
    TTA_ENABLED = False

    def __init__(
        self,
        model_path=None,
        size: str = "resdcn18",
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
        if isinstance(model_path, (str, Path)):
            self._load_weights(str(model_path))
        self.model.eval()

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        """Claim only CenterNet detect checkpoints with all three dense heads."""
        has_heads = all(
            key in weights_dict
            for key in ("hm.2.weight", "wh.2.weight", "reg.2.weight")
        )
        if not has_heads:
            return False
        return (
            "conv1.weight" in weights_dict
            and "deconv_layers.0.conv_offset_mask.weight" in weights_dict
        ) or (
            "base.base_layer.0.weight" in weights_dict
            and "dla_up.ida_0.proj_1.conv.conv_offset_mask.weight" in weights_dict
        )

    @classmethod
    def convert_upstream_state_dict(cls, state_dict: dict) -> Optional[dict]:
        """Remove only the official data-parallel prefix, with no tensor edits."""
        converted = {
            key.removeprefix("module."): value for key, value in state_dict.items()
        }
        return converted if cls.can_load(converted) else None

    @classmethod
    def detect_size(cls, weights_dict: dict) -> Optional[str]:
        converted = cls.convert_upstream_state_dict(weights_dict)
        if converted is None:
            return None
        if "base.base_layer.0.weight" in converted:
            return "dla34"
        return "resdcn18"

    @classmethod
    def detect_size_from_filename(cls, filename: str) -> Optional[str]:
        detected = super().detect_size_from_filename(filename)
        if detected is not None:
            return detected
        normalized = Path(filename).name.lower().replace("-", "_")
        if "resdcn18" in normalized:
            return "resdcn18"
        if "dla_2x" in normalized or "dla34" in normalized:
            return "dla34"
        return None

    @classmethod
    def detect_nb_classes(cls, weights_dict: dict) -> Optional[int]:
        converted = cls.convert_upstream_state_dict(weights_dict)
        if converted is None:
            return None
        return int(converted["hm.2.weight"].shape[0])

    def _init_model(self) -> nn.Module:
        return build_centernet(self.size, num_classes=self.nb_classes)

    def _get_available_layers(self) -> Dict[str, nn.Module]:
        if self.size == "dla34":
            return {
                "backbone": self.model.base,
                "neck": self.model.dla_up,
                "fusion": self.model.ida_up,
                "hm": self.model.hm,
                "wh": self.model.wh,
                "reg": self.model.reg,
            }
        return {
            "backbone": nn.Sequential(
                self.model.conv1,
                self.model.bn1,
                self.model.relu,
                self.model.maxpool,
                self.model.layer1,
                self.model.layer2,
                self.model.layer3,
                self.model.layer4,
            ),
            "neck": self.model.deconv_layers,
            "hm": self.model.hm,
            "wh": self.model.wh,
            "reg": self.model.reg,
        }

    @staticmethod
    def _get_preprocess_numpy():
        from .utils import preprocess_numpy

        return preprocess_numpy

    def _preprocess(
        self,
        image: ImageInput,
        color_format: str = "auto",
        input_size: Optional[int] = None,
    ):
        return preprocess_image(
            image,
            input_size=self.input_size if input_size is None else int(input_size),
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
        max_det: int = 100,
        **kwargs,
    ) -> dict:
        return postprocess(
            output,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            original_size=original_size,
            input_size=kwargs.get("input_size", self.input_size),
            max_det=min(max_det, 100),
            topk=100,
        )

    def train(self, *args, **kwargs):
        raise NotImplementedError(
            "CenterNet is currently inference-only; focal/offset training is not implemented."
        )

    def export(
        self,
        format: str = "onnx",
        *,
        opset: int = 18,
        **kwargs,
    ) -> str:
        if format.lower() == "onnx" and opset < 16:
            raise NotImplementedError(
                "CenterNet ONNX export requires opset 16 or newer for GridSample."
            )
        return super().export(format=format, opset=opset, **kwargs)

    def _strict_loading(self) -> bool:
        return True


__all__ = ["LibreCenterNet"]
