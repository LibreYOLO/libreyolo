"""LibreSwin: standalone Swin V1 image-classification family."""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from ...postprocess.swin import postprocess as _swin_postprocess
from ..base import BaseModel
from ..base.classify_preprocess import ClassifyPreprocessMixin
from .classifier import SwinClassifier


class LibreSwin(ClassifyPreprocessMixin, BaseModel):
    """Swin V1 image classifiers in tiny, small, base, and large sizes.

    Swin made transformers practical as general-purpose dense vision
    backbones by combining local window attention, shifted windows, and a
    hierarchical feature pyramid. This standalone museum family reuses the
    native Swin tower already shared by Grounding DINO and OMDet-Turbo.

    This release is inference-only: prediction, ImageNet-style top-1/top-5
    validation, and export are supported; the upstream training recipe is out
    of scope.
    """

    FAMILY = "swin"
    FILENAME_PREFIX = "LibreSwin"
    INPUT_SIZES = {"t": 224, "s": 224, "b": 224, "l": 224}
    SUPPORTED_TASKS = ("classify",)
    DEFAULT_TASK = "classify"
    REQUIRE_TASK_SUFFIX = True
    TRAIN_CONFIG = None
    CROP_PCT = {"t": 0.9, "s": 0.9, "b": 0.9, "l": 0.9}

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        """Claim only shipped Swin V1 patch-4/window-7 classifier graphs."""
        return (
            "patch_embed.proj.weight" in weights_dict
            and "layers.0.blocks.0.attn.relative_position_bias_table"
            in weights_dict
            and "norm.weight" in weights_dict
            and "head.fc.weight" in weights_dict
            and not any(
                "cpb_mlp" in key or key.endswith("attn.logit_scale")
                for key in weights_dict
            )
            and cls.detect_size(weights_dict) is not None
        )

    @classmethod
    def detect_size(cls, weights_dict: dict) -> Optional[str]:
        patch_key = "patch_embed.proj.weight"
        bias_key = "layers.0.blocks.0.attn.relative_position_bias_table"
        head_key = "head.fc.weight"
        if any(key not in weights_dict for key in (patch_key, bias_key, head_key)):
            return None

        patch = weights_dict[patch_key]
        bias = weights_dict[bias_key]
        head = weights_dict[head_key]
        if patch.ndim != 4 or tuple(patch.shape[1:]) != (3, 4, 4):
            return None
        embed_dim = int(patch.shape[0])
        if bias.ndim != 2 or int(bias.shape[0]) != 169:
            return None
        if head.ndim != 2 or int(head.shape[1]) != embed_dim * 8:
            return None

        stage_two_blocks = set()
        for key in weights_dict:
            match = re.match(r"^layers\.2\.blocks\.(\d+)\.norm1\.weight$", key)
            if match:
                stage_two_blocks.add(int(match.group(1)))
        depth = len(stage_two_blocks)
        if stage_two_blocks != set(range(depth)):
            return None
        return {
            (96, 6): "t",
            (96, 18): "s",
            (128, 18): "b",
            (192, 18): "l",
        }.get((embed_dim, depth))

    @classmethod
    def detect_nb_classes(cls, weights_dict: dict) -> Optional[int]:
        head = weights_dict.get("head.fc.weight")
        return int(head.shape[0]) if head is not None and head.ndim == 2 else None

    @classmethod
    def convert_upstream_state_dict(cls, weights_dict: dict) -> Optional[dict]:
        """Normalize released Microsoft/timm layouts for runtime conversion."""
        from .convert import convert_upstream, is_upstream_state_dict

        if not is_upstream_state_dict(weights_dict):
            return None
        converted = convert_upstream(weights_dict)
        return converted if cls.can_load(converted) else None

    def __init__(
        self,
        model_path=None,
        size: str = "t",
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
        self.interpolation = "bicubic"
        if isinstance(model_path, str):
            self._load_weights(model_path)

    def _init_model(self) -> nn.Module:
        return SwinClassifier(size=self.size, num_classes=self.nb_classes)

    def _get_available_layers(self) -> Dict[str, nn.Module]:
        return {
            "patch_embed": self.model.patch_embed,
            "stages": self.model.layers,
            "norm": self.model.norm,
            "classifier": self.model.head.fc,
        }

    def _rebuild_for_new_classes(self, new_nb_classes: int) -> None:
        self.nb_classes = new_nb_classes
        self.names = {i: f"class_{i}" for i in range(new_nb_classes)}
        self.model.reset_classifier(new_nb_classes)
        self.model.to(self.device)

    def _validate_imgsz(self, imgsz: Any, *, context: str) -> int:
        """Enforce the resolution-specific final-stage attention graph."""
        native = int(self._get_input_size())
        if isinstance(imgsz, (tuple, list)):
            try:
                valid = tuple(int(side) for side in imgsz) == (native, native)
            except (TypeError, ValueError):
                valid = False
        else:
            try:
                valid = int(imgsz) == native
            except (TypeError, ValueError):
                valid = False
        if not valid:
            raise ValueError(
                f"Swin V1 {context} is fixed to native imgsz={native}; "
                f"got {imgsz!r}. The final-stage attention graph is "
                "resolution-specific."
            )
        return native

    def _get_val_preprocessor(self, img_size: int | None = None):
        if img_size is not None:
            img_size = self._validate_imgsz(
                img_size, context="validation imgsz"
            )
        return super()._get_val_preprocessor(img_size=img_size)

    def _check_eval_imgsz(self, imgsz: int) -> None:
        # val() already checked its own imgsz ("validation imgsz"), so the
        # only caller that reaches this with a bad size is predict().
        self._validate_imgsz(imgsz, context="prediction imgsz")

    def _forward(self, input_tensor: torch.Tensor) -> Any:
        return self.model(input_tensor)

    def _postprocess(self, output: Any, *args, **kwargs) -> Dict:
        return _swin_postprocess(output, *args, **kwargs)

    def val(self, *args, **kwargs):
        positional = list(args)
        if len(positional) >= 3 and positional[2] is not None:
            positional[2] = self._validate_imgsz(
                positional[2], context="validation imgsz"
            )
        elif kwargs.get("imgsz") is not None:
            kwargs["imgsz"] = self._validate_imgsz(
                kwargs["imgsz"], context="validation imgsz"
            )
        return super().val(*positional, **kwargs)

    def export(self, format: str = "onnx", **kwargs) -> str:
        if kwargs.get("imgsz") is not None:
            kwargs["imgsz"] = self._validate_imgsz(
                kwargs["imgsz"], context="export imgsz"
            )
        return super().export(format=format, **kwargs)

    def train(self, *args, **kwargs):
        del args, kwargs
        raise NotImplementedError(
            "Swin is shipped inference-only in this museum release. The "
            "upstream ImageNet training recipe is not implemented in LibreYOLO."
        )
