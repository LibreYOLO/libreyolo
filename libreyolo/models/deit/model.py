"""LibreDeiT: BaseModel wrapper for DeiT ImageNet classification."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from ...postprocess.deit import postprocess as _deit_postprocess
from ..base import BaseModel
from ..base.classify_preprocess import ClassifyPreprocessMixin
from .nn import DeiT


class LibreDeiT(ClassifyPreprocessMixin, BaseModel):
    """Plain DeiT patch-16 classifiers in tiny, small, and base sizes.

    DeiT made Vision Transformers practical with ImageNet-1k alone through a
    strong data-efficient training recipe. This first museum release contains
    the plain 224-pixel variants. Distillation-token and 384-pixel variants are
    deliberately out of scope.

    Raw plain DeiT parameters have the same structural layout as a vanilla ViT
    with matching geometry. Official LibreYOLO checkpoints carry
    ``model_family='deit'`` metadata to resolve that unavoidable ambiguity.
    """

    FAMILY = "deit"
    FILENAME_PREFIX = "LibreDeiT"
    INPUT_SIZES = {"t": 224, "s": 224, "b": 224}
    SUPPORTED_TASKS = ("classify",)
    DEFAULT_TASK = "classify"
    REQUIRE_TASK_SUFFIX = True
    TRAIN_CONFIG = None
    CROP_PCT = {"t": 0.9, "s": 0.9, "b": 0.9}

    @classmethod
    def detect_size(cls, weights_dict: dict) -> Optional[str]:
        required = (
            "patch_embed.proj.weight",
            "cls_token",
            "pos_embed",
            "blocks.11.mlp.fc2.weight",
            "head.weight",
        )
        if not all(key in weights_dict for key in required):
            return None
        if any(key.startswith("blocks.12.") for key in weights_dict):
            return None
        try:
            patch = weights_dict["patch_embed.proj.weight"]
            cls_token = weights_dict["cls_token"]
            pos_embed = weights_dict["pos_embed"]
            dim = int(patch.shape[0])
            if tuple(patch.shape[1:]) != (3, 16, 16):
                return None
            if tuple(cls_token.shape) != (1, 1, dim):
                return None
            if tuple(pos_embed.shape) != (1, 197, dim):
                return None
            if int(weights_dict["head.weight"].shape[1]) != dim:
                return None
            if tuple(weights_dict["blocks.11.mlp.fc2.weight"].shape) != (
                dim,
                4 * dim,
            ):
                return None
        except (AttributeError, IndexError, TypeError, ValueError):
            return None
        return {192: "t", 384: "s", 768: "b"}.get(dim)

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        return cls.detect_size(weights_dict) is not None

    @classmethod
    def detect_nb_classes(cls, weights_dict: dict) -> Optional[int]:
        weight = weights_dict.get("head.weight")
        return int(weight.shape[0]) if hasattr(weight, "shape") else None

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
        return DeiT(size=self.size, num_classes=self.nb_classes)

    def _get_available_layers(self) -> Dict[str, nn.Module]:
        return {
            "patch_embed": self.model.patch_embed,
            "blocks": self.model.blocks,
            "norm": self.model.norm,
            "classifier": self.model.head,
        }

    def _rebuild_for_new_classes(self, new_nb_classes: int) -> None:
        self.nb_classes = new_nb_classes
        self.names = {i: f"class_{i}" for i in range(new_nb_classes)}
        self.model.reset_classifier(new_nb_classes)
        self.model.to(self.device)

    def _check_eval_imgsz(self, imgsz: int) -> None:
        if int(imgsz) != int(self.input_size):
            raise ValueError(
                f"LibreDeiT {self.size!r} has a fixed {self.input_size}x{self.input_size} "
                f"positional embedding; got imgsz={imgsz}."
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
        return _deit_postprocess(
            output,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            original_size=original_size,
            max_det=max_det,
            ratio=ratio,
            **kwargs,
        )

    def train(self, *args, **kwargs):
        del args, kwargs
        raise NotImplementedError(
            "DeiT is shipped as an inference-only museum family. The original "
            "distillation and data-efficient training recipe is not implemented."
        )

    def _strict_loading(self) -> bool:
        return True
