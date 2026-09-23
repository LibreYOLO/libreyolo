"""LibreViT: standalone classic Vision Transformer classification family."""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from PIL import Image

from ...postprocess.vit import postprocess as _vit_postprocess
from ...utils.image_loader import ImageInput
from ...validation.vit_validator import ViTClassifyValidator
from ..base import BaseModel
from .nn import VisionTransformer
from .utils import preprocess_image as _vit_preprocess
from .utils import VIT_MEAN, VIT_STD


class LibreViT(BaseModel):
    """Classic ViT image classifiers in tiny, small, base, and large sizes.

    ViT established the pure transformer over fixed image patches as a
    competitive vision backbone. Its learned class token and position
    embeddings became the architectural starting point for later CLIP, DINO,
    SAM, and transformer-based detection families.

    This first museum release is inference-only. It supports prediction,
    ImageNet-style top-1/top-5 validation, and export, while the AugReg
    fine-tuning recipe remains out of scope.
    """

    FAMILY = "vit"
    FILENAME_PREFIX = "LibreViT"
    INPUT_SIZES = {"ti": 224, "s": 224, "b": 224, "l": 224}
    SUPPORTED_TASKS = ("classify",)
    DEFAULT_TASK = "classify"
    REQUIRE_TASK_SUFFIX = True
    TRAIN_CONFIG = None
    validator_class = ViTClassifyValidator
    # AugReg normalization, read by val, calibration and export metadata
    # (#886); predict's transform in utils.py uses the same constants.
    norm_mean = VIT_MEAN
    norm_std = VIT_STD

    # All four AugReg ImageNet-1k checkpoints use the same timm eval config.
    CROP_PCT = {"ti": 0.9, "s": 0.9, "b": 0.9, "l": 0.9}

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        """Claim only shipped 224px patch-16 classic-ViT checkpoints."""
        return (
            "cls_token" in weights_dict
            and "pos_embed" in weights_dict
            and "patch_embed.proj.weight" in weights_dict
            and "head.weight" in weights_dict
            and cls.detect_size(weights_dict) is not None
        )

    @classmethod
    def detect_size(cls, weights_dict: dict) -> Optional[str]:
        """Infer the exact shipped tier from embedding width and block depth."""
        patch_key = "patch_embed.proj.weight"
        pos_key = "pos_embed"
        head_key = "head.weight"
        cls_key = "cls_token"
        if any(
            key not in weights_dict for key in (patch_key, pos_key, head_key, cls_key)
        ):
            return None

        patch = weights_dict[patch_key]
        pos = weights_dict[pos_key]
        cls_token = weights_dict[cls_key]
        head = weights_dict[head_key]
        if patch.ndim != 4 or tuple(patch.shape[1:]) != (3, 16, 16):
            return None
        embed_dim = int(patch.shape[0])
        # 224 / 16 = 14, plus one class token. Reject patch32, 384px, hybrid,
        # and in21k-only head layouts instead of silently resizing them.
        if tuple(pos.shape) != (1, 197, embed_dim):
            return None
        if tuple(cls_token.shape) != (1, 1, embed_dim):
            return None
        if head.ndim != 2 or int(head.shape[1]) != embed_dim:
            return None

        block_indices = set()
        for key in weights_dict:
            match = re.match(r"^blocks\.(\d+)\.norm1\.weight$", key)
            if match:
                block_indices.add(int(match.group(1)))
        signature = (embed_dim, len(block_indices))
        size = {
            (192, 12): "ti",
            (384, 12): "s",
            (768, 12): "b",
            (1024, 24): "l",
        }.get(signature)
        if size is None or block_indices != set(range(len(block_indices))):
            return None
        return size

    @classmethod
    def detect_nb_classes(cls, weights_dict: dict) -> Optional[int]:
        head = weights_dict.get("head.weight")
        return int(head.shape[0]) if head is not None and head.ndim == 2 else None

    def __init__(
        self,
        model_path=None,
        size: str = "ti",
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
        return VisionTransformer(
            size=self.size,
            num_classes=self.nb_classes,
            init_weights=not self._loading_from_weights,
        )

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

    def _get_preprocess_numpy(self):
        from functools import partial

        from .utils import preprocess_numpy

        return partial(preprocess_numpy, crop_pct=self.crop_pct)

    def _preprocess(
        self,
        image: ImageInput,
        color_format: str = "auto",
        input_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Image.Image, Tuple[int, int], float]:
        effective_size = input_size if input_size is not None else self.input_size
        return _vit_preprocess(
            image,
            input_size=effective_size,
            crop_pct=self.crop_pct,
            color_format=color_format,
        )

    def _forward(self, input_tensor: torch.Tensor) -> Any:
        return self.model(input_tensor)

    def _postprocess(self, output: Any, *args, **kwargs) -> Dict:
        return _vit_postprocess(output, *args, **kwargs)

    def train(self, *args, **kwargs):
        del args, kwargs
        raise NotImplementedError(
            "ViT is shipped inference-only in this museum release. The AugReg "
            "fine-tuning recipe is not implemented in LibreYOLO yet."
        )
