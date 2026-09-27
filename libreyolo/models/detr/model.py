"""LibreDETR: BaseModel integration for the original DETR detector."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from ...utils.image_loader import ImageInput
from ...utils.coco import COCO91_TO_COCO80
from ...utils.serialization import load_untrusted_torch_file
from ...validation.preprocessors import DETRValPreprocessor
from ..base import BaseModel
from .nn import LibreDETRModel
from .utils import preprocess_image, unwrap_detr_checkpoint

_COCO91_CLASS_SLOTS = 91
_QUERY_KEY = "query_embed.weight"
_CROSS_ATTN_KEY = "transformer.decoder.layers.0.multihead_attn.in_proj_weight"
_BACKBONE_STEM_KEY = "backbone.0.body.conv1.weight"
_CLASS_KEY = "class_embed.weight"


class LibreDETR(BaseModel):
    """DETR (ECCV 2020), the first end-to-end set-prediction detector.

    The four shipped variants use ResNet-50/101, optionally with a dilated C5
    stage (DC5), 100 learned object queries, and a six-layer transformer
    encoder-decoder. Inference is NMS-free. LibreYOLO uses a fixed 800x800 RGB
    ImageNet-normalized canvas; the official evaluation recipe instead keeps
    aspect ratio with an 800-pixel short side and a 1333-pixel long-side cap.

    This museum family is inference-only. The 500-epoch Hungarian-matching
    training recipe remains available in the official Apache-2.0 source but is
    intentionally outside this port.
    """

    FAMILY = "detr"
    FILENAME_PREFIX = "LibreDETR"
    INPUT_SIZES = {"r50": 800, "r50dc5": 800, "r101": 800, "r101dc5": 800}
    SUPPORTED_TASKS = ("detect",)
    DEFAULT_TASK = "detect"
    TRAIN_CONFIG = None
    val_preprocessor_class = DETRValPreprocessor
    TTA_FIXED_SIZE = True

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        """Match vanilla DETR's unique packed cross-attention signature."""
        required = (_QUERY_KEY, _CROSS_ATTN_KEY, _BACKBONE_STEM_KEY, _CLASS_KEY)
        if not all(key in weights_dict for key in required):
            return False
        try:
            return (
                tuple(weights_dict[_QUERY_KEY].shape) == (100, 256)
                and tuple(weights_dict[_CROSS_ATTN_KEY].shape) == (768, 256)
                and tuple(weights_dict[_BACKBONE_STEM_KEY].shape) == (64, 3, 7, 7)
                and int(weights_dict[_CLASS_KEY].shape[1]) == 256
            )
        except (AttributeError, IndexError, TypeError, ValueError):
            return False

    @classmethod
    def detect_size_from_filename(cls, filename: str) -> Optional[str]:
        detected = super().detect_size_from_filename(filename)
        if detected is not None:
            return detected
        name = Path(filename).name.lower()
        # "rtdetr-r50.pt" and "LibreRTDETR-r50.pt" name RT-DETR, not DETR.
        match = re.search(r"(?:^|[^a-z])detr-r(50|101)(-dc5)?(?:-|\.|$)", name)
        if match is None:
            return None
        depth, dc5 = match.groups()
        return f"r{depth}{'dc5' if dc5 else ''}"

    @classmethod
    def detect_size(cls, weights_dict: dict) -> Optional[str]:
        """Return ``None`` because DC5 does not alter any checkpoint tensor.

        ResNet depth is visible in the key count, but dilation is a runtime
        configuration with exactly the same state dict. Official raw files are
        therefore resolved by their unambiguous filenames; LibreYOLO checkpoints
        carry authoritative ``size`` metadata.
        """
        del weights_dict
        return None

    @classmethod
    def detect_nb_classes(cls, weights_dict: dict) -> Optional[int]:
        architectural = cls._arch_classes_from_state_dict(weights_dict)
        if architectural is None:
            return None
        return 80 if architectural == _COCO91_CLASS_SLOTS else architectural

    @staticmethod
    def _arch_classes_from_state_dict(weights_dict: dict) -> Optional[int]:
        weight = weights_dict.get(_CLASS_KEY)
        if weight is None or getattr(weight, "ndim", 0) != 2:
            return None
        # The final output is the no-object class and is not user-facing.
        return int(weight.shape[0]) - 1

    def __init__(
        self,
        model_path,
        size: str,
        nb_classes: int = 80,
        device: str = "auto",
        **kwargs,
    ) -> None:
        # ``size`` is deliberately required: DC5 dilation changes the runtime
        # graph without changing any serialized tensor shape, so a renamed raw
        # DC5 checkpoint would strict-load into the wrong (non-dilated) graph
        # if a default were silently applied.
        architectural = self._peek_arch_classes(model_path)
        if architectural is None:
            architectural = _COCO91_CLASS_SLOTS if nb_classes == 80 else nb_classes
        self._arch_num_classes = architectural
        user_classes = 80 if architectural == _COCO91_CLASS_SLOTS else architectural

        super().__init__(
            model_path=None,
            size=size,
            nb_classes=user_classes,
            device=device,
            **kwargs,
        )

        self.model_path = model_path if isinstance(model_path, (str, Path)) else None
        if isinstance(model_path, dict):
            state_dict = self._prepare_state_dict(
                self._strip_ddp_prefix(unwrap_detr_checkpoint(model_path))
            )
            self.model.load_state_dict(state_dict, strict=self._strict_loading())
            self.model.to(self.device).eval()
        elif model_path is not None:
            self._load_weights(str(model_path))

    @staticmethod
    def _peek_arch_classes(model_path) -> Optional[int]:
        if isinstance(model_path, dict):
            return LibreDETR._arch_classes_from_state_dict(
                unwrap_detr_checkpoint(model_path)
            )
        if not isinstance(model_path, (str, Path)):
            return None
        path = Path(BaseModel._resolve_weights_path(str(model_path)))
        if not path.exists():
            return None
        try:
            loaded = load_untrusted_torch_file(
                str(path), map_location="cpu", context="DETR head inspection"
            )
        except Exception:
            return None
        if not isinstance(loaded, dict):
            return None
        return LibreDETR._arch_classes_from_state_dict(unwrap_detr_checkpoint(loaded))

    def _init_model(self) -> nn.Module:
        return LibreDETRModel(size=self.size, nc=self._arch_num_classes)

    def _rebuild_for_new_classes(self, new_nb_classes: int):
        self._arch_num_classes = new_nb_classes
        super()._rebuild_for_new_classes(new_nb_classes)

    def _get_available_layers(self) -> Dict[str, nn.Module]:
        return {
            "backbone": self.model.backbone,
            "transformer": self.model.transformer,
            "class_embed": self.model.class_embed,
            "bbox_embed": self.model.bbox_embed,
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
    ) -> Tuple[torch.Tensor, Any, Tuple[int, int], float]:
        return preprocess_image(
            image,
            input_size=input_size if input_size is not None else self.input_size,
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
    ) -> Dict:
        from ...postprocess.detr import postprocess

        class_map = (
            COCO91_TO_COCO80 if self._arch_num_classes == _COCO91_CLASS_SLOTS else None
        )
        return postprocess(
            output,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            original_size=original_size,
            max_det=max_det,
            class_map=class_map,
            **kwargs,
        )

    def train(self, *args, **kwargs):
        del args, kwargs
        raise NotImplementedError(
            "DETR is shipped inference-only. Its official 500-epoch Hungarian "
            "matching training recipe is not implemented in LibreYOLO yet."
        )

    def _strict_loading(self) -> bool:
        return True
