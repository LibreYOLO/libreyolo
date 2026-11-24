"""LibreYOLO wrapper for PIDNet semantic segmentation.

PIDNet is a real-time three-branch CNN for semantic segmentation. This family
ships Cityscapes 19-class checkpoints as ``LibrePIDNet{s,m,l}-sem.pt`` and
reuses LibreYOLO's shared semantic dataset, validator, and result surface.

The network code is ported from the MIT-licensed upstream repository:
https://github.com/XuJiacong/PIDNet
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, ClassVar, Dict, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from ...tasks import normalize_task
from ...utils.image_loader import ImageInput, ImageLoader
from ...utils.serialization import (
    load_untrusted_torch_file,
    validate_checkpoint_metadata,
)
from ..base.model import BaseModel
from .nn import SIZE_CONFIGS, LibrePIDNetNet

logger = logging.getLogger(__name__)


CITYSCAPES_NAMES: dict[int, str] = {
    0: "road",
    1: "sidewalk",
    2: "building",
    3: "wall",
    4: "fence",
    5: "pole",
    6: "traffic light",
    7: "traffic sign",
    8: "vegetation",
    9: "terrain",
    10: "sky",
    11: "person",
    12: "rider",
    13: "car",
    14: "truck",
    15: "bus",
    16: "train",
    17: "motorcycle",
    18: "bicycle",
}

_AUX_HEAD_PREFIXES = ("seghead_p.", "seghead_d.")
_COMMON_PREFIXES = ("module.", "model.", "net.")


def _strip_known_prefix(key: str) -> str:
    changed = True
    while changed:
        changed = False
        for prefix in _COMMON_PREFIXES:
            if key.startswith(prefix):
                key = key[len(prefix) :]
                changed = True
                break
    return key


def _canonical_state_dict_keys(weights_dict: dict) -> set[str]:
    return {_strip_known_prefix(str(key)) for key in weights_dict}


def _input_size_hw(input_size: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(input_size, int):
        return input_size, input_size
    if len(input_size) != 2:
        raise ValueError(
            f"input_size must be int or (height, width), got {input_size!r}"
        )
    return int(input_size[0]), int(input_size[1])


def preprocess_numpy(
    img_rgb_hwc: np.ndarray,
    input_size: int | tuple[int, int] = 1024,
) -> tuple[np.ndarray, float]:
    """Letterbox RGB image to PIDNet's canvas as CHW float32 in [0, 1]."""
    orig_h, orig_w = img_rgb_hwc.shape[:2]
    input_h, input_w = _input_size_hw(input_size)
    ratio = min(input_h / orig_h, input_w / orig_w)
    new_h = max(int(orig_h * ratio), 1)
    new_w = max(int(orig_w * ratio), 1)

    resized = cv2.resize(img_rgb_hwc, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded = np.full((input_h, input_w, 3), 114, dtype=np.uint8)
    padded[:new_h, :new_w] = resized

    arr = np.ascontiguousarray(padded, dtype=np.float32) / 255.0
    return arr.transpose(2, 0, 1), ratio


class LibrePIDNet(BaseModel):
    """PIDNet S/M/L family for dense semantic segmentation."""

    FAMILY: ClassVar[str] = "pidnet"
    FILENAME_PREFIX: ClassVar[str] = "LibrePIDNet"
    # Forward is pure tensor work with no host sync, verified to capture and
    # replay bit-identically (tests/unit/test_cuda_graph_families.py).
    SUPPORTS_CUDA_GRAPH = True
    WEIGHT_EXT: ClassVar[str] = ".pt"
    SUPPORTED_TASKS: ClassVar[Tuple[str, ...]] = ("semantic",)
    DEFAULT_TASK: ClassVar[str] = "semantic"
    REQUIRE_TASK_SUFFIX: ClassVar[bool] = True
    INPUT_SIZES: ClassVar[Dict[str, int]] = {"s": 1024, "m": 1024, "l": 1024}

    semantic_resize_mode: ClassVar[str] = "letterbox"
    semantic_imgsz_divisor: ClassVar[int] = 8

    _SIZE_CONFIGS: ClassVar[dict] = SIZE_CONFIGS
    _UPSTREAM_URL: ClassVar[str] = "https://github.com/XuJiacong/PIDNet"

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        keys = _canonical_state_dict_keys(weights_dict)
        has_stem = "conv1.0.weight" in keys
        has_head = "final_layer.conv2.weight" in keys
        # DAPPM (``spp.``) is shared with DDRNet, PIDNet's ancestor, so only
        # the PID-specific fusion modules count as evidence here.
        has_pid_modules = any(
            key.startswith(("pag3.", "pag4.", "diff3.", "diff4.", "dfm."))
            for key in keys
        )
        return has_stem and has_head and has_pid_modules

    @staticmethod
    def _tensor_for_key(weights_dict: dict, wanted: str):
        for key, value in weights_dict.items():
            if _strip_known_prefix(str(key)) == wanted:
                return value
        return None

    @classmethod
    def detect_size(cls, weights_dict: dict) -> Optional[str]:
        conv = cls._tensor_for_key(weights_dict, "conv1.0.weight")
        if conv is None or getattr(conv, "ndim", 0) < 1:
            return None
        planes = int(conv.shape[0])
        if planes == 32:
            return "s"
        if planes != 64:
            return None

        head = cls._tensor_for_key(weights_dict, "final_layer.conv2.weight")
        if head is not None and getattr(head, "ndim", 0) >= 2:
            head_channels = int(head.shape[1])
            if head_channels == 256:
                return "l"
            if head_channels == 128:
                return "m"

        keys = _canonical_state_dict_keys(weights_dict)
        if any(key.startswith("spp.process1.") for key in keys):
            return "l"
        return "m"

    @classmethod
    def convert_upstream_state_dict(cls, state_dict: dict) -> Optional[dict]:
        return None

    @classmethod
    def detect_nb_classes(cls, weights_dict: dict) -> Optional[int]:
        head = cls._tensor_for_key(weights_dict, "final_layer.conv2.weight")
        if head is not None and getattr(head, "ndim", 0) >= 1:
            return int(head.shape[0])
        bias = cls._tensor_for_key(weights_dict, "final_layer.conv2.bias")
        if bias is not None and getattr(bias, "ndim", 0) >= 1:
            return int(bias.shape[0])
        return None

    def __init__(
        self,
        model_path=None,
        size: str = "s",
        nb_classes: int = 19,
        device: str = "auto",
        task: str | None = None,
        **kwargs,
    ) -> None:
        resolved_task = normalize_task(task) if task is not None else "semantic"
        if resolved_task != "semantic":
            raise ValueError(
                f"LibrePIDNet supports only task='semantic'; got {task!r}."
            )
        super().__init__(
            model_path=model_path,
            size=size,
            nb_classes=nb_classes,
            device=device,
            task=resolved_task,
            **kwargs,
        )
        if self.nb_classes == 19:
            self.names = dict(CITYSCAPES_NAMES)
        self.model.eval()
        if self.model_path is not None:
            self._load_weights(str(self.model_path))

    def _init_model(self) -> nn.Module:
        return LibrePIDNetNet(size=self.size, num_classes=self.nb_classes)

    def _get_available_layers(self) -> Dict[str, nn.Module]:
        return {
            "stem": self.model.conv1,
            "i_branch": nn.ModuleList(
                [
                    self.model.layer1,
                    self.model.layer2,
                    self.model.layer3,
                    self.model.layer4,
                    self.model.layer5,
                ]
            ),
            "p_branch": nn.ModuleList(
                [self.model.layer3_, self.model.layer4_, self.model.layer5_]
            ),
            "d_branch": nn.ModuleList(
                [self.model.layer3_d, self.model.layer4_d, self.model.layer5_d]
            ),
            "fusion": nn.ModuleList([self.model.pag3, self.model.pag4, self.model.dfm]),
            "head": self.model.final_layer,
        }

    @staticmethod
    def _get_preprocess_numpy():
        return preprocess_numpy

    def _preprocess(
        self,
        image: ImageInput,
        color_format: str = "auto",
        input_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Image.Image, Tuple[int, int], float]:
        effective_res = input_size if input_size is not None else self._get_input_size()
        if effective_res % self.semantic_imgsz_divisor:
            raise ValueError(
                f"LibrePIDNet semantic imgsz={effective_res} must be divisible "
                f"by {self.semantic_imgsz_divisor} (PIDNet output stride)."
            )
        img = ImageLoader.load(image, color_format=color_format)
        orig_w, orig_h = img.size
        chw, ratio = preprocess_numpy(np.asarray(img), effective_res)
        return torch.from_numpy(chw).unsqueeze(0), img, (orig_w, orig_h), ratio

    def _forward(self, input_tensor: torch.Tensor) -> Any:
        return self.model(input_tensor)

    def _postprocess_semantic_logits(
        self,
        output: Any,
        original_size: Tuple[int, int],
        ratio: float = 1.0,
        **kwargs,
    ) -> torch.Tensor:
        """Interpolate raw semantic logits to ``original_size``, pre-argmax.

        Shared by ``_postprocess`` (single-view predict/val) and
        ``BaseModel._predict_augment_semantic`` (flip TTA), which needs the
        pre-argmax logits to average across augmented views.
        """
        logits = output
        if isinstance(logits, dict):
            logits = logits.get("semantic_logits", logits.get("predictions"))
        if isinstance(logits, (list, tuple)):
            logits = logits[1] if len(logits) > 1 else logits[0]
        orig_w, orig_h = original_size
        input_size = kwargs.get("input_size", self._get_input_size())
        input_h, input_w = _input_size_hw(input_size)
        scale_y = logits.shape[-2] / input_h
        scale_x = logits.shape[-1] / input_w
        valid_h = min(logits.shape[-2], max(int(round(orig_h * ratio * scale_y)), 1))
        valid_w = min(logits.shape[-1], max(int(round(orig_w * ratio * scale_x)), 1))
        logits = logits[..., :valid_h, :valid_w]
        return F.interpolate(
            logits.float(),
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=True,
        )

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
        logits = self._postprocess_semantic_logits(output, original_size, ratio=ratio, **kwargs)
        return {"semantic": logits.argmax(dim=1)[0].cpu()}

    def _strict_loading(self) -> bool:
        return False

    def _prepare_state_dict(self, state_dict: dict) -> dict:
        prepared = {}
        target_keys = set(self.model.state_dict())
        for raw_key, value in state_dict.items():
            key = _strip_known_prefix(str(raw_key))
            if key.startswith(_AUX_HEAD_PREFIXES):
                continue
            if key in target_keys:
                prepared[key] = value
        return prepared

    def _validate_loaded_state_dict_for_task(
        self,
        state_dict: dict,
        checkpoint: dict | None = None,
    ) -> None:
        if not self.can_load(state_dict):
            raise RuntimeError(
                "Checkpoint does not look like a PIDNet semantic segmentation model."
            )

    def _load_weights(self, model_path: str | dict[str, Any]) -> None:
        if isinstance(model_path, str):
            if not Path(model_path).exists():
                from ...utils.download import download_weights

                download_weights(model_path, self.size)
            loaded = load_untrusted_torch_file(
                model_path, map_location="cpu", context="PIDNet semantic weights"
            )
        else:
            loaded = model_path

        if not isinstance(loaded, dict):
            raise TypeError("LibrePIDNet checkpoints must be dictionaries")
        metadata_errors = validate_checkpoint_metadata(loaded, strict=False)
        if metadata_errors:
            raise ValueError(
                "Raw upstream PIDNet checkpoints must be converted before loading. "
                "Use weights/convert_pidnet_weights.py to create a LibreYOLO "
                "checkpoint with Cityscapes semantic metadata."
            )

        ckpt_family = loaded.get("model_family")
        if isinstance(ckpt_family, str) and ckpt_family and ckpt_family != self.FAMILY:
            raise RuntimeError(
                f"Checkpoint was trained with model_family='{ckpt_family}' "
                f"but is being loaded into '{self.FAMILY}'."
            )

        ckpt_task = loaded.get("task")
        if isinstance(ckpt_task, str) and normalize_task(ckpt_task) != "semantic":
            raise RuntimeError(
                f"Checkpoint was trained for task={normalize_task(ckpt_task)!r}, "
                "but LibrePIDNet is semantic-only."
            )

        if isinstance(loaded.get("model"), dict):
            raw_state = loaded["model"]
        elif isinstance(loaded.get("state_dict"), dict):
            raw_state = loaded["state_dict"]
        else:
            raw_state = loaded
        state = self._prepare_state_dict(raw_state)
        ckpt_nc = loaded.get("nc") or self.detect_nb_classes(state)
        if ckpt_nc is not None and int(ckpt_nc) != self.nb_classes:
            self._rebuild_for_checkpoint_classes(int(ckpt_nc), state)
            state = self._prepare_state_dict(raw_state)

        if not self.can_load(state):
            raise RuntimeError(
                "Checkpoint does not look like a PIDNet semantic segmentation model."
            )
        result = self.model.load_state_dict(state, strict=False)
        missing = list(getattr(result, "missing_keys", []) or [])
        missing_required = [
            key
            for key in missing
            if not key.startswith(("pixel_mean", "pixel_std"))
            and not key.endswith("num_batches_tracked")
        ]
        if missing_required:
            preview = ", ".join(missing_required[:5])
            raise RuntimeError(f"PIDNet checkpoint is missing model keys: {preview}")

        ckpt_names = loaded.get("names")
        if ckpt_names is not None:
            self.names = self._sanitize_names(ckpt_names, self.nb_classes)
        elif self.nb_classes == 19:
            self.names = dict(CITYSCAPES_NAMES)
        self.model.to(self.device).eval()

    def train(self, *args, **kwargs):
        raise NotImplementedError(
            "Training PIDNet is out of scope for LibreYOLO v1. "
            f"Train upstream at {self._UPSTREAM_URL} and convert the result "
            "with weights/convert_pidnet_weights.py."
        )

    def export(self, format: str = "onnx", **kwargs) -> str:
        """Export dense logits through the shared semantic runtime contract."""
        return super().export(format=format, **kwargs)


__all__ = ["CITYSCAPES_NAMES", "LibrePIDNet", "preprocess_numpy"]
