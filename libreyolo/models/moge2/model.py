"""LibreYOLO wrapper for MoGe-2 surface-normal estimation.

MoGe-2 is Microsoft's single-forward monocular geometry model. LibreYOLO ships
the official ViT-S, ViT-B, and ViT-L normal checkpoints as an
inference/validation/export family. Their native normal output already uses
the LibreYOLO OpenCV camera frame, so the family boundary only resizes and
renormalizes the vectors.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, ClassVar, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from ...utils.image_loader import ImageInput, ImageLoader
from ..base.model import BaseModel
from .nn import MoGe2NormalNet
from .utils import PATCH_SIZE, preprocess_numpy

_NORMAL_STATE_PREFIXES = ("encoder.", "neck.", "normal_head.")


def _normal_only_state_dict(state_dict: dict) -> dict:
    """Keep only tensors used by the normal-only inference graph."""
    cleaned = {}
    for key, value in state_dict.items():
        while key.startswith(("module.", "_orig_mod.", "model.")):
            key = key.split(".", 1)[1]
        if key.startswith(_NORMAL_STATE_PREFIXES):
            cleaned[key] = value
    return cleaned


class LibreMoGe2(BaseModel):
    """MoGe-2 ViT-S/B/L: RGB image to a dense OpenCV-frame normal field."""

    FAMILY = "moge2"
    FILENAME_PREFIX = "LibreMoGe2"
    WEIGHT_EXT = ".pt"
    INPUT_SIZES: ClassVar[Dict[str, int]] = {
        "s": 518,
        "b": 518,
        "l": 518,
    }
    SUPPORTED_TASKS = ("normal",)
    DEFAULT_TASK = "normal"
    REQUIRE_TASK_SUFFIX = True
    TRAIN_CONFIG = None
    SUPPORTS_BATCHED_PREDICT = False

    normal_imgsz_divisor = PATCH_SIZE
    normal_resize_mode = "letterbox"

    _EMBED_DIM_TO_SIZE: ClassVar[Dict[int, str]] = {
        384: "s",
        768: "b",
        1024: "l",
    }
    # Sizes LibreYOLO mirrors on its own Hugging Face org. MoGe-2 is MIT, which
    # permits redistribution, so these no longer depend on a third-party repo
    # staying put.
    #
    # The mirror serves the *converted* LibreYOLO checkpoint, so the upstream
    # digests in _OFFICIAL_WEIGHTS cannot describe it; these are pinned
    # separately. The mirror URL resolves `main`, which is mutable, so this
    # pinning is what stops a replaced or corrupted upload from loading
    # silently. Regenerate a value only when the mirror is deliberately
    # re-uploaded.
    #
    # Sizes absent from this map still download from upstream and are still
    # checksum-pinned against _OFFICIAL_WEIGHTS.
    _MIRROR_SHA256: ClassVar[Dict[str, str]] = {
        "s": "0b3c1301ddcae5569234010905f093fa8bec5866c7c06197761ea501651f9d9c",
        "l": "342c13b7028a2e87d164ee9647ad4f34d822dcb73221004c9f25d0458e17580a",
    }

    _OFFICIAL_WEIGHTS: ClassVar[Dict[str, tuple[str, str]]] = {
        "s": (
            "https://huggingface.co/Ruicheng/moge-2-vits-normal/resolve/"
            "679230677b4d282c6f304189a93e98e14f085902/model.pt",
            "79a16621928c2bf0ed04659218c55c01075e950507f40bb3332fb4c873d3e1dc",
        ),
        "b": (
            "https://huggingface.co/Ruicheng/moge-2-vitb-normal/resolve/"
            "54ad3a693e61907ea4633d13dec6ee682fa09419/model.pt",
            "16b8110e86d5dc5a849db120ca96ef3a223fd30b0c9146d1d81db504073da5f6",
        ),
        "l": (
            "https://huggingface.co/Ruicheng/moge-2-vitl-normal/resolve/"
            "b135031bae30b5ac2ae141a0e68717795ce38340/model.pt",
            "280741fd09bc3f403ccff9967784c2a391b52d2c0742ae3efdb21d9f90cc1a01",
        ),
    }

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        return (
            "encoder.backbone.cls_token" in weights_dict
            and "neck.input_blocks.0.weight" in weights_dict
            and "normal_head.output_blocks.4.weight" in weights_dict
        )

    @classmethod
    def detect_size(cls, weights_dict: dict) -> Optional[str]:
        cls_token = weights_dict.get("encoder.backbone.cls_token")
        if cls_token is None or getattr(cls_token, "ndim", 0) < 1:
            return None
        return cls._EMBED_DIM_TO_SIZE.get(int(cls_token.shape[-1]))

    @classmethod
    def detect_nb_classes(cls, weights_dict: dict) -> Optional[int]:
        return 1

    @classmethod
    def detect_checkpoint_task(cls, state_dict: dict) -> Optional[str]:
        if any(key.startswith("normal_head.") for key in state_dict):
            return "normal"
        return None

    @classmethod
    def convert_upstream_state_dict(cls, state_dict: dict) -> Optional[dict]:
        if not cls.can_load(state_dict):
            return None
        return _normal_only_state_dict(state_dict)

    @classmethod
    def get_download_url(cls, filename: str) -> Optional[str]:
        size = cls.detect_size_from_filename(filename)
        if size is None or cls.detect_task_from_filename(filename) != "normal":
            return None
        if size in cls._MIRROR_SHA256:
            # BaseModel builds the LibreYOLO org URL for the canonical filename.
            return super().get_download_url(filename)
        entry = cls._OFFICIAL_WEIGHTS.get(size)
        return entry[0] if entry else None

    @staticmethod
    def _sha256(path: str) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @classmethod
    def verify_downloaded_file(cls, local_path: str, source_url: str) -> None:
        # Mirror and upstream serve different bytes (converted vs original), so
        # each has its own recorded digest. Both are checked: `resolve/main` is
        # mutable, and an unverified mirror would be a weaker guarantee than the
        # upstream path it replaced.
        if source_url.startswith("https://huggingface.co/LibreYOLO/"):
            size = next(
                (s for s in cls._MIRROR_SHA256 if f"LibreMoGe2{s}-normal" in source_url),
                None,
            )
            if size is None:
                raise ValueError(f"Unrecognized MoGe-2 mirror URL: {source_url}")
            expected = cls._MIRROR_SHA256[size]
            actual = cls._sha256(local_path)
            if actual != expected:
                raise ValueError(
                    f"MoGe-2 {size} mirrored checkpoint SHA-256 mismatch: "
                    f"expected {expected}, got {actual}."
                )
            return

        size = next(
            (
                size
                for size, (url, _) in cls._OFFICIAL_WEIGHTS.items()
                if url == source_url
            ),
            None,
        )
        if size is None:
            raise ValueError(f"Unrecognized MoGe-2 weight URL: {source_url}")
        expected = cls._OFFICIAL_WEIGHTS[size][1]
        actual = cls._sha256(local_path)
        if actual != expected:
            raise ValueError(
                f"MoGe-2 {size} checkpoint SHA-256 mismatch: "
                f"expected {expected}, got {actual}."
            )

    def __init__(
        self,
        model_path,
        size: str = "l",
        nb_classes: int = 1,
        device: str = "auto",
        task: str | None = None,
        **kwargs,
    ):
        super().__init__(
            model_path=model_path,
            size=size,
            nb_classes=1,
            device=device,
            task=task,
            **kwargs,
        )
        self.model.eval()
        if model_path is not None and isinstance(model_path, (str, Path)):
            self._load_weights(str(model_path))
        self.nb_classes = 1
        self.names = {0: "normal"}

    def _prepare_state_dict(self, state_dict: dict) -> dict:
        selected = _normal_only_state_dict(state_dict)
        return selected if selected else state_dict

    def _init_model(self) -> nn.Module:
        return MoGe2NormalNet(size=self.size)

    def _get_available_layers(self) -> Dict[str, nn.Module]:
        return {
            "encoder": self.model.encoder,
            "neck": self.model.neck,
            "normal_head": self.model.normal_head,
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
        if effective_res <= 0:
            raise ValueError(f"MoGe-2 imgsz must be positive, got {effective_res}.")
        img = ImageLoader.load(image, color_format=color_format)
        orig_w, orig_h = img.size
        chw, ratio = preprocess_numpy(np.asarray(img), effective_res)
        return torch.from_numpy(chw).unsqueeze(0), img, (orig_w, orig_h), ratio

    def _forward(self, input_tensor: torch.Tensor) -> Any:
        if input_tensor.is_cuda:
            with torch.autocast("cuda", dtype=torch.float16):
                return self.model(input_tensor)
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
        del conf_thres, iou_thres, max_det, kwargs
        normal = output
        if isinstance(normal, dict):
            normal = normal.get("normal", normal.get("normals"))
        if isinstance(normal, (list, tuple)):
            normal = normal[0]
        if normal is None:
            raise ValueError("MoGe-2 output did not contain a normal map.")
        if normal.ndim == 4 and normal.shape[-1] == 3:
            normal = normal.permute(0, 3, 1, 2)
        if normal.ndim != 4 or normal.shape[1] != 3:
            raise ValueError(
                "MoGe-2 normal output must be [B, 3, H, W] or [B, H, W, 3], "
                f"got {tuple(normal.shape)}."
            )

        orig_w, orig_h = original_size
        normal = F.interpolate(
            normal.float(),
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=False,
        )
        finite = torch.isfinite(normal).all(dim=1, keepdim=True)
        safe = torch.where(finite, normal, 0.0)
        norms = torch.linalg.vector_norm(safe, dim=1, keepdim=True)
        valid = finite & (norms > 1e-12)
        unit = safe / norms.clamp_min(1e-12)
        fallback = torch.zeros_like(unit)
        fallback[:, 2] = -1.0
        unit = torch.where(valid, unit, fallback)
        return {"normal": unit[0].permute(1, 2, 0).cpu()}

    def train(self, *args, **kwargs):
        raise NotImplementedError(
            "MoGe-2 training is not part of LibreYOLO's normal-family "
            "integration. Use a pinned official checkpoint for predict, val, "
            "and export."
        )

    def export(self, format: str = "onnx", **kwargs) -> str:
        """Export through the fixed-resolution dense-normal contract."""
        return super().export(format=format, **kwargs)


__all__ = ["LibreMoGe2"]
