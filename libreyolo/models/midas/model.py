"""LibreMiDaS: MiDaS relative monocular depth in the LibreYOLO factory.

MiDaS introduced zero-shot monocular relative depth through scale-and-shift
invariant training across mixed datasets. Its later DPT decoder established the
transformer dense-prediction pattern inherited by Depth Anything and related
families.

Predictions are relative inverse depth: larger values mean closer, and values
have no metric unit or cross-image scale. This first integration is
inference-only and ships two complementary official variants:

* ``s``: MiDaS v2.1 Small, EfficientNet-Lite3, 256-pixel upper-bound resize,
  ImageNet mean/std normalization.
* ``l``: DPT-Large, ViT-L/16, 384-pixel minimal resize, mean/std 0.5.
"""

from __future__ import annotations

import hashlib
from functools import partial
from pathlib import Path
from typing import Any, ClassVar, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from ...utils.image_loader import ImageInput, ImageLoader
from ..base.model import BaseModel
from .convert import UPSTREAM_URLS, verify_and_wrap_download
from .nn import build_midas_model
from .utils import IMGSZ_DIVISOR, preprocess_numpy


class LibreMiDaS(BaseModel):
    """MiDaS image-to-relative-inverse-depth family."""

    FAMILY = "midas"
    FILENAME_PREFIX = "LibreMiDaS"
    WEIGHT_EXT = ".pt"
    # Sizes mirrored on the LibreYOLO org, pinned by the digest of the file the
    # mirror serves. That file is the converted LibreYOLO checkpoint, so the
    # upstream digests in convert.py cannot describe it. The mirror URL resolves
    # `main`, which is mutable, so this pinning is what stops a replaced or
    # corrupted upload from loading silently. Anything absent here still
    # downloads the official release asset and stays pinned against upstream.
    _MIRROR_SHA256: ClassVar[Dict[str, str]] = {
        "s": "c87fe0e7702e6b0e4b84475dcb1e61f5daa5d06b7fc2b78c3d7a1f2d5ad0b960",
        "l": "ccda8065b7184ff4e00357aa9d40f62f131c26c3402ac807f55321c55b18533b",
    }
    INPUT_SIZES: ClassVar[Dict[str, int]] = {"s": 256, "l": 384}
    SUPPORTED_TASKS = ("depth",)
    DEFAULT_TASK = "depth"
    REQUIRE_TASK_SUFFIX = True
    TRAIN_CONFIG = None

    depth_imgsz_divisor = IMGSZ_DIVISOR
    depth_resize_mode = "letterbox"
    SUPPORTS_BATCHED_PREDICT = False

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        has_decoder = (
            "scratch.refinenet1.resConfUnit1.conv1.weight" in weights_dict
            and "scratch.output_conv.4.weight" in weights_dict
        )
        has_dpt = "pretrained.model.cls_token" in weights_dict
        has_small = "pretrained.layer1.3.0.conv_dw.weight" in weights_dict
        return has_decoder and (has_dpt or has_small)

    @classmethod
    def detect_size(cls, weights_dict: dict) -> Optional[str]:
        cls_token = weights_dict.get("pretrained.model.cls_token")
        if cls_token is not None and tuple(cls_token.shape) == (1, 1, 1024):
            return "l"
        stem = weights_dict.get("pretrained.layer1.0.weight")
        if stem is not None and tuple(stem.shape[:2]) == (32, 3):
            return "s"
        return None

    @classmethod
    def detect_nb_classes(cls, weights_dict: dict) -> Optional[int]:
        return 1 if cls.can_load(weights_dict) else None

    @classmethod
    def get_download_url(cls, filename: str) -> Optional[str]:
        """Serve from the LibreYOLO mirror; fall back to the official asset.

        isl-org/MiDaS is MIT and that licence covers the released checkpoints,
        so LibreYOLO redistributes them on its own org rather than depending on
        a GitHub release staying put. ADR 0006's stricter bar, which asks for
        training data permitting commercial use as well as redistribution, is
        superseded for this family: the redistribution rests on the publisher's
        own grant over the bytes, and the training-data caveat is stated on the
        model card and in the notice below instead of blocking the mirror.
        """
        size = cls.detect_size_from_filename(filename)
        task = cls.detect_task_from_filename(filename)
        if size is None or task != "depth":
            return None
        if size in cls._MIRROR_SHA256:
            return super().get_download_url(filename)
        return UPSTREAM_URLS.get(size)

    @classmethod
    def get_download_notice(cls, filename: str, url: str) -> Optional[str]:
        del filename, url
        return (
            "MiDaS was trained on a twelve-dataset mixture whose individual "
            "terms are not all permissive. LibreYOLO redistributes these "
            "weights under the MIT licence isl-org applied to them; if your "
            "use is commercial, satisfy yourself about the training-data terms."
        )

    @classmethod
    def verify_downloaded_file(cls, local_path: str, source_url: str) -> None:
        # Mirror and upstream serve different bytes (converted vs original), so
        # each has its own recorded digest. Both are checked: an unverified
        # mirror would be a weaker guarantee than the upstream path it replaced.
        if source_url.startswith("https://huggingface.co/LibreYOLO/"):
            size = next(
                (s for s in cls._MIRROR_SHA256 if f"LibreMiDaS{s}-depth" in source_url),
                None,
            )
            if size is None:
                raise ValueError(f"Unrecognized MiDaS mirror URL: {source_url}")
            expected = cls._MIRROR_SHA256[size]
            digest = hashlib.sha256()
            with open(local_path, "rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            actual = digest.hexdigest()
            if actual != expected:
                raise ValueError(
                    f"MiDaS {size} mirrored checkpoint SHA-256 mismatch: "
                    f"expected {expected}, got {actual}."
                )
            return
        verify_and_wrap_download(local_path, source_url)

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
        self.names = {0: "depth"}

    def _init_model(self) -> nn.Module:
        return build_midas_model(self.size)

    def _get_available_layers(self) -> Dict[str, nn.Module]:
        return {
            "pretrained": self.model.pretrained,
            "scratch": self.model.scratch,
        }

    def _get_preprocess_numpy(self):
        """Return the variant-bound RGB preprocessor used by export tooling."""
        return partial(preprocess_numpy, size=self.size)

    def _preprocess(
        self,
        image: ImageInput,
        color_format: str = "auto",
        input_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Image.Image, Tuple[int, int], float]:
        effective_res = input_size if input_size is not None else self._get_input_size()
        if effective_res % self.depth_imgsz_divisor:
            raise ValueError(
                f"MiDaS imgsz={effective_res} must be divisible by "
                f"{self.depth_imgsz_divisor}."
            )
        img = ImageLoader.load(image, color_format=color_format).convert("RGB")
        orig_w, orig_h = img.size
        chw, ratio = preprocess_numpy(np.asarray(img), effective_res, self.size)
        return torch.from_numpy(chw).unsqueeze(0), img, (orig_w, orig_h), ratio

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
        del conf_thres, iou_thres, max_det, kwargs
        depth = output
        if isinstance(depth, dict):
            depth = depth.get("depth", depth.get("predictions"))
            if depth is None:
                raise ValueError(
                    "MiDaS output dict must contain 'depth' or 'predictions'."
                )
        if isinstance(depth, (list, tuple)):
            if not depth:
                raise ValueError("MiDaS received an empty output sequence.")
            depth = depth[0]
        depth = torch.as_tensor(depth)
        if depth.ndim == 2:
            depth = depth.unsqueeze(0).unsqueeze(0)
        if depth.ndim == 3:
            depth = depth.unsqueeze(1)
        if depth.ndim != 4 or depth.shape[1] != 1:
            raise ValueError(
                "MiDaS postprocessing expects [B, 1, H, W], [B, H, W], or "
                f"[H, W] depth; got {tuple(depth.shape)}."
            )
        orig_w, orig_h = original_size
        depth = F.interpolate(
            depth.float(),
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=True,
        )
        return {"depth": depth[0, 0].cpu()}

    def train(self, *args, **kwargs):
        raise NotImplementedError(
            "MiDaS training is not implemented in LibreYOLO. This museum-tier "
            "port supports pretrained inference, zero-shot validation, and export."
        )
