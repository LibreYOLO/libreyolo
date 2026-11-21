"""LibreDepthPro2: Apple Depth Pro 2 metric image & video depth in LibreYOLO.

Depth Pro 2 (https://github.com/apple-aiml-research/ml-depth-pro-2, arXiv:2611.26603)
is a multi-view DINOv3 EVA ViT + DPT decoder model paired with a dedicated ViT-S/16
field-of-view estimator (``FOVEstimator``) that predicts sharp, metric depth in metres
and camera focal length for both single images and video sequences at native resolution.

Multi-frame input runs through LibreYOLO's shared sequence runner
(``SEQUENCE_INPUT = True``, ``BaseModel._forward_sequence``):

* Single image -> one-frame clip returning metric depth + camera intrinsics.
* Multi-image / directory / video / stream -> temporally consistent cross-frame
  attention via ``DepthStream`` in either ``mode="buffered"`` (overlapping windows
  with cross-faded seams, default ``window=16, overlap=4``) or ``mode="unbuffered"``
  (causal per-frame ``KVCache``).

When a bundled LibreYOLO checkpoint (containing ``fov.*``) is loaded or an explicit
``f_px`` (or EXIF ``FocalLengthIn35mmFilm``) is available, ``Results.depth_map``
carries metric depth in metres (``encoding="depth"``, ``metric=True``) and OpenCV
pixel intrinsics. Raw upstream depth-only checkpoints without ``fov.*`` also load
strictly and output focal-normalized inverse depth (``encoding="inverse_depth"``,
``metric=False``) unless ``f_px`` is supplied.

Code in this package is governed by the Apple Sample Code License (see ``LICENSE``).
Official pretrained weights are licensed under the Apple Machine Learning Research
Model License (``apple-amlr``, non-commercial research only; see ``LICENSE_MODEL``
and ``NOTICE``). Inference and zero-shot validation only.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from ...utils.image_loader import ImageInput, ImageLoader
from ..base.model import BaseModel
from .convert import (
    INPUT_SIZES,
    UPSTREAM_URLS,
    detect_size,
    has_fov_weights,
    is_depthpro2_state_dict,
    verify_and_wrap_download,
)
from .fov import FOVEstimator
from .nn import (
    DEFAULT_OVERLAP,
    DEFAULT_WINDOW,
    INPUT_MULTIPLE,
    DepthPro2Net,
    VideoMode,
    build_depthpro2_model,
)
from .utils import extract_exif_f_px, normalize_rgb, pixel_intrinsics, preprocess_numpy

_DOWNLOAD_NOTICE = (
    "Depth Pro 2 is licensed under the Apple Machine Learning Research Model "
    "License (apple-amlr), Copyright © 2026 Apple Inc. All Rights Reserved. "
    "Official pretrained weights are downloaded directly from Apple's Hugging Face "
    "repositories and are restricted to non-commercial research use only. "
    "See libreyolo/models/depthpro2/LICENSE_MODEL for full terms."
)

_PRECISION_ALIASES: dict[str, torch.dtype] = {
    "fp32": torch.float32,
    "float32": torch.float32,
    "fp16": torch.float16,
    "float16": torch.float16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
}


def _resolve_precision(precision: torch.dtype | str | None) -> torch.dtype:
    if precision is None:
        return torch.float32
    if isinstance(precision, torch.dtype):
        if precision not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError(
                f"Unsupported Depth Pro 2 precision {precision}; "
                "expected float32, float16, or bfloat16."
            )
        return precision
    key = str(precision).strip().lower()
    if key not in _PRECISION_ALIASES:
        raise ValueError(
            f"Unsupported Depth Pro 2 precision {precision!r}; "
            "expected 'float32', 'float16', or 'bfloat16'."
        )
    return _PRECISION_ALIASES[key]


class LibreDepthPro2(BaseModel):
    """Depth Pro 2 (S/B/L/H): metric monocular & video depth with focal length estimation."""

    FAMILY = "depthpro2"
    FILENAME_PREFIX = "LibreDepthPro2"
    WEIGHT_EXT = ".pt"
    INPUT_SIZES: ClassVar[dict[str, int]] = dict(INPUT_SIZES)
    SUPPORTED_TASKS = ("depth",)
    DEFAULT_TASK = "depth"
    REQUIRE_TASK_SUFFIX = True
    TRAIN_CONFIG = None
    SEQUENCE_INPUT: ClassVar[bool] = True
    SUPPORTS_BATCHED_PREDICT = False
    TTA_ENABLED = False

    depth_imgsz_divisor = INPUT_MULTIPLE
    depth_resize_mode = "letterbox"
    depth_encoding = "depth"
    model: DepthPro2Net

    # ====================================================================
    # Checkpoint detection & pinned upstream downloads
    # ====================================================================

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        return is_depthpro2_state_dict(weights_dict)

    @classmethod
    def detect_size(cls, weights_dict: dict) -> str | None:
        return detect_size(weights_dict)

    @classmethod
    def detect_nb_classes(cls, weights_dict: dict) -> int | None:
        return 1 if cls.can_load(weights_dict) else None

    @classmethod
    def get_download_url(cls, filename: str) -> str | None:
        """Resolve canonical ``LibreDepthPro2{s,b,l,h}-depth.pt`` to Apple's pinned HF asset."""
        size = cls.detect_size_from_filename(filename)
        task = cls.detect_task_from_filename(filename)
        if size is None or task != "depth":
            return None
        return UPSTREAM_URLS.get(size)

    @classmethod
    def get_download_notice(cls, filename: str, url: str) -> str | None:
        del filename, url
        return _DOWNLOAD_NOTICE

    @classmethod
    def verify_downloaded_file(cls, local_path: str, source_url: str) -> None:
        verify_and_wrap_download(local_path, source_url)

    # ====================================================================
    # Construction
    # ====================================================================

    def __init__(
        self,
        model_path,
        size: str = "s",
        nb_classes: int = 1,
        device: str = "auto",
        task: str | None = None,
        *,
        fov: bool = True,
        mode: VideoMode = "buffered",
        window: int = DEFAULT_WINDOW,
        overlap: int = DEFAULT_OVERLAP,
        f_px: float | None = None,
        use_exif: bool = True,
        precision: torch.dtype | str | None = None,
        **kwargs,
    ):
        if mode not in ("buffered", "unbuffered"):
            raise ValueError(
                f"Depth Pro 2 mode must be 'buffered' or 'unbuffered', got {mode!r}."
            )
        if window < 1:
            raise ValueError(f"Depth Pro 2 window must be at least 1, got {window}.")
        if mode == "buffered" and not 0 <= overlap < window:
            raise ValueError(
                f"Depth Pro 2 overlap must be in [0, {window - 1}], got {overlap}."
            )
        self._use_fov = bool(fov)
        self.mode: VideoMode = mode
        self.window = int(window)
        self.overlap = int(overlap)
        self.f_px: float | None = float(f_px) if f_px is not None else None
        self.use_exif = bool(use_exif)
        self.precision = _resolve_precision(precision)
        self._pending_exif_f_px: list[float | None] = []

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
        self._sync_depth_encoding()
        self.nb_classes = 1
        self.names = {0: "depth"}

    def _sync_depth_encoding(self) -> None:
        """Keep ``depth_encoding`` consistent with whether metric depth is producible."""
        has_focal_source = self.model.fov is not None or self.f_px is not None
        self.depth_encoding = "depth" if has_focal_source else "inverse_depth"

    def _init_model(self) -> nn.Module:
        return build_depthpro2_model(
            self.size, fov=self._use_fov, precision=self.precision
        )

    def _prepare_state_dict(self, state_dict: dict) -> dict:
        if not self._use_fov:
            return {k: v for k, v in state_dict.items() if not k.startswith("fov.")}
        return state_dict

    def _prepare_model_for_state_dict(self, state_dict: dict) -> None:
        """Adapt the optional ``fov`` submodule to match the checkpoint being loaded."""
        ckpt_has_fov = has_fov_weights(state_dict)
        if not ckpt_has_fov and self.model.fov is not None:
            self.model.fov = None
        elif ckpt_has_fov and self._use_fov and self.model.fov is None:
            self.model.fov = FOVEstimator(precision=self.precision).to(self.device)
        self._sync_depth_encoding()

    def _get_available_layers(self) -> dict[str, nn.Module]:
        layers: dict[str, nn.Module] = {
            "encoder": self.model.encoder,
            "decoder": self.model.decoder,
            "head": self.model.head,
        }
        if self.model.fov is not None:
            layers["fov"] = self.model.fov
        return layers

    # ====================================================================
    # Inference surface
    # ====================================================================

    @staticmethod
    def _get_preprocess_numpy():
        return preprocess_numpy

    def _preprocess(
        self,
        image: ImageInput,
        color_format: str = "auto",
        input_size: int | None = None,
    ) -> tuple[torch.Tensor, Image.Image, tuple[int, int], float]:
        del input_size
        img = ImageLoader.load(image, color_format=color_format)
        orig_w, orig_h = img.size
        exif_f_px: float | None = None
        if self.use_exif and self.f_px is None:
            exif_f_px = extract_exif_f_px(img, orig_w, orig_h)
            if exif_f_px is None and isinstance(image, (str, Path)):
                try:
                    with Image.open(image) as raw_img:
                        exif_f_px = extract_exif_f_px(raw_img, orig_w, orig_h)
                except (OSError, ValueError, TypeError, AttributeError):
                    exif_f_px = None
        self._pending_exif_f_px.append(exif_f_px)
        chw, ratio = preprocess_numpy(np.asarray(img.convert("RGB")))
        return torch.from_numpy(chw).unsqueeze(0), img, (orig_w, orig_h), ratio

    def _forward_sequence(
        self, frames: Iterable[tuple[torch.Tensor, tuple[int, int]]]
    ) -> Iterator[dict[str, Any]]:
        """Run Depth Pro 2 across a single image or multi-frame video sequence."""
        self._pending_exif_f_px.clear()
        param = next(self.model.parameters())
        device = param.device
        first_exif_f_px: list[float | None] = []

        def normalized_frames() -> Iterator[torch.Tensor]:
            for tensor, _orig_size in frames:
                exif_val = (
                    self._pending_exif_f_px.pop(0) if self._pending_exif_f_px else None
                )
                if not first_exif_f_px:
                    first_exif_f_px.append(exif_val)
                frame_3hw = tensor[0].to(device=device, dtype=torch.float32)
                yield normalize_rgb(frame_3hw)

        # Peek the first frame so a single photo with EXIF focal length uses it
        # identically to upstream CLI, while videos without EXIF use FOVEstimator.
        frame_iter = normalized_frames()
        try:
            first_frame = next(frame_iter)
        except StopIteration:
            return

        effective_f_px = self.f_px
        if effective_f_px is None and first_exif_f_px and first_exif_f_px[0] is not None:
            effective_f_px = first_exif_f_px[0]

        def all_frames() -> Iterator[torch.Tensor]:
            yield first_frame
            yield from frame_iter

        with torch.no_grad():
            for prediction in self.model.infer_video(
                all_frames(),
                window=self.window,
                overlap=self.overlap,
                f_px=effective_f_px,
                mode=self.mode,
            ):
                yield dict(prediction)

    def _forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Independent images ``(B, 3, H, W)`` in ``[0, 1]`` to ``(B, 1, H, W)`` depth.

        Used by ``DepthValidator`` and any direct tensor caller: each image runs
        independently so unrelated validation samples do not cross-attend.
        """
        param = next(self.model.parameters())
        norm = normalize_rgb(input_tensor.to(device=param.device, dtype=torch.float32))
        outputs = []
        for idx in range(norm.shape[0]):
            pred = self.model.infer(norm[idx], f_px=self.f_px)
            map_2d = pred["depth"] if pred["depth"] is not None else pred["inverse_depth"]
            if map_2d is None:
                raise RuntimeError("DepthPro2.infer returned neither depth nor inverse_depth.")
            outputs.append(map_2d.unsqueeze(0).unsqueeze(0))
        return torch.cat(outputs, dim=0)

    def _postprocess(
        self,
        output: Any,
        conf_thres: float,
        iou_thres: float,
        original_size: tuple[int, int],
        max_det: int = 300,
        ratio: float = 1.0,
        **kwargs,
    ) -> dict[str, Any]:
        del conf_thres, iou_thres, max_det, ratio, kwargs
        orig_w, orig_h = original_size

        if isinstance(output, dict) and "inverse_depth" in output:
            depth = output.get("depth")
            f_px = output.get("focallength_px")
            if depth is not None:
                if tuple(depth.shape[-2:]) != (orig_h, orig_w):
                    raise ValueError(
                        f"Depth Pro 2 output shape {tuple(depth.shape)} does not match "
                        f"input canvas ({orig_h}, {orig_w})."
                    )
                result: dict[str, Any] = {
                    "depth": depth.float().cpu(),
                    "depth_encoding": "depth",
                    "depth_metric": True,
                }
                if f_px is not None:
                    result["intrinsics"] = pixel_intrinsics(f_px, orig_w, orig_h).cpu()
                return result

            inv_depth = output["inverse_depth"]
            if tuple(inv_depth.shape[-2:]) != (orig_h, orig_w):
                raise ValueError(
                    f"Depth Pro 2 inverse-depth shape {tuple(inv_depth.shape)} does not "
                    f"match input canvas ({orig_h}, {orig_w})."
                )
            return {
                "depth": inv_depth.float().cpu(),
                "depth_encoding": "inverse_depth",
                "depth_metric": False,
            }

        tensor = torch.as_tensor(output)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0).unsqueeze(0)
        elif tensor.ndim == 3:
            tensor = tensor.unsqueeze(1)
        if tensor.ndim != 4 or tensor.shape[1] != 1:
            raise ValueError(
                "Depth Pro 2 postprocessing expects a prediction dict or "
                f"[B, 1, H, W] tensor, got {tuple(tensor.shape)}."
            )
        if tuple(tensor.shape[-2:]) != (orig_h, orig_w):
            tensor = F.interpolate(
                tensor.float(),
                size=(orig_h, orig_w),
                mode="bilinear",
                align_corners=False,
            )
        is_metric = self.depth_encoding == "depth"
        return {
            "depth": tensor[0, 0].float().cpu(),
            "depth_encoding": self.depth_encoding,
            "depth_metric": is_metric,
        }

    # ====================================================================
    # Out of scope
    # ====================================================================

    def train(self, *args, **kwargs):
        raise NotImplementedError(
            "Training and fine-tuning Depth Pro 2 are not supported by LibreYOLO. "
            "Use a pretrained checkpoint, or convert official weights with "
            "weights/convert_depthpro2_weights.py."
        )

    def export(self, format: str = "onnx", **kwargs) -> str:
        raise NotImplementedError(
            "Depth Pro 2 export is not supported: the network uses dynamic "
            "multi-frame cross-attention with host-side window streaming and an "
            "auxiliary variable-resolution FoV estimator, which the single-image "
            "depth export contract does not describe."
        )


__all__ = ["LibreDepthPro2"]
