"""LibreYOLO wrapper for MoGe-3 monocular geometry.

MoGe-3 is Microsoft's monocular geometry model: one RGB image to a metric
camera-space point map, depth, a validity mask, camera intrinsics, and surface
normals. It extends MoGe-2 with a sparse 3D UNet that refines the predicted
log-depth on a voxel grid for a few steps. LibreYOLO ships the official ViT-L
and ViT-g checkpoints as an inference, validation, and export family.

Device notes: the whole graph, including the sparse refiner, is plain PyTorch
and runs on CPU, CUDA, and MPS; no compiled extension is used. Camera
recovery solves a tiny least-squares problem with SciPy on the CPU, as
upstream does. On MPS the 4x4 camera matrix inverse runs on the CPU. The
refiner's neighbor tables are int64 index tensors over every output pixel, so
its memory grows with the token budget (``imgsz``), not the input resolution.
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
from .nn import PATCH_SIZE, TASK_MODULES, MoGe3Net
from .utils import num_tokens_for, pixel_intrinsics, preprocess_numpy, recover_geometry

_WRAPPER_PREFIXES = ("module.", "_orig_mod.", "model.")


def _clean_state_dict(state_dict: dict, task: str | None = None) -> dict:
    """Strip wrapper prefixes; with ``task``, keep only that graph's tensors."""
    prefixes = tuple(f"{name}." for name in TASK_MODULES[task]) if task else None
    cleaned = {}
    for key, value in state_dict.items():
        while key.startswith(_WRAPPER_PREFIXES):
            key = key.split(".", 1)[1]
        if prefixes is None or key.startswith(prefixes):
            cleaned[key] = value
    return cleaned


class LibreMoGe3(BaseModel):
    """MoGe-3 ViT-L/ViT-g: metric depth with point map and intrinsics, or normals.

    ``task="depth"`` returns ``Results.depth_map`` holding metric depth in
    metres (``encoding="depth"``, ``metric=True``) plus the camera-space point
    map, OpenCV pixel intrinsics, and validity mask on the original canvas.
    Pixels outside the predicted mask are ``inf``. ``task="normal"`` returns
    OpenCV-frame unit normals in ``Results.normal_map``.

    ``imgsz`` sets the ViT token budget, ``(imgsz // 14) ** 2`` tokens laid
    out to the image aspect ratio; the default 840 is upstream's highest
    resolution level (3600 tokens). Inputs keep their native resolution and
    outputs are interpolated back to it. Inference runs in FP32, matching
    upstream defaults; the refiner always runs three steps.
    """

    FAMILY = "moge3"
    FILENAME_PREFIX = "LibreMoGe3"
    WEIGHT_EXT = ".pt"
    INPUT_SIZES: ClassVar[Dict[str, int]] = {"l": 840, "g": 840}
    SUPPORTED_TASKS = ("depth", "normal")
    DEFAULT_TASK = "depth"
    REQUIRE_TASK_SUFFIX = True
    TRAIN_CONFIG = None
    SUPPORTS_BATCHED_PREDICT = False
    TTA_ENABLED = False

    depth_imgsz_divisor = PATCH_SIZE
    normal_imgsz_divisor = PATCH_SIZE
    depth_resize_mode = "letterbox"
    normal_resize_mode = "letterbox"
    depth_encoding = "depth"

    _EMBED_DIM_TO_SIZE: ClassVar[Dict[int, str]] = {1024: "l", 1536: "g"}

    # Converted checkpoints mirrored on the LibreYOLO Hugging Face org, keyed
    # by (size, task). The mirror URL resolves `main`, so each file is
    # checksum-pinned; regenerate a value only on a deliberate re-upload.
    _MIRROR_SHA256: ClassVar[Dict[Tuple[str, str], str]] = {}

    # Official checkpoints at pinned revisions. One file per size serves both
    # tasks; the loader keeps the tensors the requested task needs.
    _OFFICIAL_WEIGHTS: ClassVar[Dict[str, tuple[str, str]]] = {
        "l": (
            "https://huggingface.co/Ruicheng/moge-3-vitl/resolve/"
            "184008f877d7ad1ad4c2cd2182a9bd1f63d0e5be/model.pt",
            "9b41b7b9f65ad80aab7ad686f5e9cc0d1fd33f1964022618dfbcd52fc1fb7925",
        ),
        "g": (
            "https://huggingface.co/Ruicheng/moge-3-vitg/resolve/"
            "6ef26c5a4b4148dab5ccaacdb08b72dc66380475/model.pt",
            "ce7c15417e9105c2ace7b4272e2cc69e36940921211eb7fa05d4d0bb03f0a00c",
        ),
    }

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        # The refiner is what separates MoGe-3 from MoGe-2, whose encoder,
        # neck, and heads share these names (and, at ViT-L, these shapes).
        return (
            "refiner.input_proj.weight" in weights_dict
            and "encoder.backbone.cls_token" in weights_dict
            and "neck.input_blocks.0.weight" in weights_dict
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
    def convert_upstream_state_dict(cls, state_dict: dict) -> Optional[dict]:
        cleaned = _clean_state_dict(state_dict)
        return cleaned if cls.can_load(cleaned) else None

    @classmethod
    def get_download_url(cls, filename: str) -> Optional[str]:
        size = cls.detect_size_from_filename(filename)
        task = cls.detect_task_from_filename(filename)
        if size is None or task not in cls.SUPPORTED_TASKS:
            return None
        if (size, task) in cls._MIRROR_SHA256:
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
        if source_url.startswith("https://huggingface.co/LibreYOLO/"):
            key = next(
                (
                    (size, task)
                    for size, task in cls._MIRROR_SHA256
                    if f"/{cls.FILENAME_PREFIX}{size}-{task}/" in source_url
                ),
                None,
            )
            if key is None:
                raise ValueError(f"Unrecognized MoGe-3 mirror URL: {source_url}")
            expected = cls._MIRROR_SHA256[key]
        else:
            size = next(
                (
                    s
                    for s, (url, _) in cls._OFFICIAL_WEIGHTS.items()
                    if url == source_url
                ),
                None,
            )
            if size is None:
                raise ValueError(f"Unrecognized MoGe-3 weight URL: {source_url}")
            expected = cls._OFFICIAL_WEIGHTS[size][1]
        actual = cls._sha256(local_path)
        if actual != expected:
            raise ValueError(
                f"MoGe-3 checkpoint SHA-256 mismatch for {source_url}: "
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
        self._pending_num_tokens: int | None = None
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
        self.names = {0: self.task}

    def _allow_checkpoint_task_mismatch(self, checkpoint_task: str) -> bool:
        # Every MoGe-3 checkpoint carries all heads, so either task loads it.
        return checkpoint_task in self.SUPPORTED_TASKS

    def _prepare_state_dict(self, state_dict: dict) -> dict:
        selected = _clean_state_dict(state_dict, self.task)
        return selected if selected else state_dict

    def _init_model(self) -> nn.Module:
        return MoGe3Net(size=self.size, task=self.task)

    def _get_available_layers(self) -> Dict[str, nn.Module]:
        return {
            name: getattr(self.model, name)
            for name in TASK_MODULES[self.task]
            if hasattr(self.model, name)
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
        self._pending_num_tokens = num_tokens_for(effective_res)
        img = ImageLoader.load(image, color_format=color_format)
        chw, ratio = preprocess_numpy(np.asarray(img.convert("RGB")), effective_res)
        return torch.from_numpy(chw).unsqueeze(0), img, img.size, ratio

    def _forward(self, input_tensor: torch.Tensor) -> Any:
        # predict() sets the token budget from imgsz; validation and export
        # feed a patch-aligned canvas whose own patch grid is the budget.
        num_tokens, self._pending_num_tokens = self._pending_num_tokens, None
        output = self.model(input_tensor, num_tokens=num_tokens)
        if self.task == "normal":
            return output
        geometry = recover_geometry(
            output["points"], output["mask"], output["metric_scale"]
        )
        return {
            "depth": geometry["depth"],
            "points": geometry["points"],
            "intrinsics": geometry["intrinsics"],
            "mask": geometry["mask"],
        }

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
        orig_w, orig_h = original_size
        if self.task == "normal":
            return {"normal": self._postprocess_normal(output, orig_h, orig_w)}

        depth, points, mask = output["depth"], output["points"], output["mask"]
        if tuple(depth.shape[-2:]) != (orig_h, orig_w):
            raise ValueError(
                "MoGe-3 depth output must match the input canvas; got "
                f"{tuple(depth.shape[-2:])} for a {orig_h}x{orig_w} image."
            )
        intrinsics = pixel_intrinsics(output["intrinsics"][0].float(), orig_w, orig_h)
        return {
            "depth": depth[0].float().cpu(),
            "depth_encoding": self.depth_encoding,
            "depth_metric": True,
            "point_map": points[0].float().cpu(),
            "intrinsics": intrinsics.cpu(),
            "depth_mask": mask[0].cpu(),
        }

    @staticmethod
    def _postprocess_normal(normal: torch.Tensor, orig_h: int, orig_w: int):
        if normal.ndim != 4 or normal.shape[1] != 3:
            raise ValueError(
                f"MoGe-3 normal output must be [B, 3, H, W], got {tuple(normal.shape)}."
            )
        if tuple(normal.shape[-2:]) != (orig_h, orig_w):
            normal = F.interpolate(
                normal.float(),
                size=(orig_h, orig_w),
                mode="bilinear",
                align_corners=False,
            )
        normal = normal.float()
        finite = torch.isfinite(normal).all(dim=1, keepdim=True)
        safe = torch.where(finite, normal, 0.0)
        norms = torch.linalg.vector_norm(safe, dim=1, keepdim=True)
        valid = finite & (norms > 1e-12)
        unit = safe / norms.clamp_min(1e-12)
        fallback = torch.zeros_like(unit)
        fallback[:, 2] = -1.0
        unit = torch.where(valid, unit, fallback)
        return unit[0].permute(1, 2, 0).cpu()

    def train(self, *args, **kwargs):
        raise NotImplementedError(
            "MoGe-3 training is not part of LibreYOLO: the upstream recipe is a "
            "multi-dataset pretraining pipeline that needs camera-calibrated "
            "point-map labels. Use a pinned official checkpoint for predict, "
            "val, and export."
        )

    def export(self, format: str = "onnx", **kwargs) -> str:
        """Export the fixed-canvas normal graph; the depth graph is not capturable."""
        if self.task == "depth":
            raise NotImplementedError(
                "MoGe-3 depth export is not supported: the sparse refiner builds "
                "data-dependent voxel sets and camera recovery runs a CPU "
                "least-squares solve. Export the normal task, or use the native "
                ".pt model for depth."
            )
        return super().export(format=format, **kwargs)


__all__ = ["LibreMoGe3"]
