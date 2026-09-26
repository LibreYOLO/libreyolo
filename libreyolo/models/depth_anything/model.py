"""LibreYOLO wrapper for Depth Anything V2 monocular depth estimation.

Depth Anything V2 (https://github.com/DepthAnything/Depth-Anything-V2, NeurIPS
2024) is a DINOv2 encoder + DPT head that predicts a dense relative inverse
depth map. That is exactly LibreYOLO's ``depth`` task primitive
(``Results.depth_map``, higher = closer, no metric unit), so this family reuses
the shared depth dataset, validator, results, and checkpoint schema: ``predict``
and zero-shot ``val`` (AbsRel / RMSE / delta) work out of the box.

The vendored upstream code is Apache-2.0 (``_vendor/``). Depth Anything V2
*weights* are split: the Small encoder is Apache-2.0, while Base/Large/Giant are
CC-BY-NC-4.0 (non-commercial), so LibreYOLO does not redistribute them - convert
the official checkpoints with ``weights/convert_depth_anything_v2_weights.py``.

Training is out of scope here (see :meth:`LibreDepthAnythingV2.train`).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, ClassVar, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from ...utils.image_loader import ImageInput, ImageLoader
from ..base.model import BaseModel
from .nn import LibreDepthAnythingV2Net
from .utils import preprocess_numpy

logger = logging.getLogger(__name__)


class LibreDepthAnythingV2(BaseModel):
    """Depth Anything V2: image -> dense relative inverse-depth map."""

    FAMILY = "depth_anything"
    FILENAME_PREFIX = "LibreDepthAnythingV2"
    WEIGHT_EXT = ".pt"
    # DA V2 always runs at 518 (37 * 14). Size codes follow the encoder:
    # s / b / l / g -> ViT-S / B / L / G.
    INPUT_SIZES: ClassVar[Dict[str, int]] = {
        "s": 518,
        "b": 518,
        "l": 518,
        "g": 518,
    }
    # Size g is accepted for local conversions only.
    UNPUBLISHED_WEIGHTS: ClassVar[Dict[str, str]] = {
        "LibreDepthAnythingV2g-depth": (
            "The ViT-G checkpoint is not publicly released upstream; pass a "
            "local checkpoint or use LibreDepthAnythingV2l-depth.pt."
        ),
    }
    SUPPORTED_TASKS = ("depth",)
    # Forward is pure tensor work with no host sync, verified to capture and
    # replay bit-identically (tests/unit/test_cuda_graph_families.py).
    SUPPORTS_CUDA_GRAPH = True
    DEFAULT_TASK = "depth"
    REQUIRE_TASK_SUFFIX = True

    # DINOv2 patch grid; the depth dataset and validator enforce divisibility.
    depth_imgsz_divisor = 14
    # Validation goes through the shared DepthDataset, which letterboxes to a
    # fixed square (padded pixels are invalid depth, excluded from metrics).
    # Note: predict uses DA's native keep-aspect lower-bound resize, so val
    # metrics on non-square images are a documented approximation of predict.
    depth_resize_mode = "letterbox"

    # Keep-aspect preprocessing yields per-image variable sizes, so the stacked
    # single-forward batched-predict path does not apply. TTA_ENABLED is left at
    # its default (True) so predict(augment=True) reaches the depth task's
    # explicit rejection instead of being silently ignored.
    SUPPORTS_BATCHED_PREDICT = False

    _SIZE_TO_ENCODER: ClassVar[Dict[str, str]] = {
        "s": "vits",
        "b": "vitb",
        "l": "vitl",
        "g": "vitg",
    }
    # The DINOv2 cls-token width uniquely identifies the encoder size.
    _EMBED_DIM_TO_SIZE: ClassVar[Dict[int, str]] = {
        384: "s",
        768: "b",
        1024: "l",
        1536: "g",
    }

    # ====================================================================
    # Checkpoint detection
    # ====================================================================

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        # DA V2 checkpoints carry the DINOv2 encoder under ``pretrained.*`` and
        # the DPT head under ``depth_head.*``. Requiring both prefixes prevents
        # unrelated depth heads from being detected as DA V2 checkpoints.
        has_encoder = any(k.startswith("pretrained.") for k in weights_dict)
        has_head = any(k.startswith("depth_head.") for k in weights_dict)
        return has_encoder and has_head

    @classmethod
    def detect_size(cls, weights_dict: dict) -> Optional[str]:
        cls_token = weights_dict.get("pretrained.cls_token")
        if cls_token is not None and getattr(cls_token, "ndim", 0) >= 1:
            return cls._EMBED_DIM_TO_SIZE.get(int(cls_token.shape[-1]))
        return None

    @classmethod
    def detect_nb_classes(cls, weights_dict: dict) -> Optional[int]:
        # The single class slot is a schema placeholder; depth has no classes.
        return 1

    # ====================================================================
    # Construction
    # ====================================================================

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
            nb_classes=1,  # always 1 ("depth"); ignore any caller-provided value
            device=device,
            task=task,
            **kwargs,
        )
        self.model.eval()
        # A file path is stored but not loaded by BaseModel; load it here (a raw
        # state_dict passed as ``model_path`` is already loaded by BaseModel).
        if model_path is not None and isinstance(model_path, (str, Path)):
            self._load_weights(str(model_path))
        # Enforce the single depth slot last, after any loading path. Raw .pth
        # files loaded via the generic autoconverter would otherwise write
        # names={0: "class_0"} (build_class_names fallback), violating the
        # checkpoint schema's {0: "depth"} for this task.
        self.nb_classes = 1
        self.names = {0: "depth"}

    def _init_model(self) -> nn.Module:
        return LibreDepthAnythingV2Net(self._SIZE_TO_ENCODER[self.size])

    def _get_available_layers(self) -> Dict[str, nn.Module]:
        return {
            "pretrained": self.model.pretrained,
            "depth_head": self.model.depth_head,
        }

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
        input_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Image.Image, Tuple[int, int], float]:
        effective_res = input_size if input_size is not None else self._get_input_size()
        if effective_res % self.depth_imgsz_divisor:
            raise ValueError(
                f"Depth Anything V2 imgsz={effective_res} must be divisible by "
                f"{self.depth_imgsz_divisor} (DINOv2 patch grid)."
            )
        img = ImageLoader.load(image, color_format=color_format)
        orig_w, orig_h = img.size
        chw, ratio = preprocess_numpy(np.asarray(img), effective_res)
        img_tensor = torch.from_numpy(chw).unsqueeze(0)
        return img_tensor, img, (orig_w, orig_h), ratio

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
        depth = output
        if isinstance(depth, dict):
            depth = depth.get("depth", depth.get("predictions"))
        orig_w, orig_h = original_size
        # align_corners=True matches upstream DA infer_image, so converted
        # checkpoints reproduce upstream depth maps at non-native sizes.
        depth = F.interpolate(
            depth.float(),
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=True,
        )
        return {"depth": depth[0, 0].cpu()}

    # ====================================================================
    # Weights I/O
    # ====================================================================
    # Weights load (and auto-download) through BaseModel. A missing checkpoint
    # is fetched from the LibreYOLO Hugging Face org via the inherited
    # ``get_download_url`` -> ``LibreYOLO/LibreDepthAnythingV2{s,l}-depth``,
    # which mirrors the converted checkpoints. The offline path is still to
    # convert an upstream checkpoint with
    # ``weights/convert_depth_anything_v2_weights.py``. Note Base/Large/Giant
    # (ViT-B/L/G) are CC-BY-NC-4.0 (non-commercial); see THIRD_PARTY_NOTICES.txt.

    # ====================================================================
    # Out of scope
    # ====================================================================

    def train(self, *args, **kwargs):
        raise NotImplementedError(
            "Training and fine-tuning Depth Anything V2 are not supported by "
            "LibreYOLO. Use a pretrained LibreYOLO checkpoint, or convert a "
            "compatible pretrained upstream checkpoint with "
            "weights/convert_depth_anything_v2_weights.py."
        )

    def export(self, format: str = "onnx", **kwargs) -> str:
        """Export through the shared fixed-resolution depth contract."""
        return super().export(format=format, **kwargs)
