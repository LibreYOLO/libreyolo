"""LibreYOLO wrapper for DPFlow optical flow.

DPFlow (Morimitsu et al., "DPFlow: Adaptive Optical Flow Estimation with a
Dual-Pyramid Framework", CVPR 2025) is a recurrent flow estimator whose
dual-pyramid encoder adapts its depth to the input size: the number of pyramid
levels is chosen from the image diagonal (3 levels around 1K, one more per
doubling), so the model generalises to 4K and 8K inputs while being trained at
low resolution. LibreYOLO keeps that rule and always runs at the native input
resolution; ``imgsz`` does not resize the frames.

The vendored code (``_vendor/``) comes from ptlflow
(https://github.com/hmorimitsu/ptlflow, ``ptlflow/models/dpflow``) under the
Apache License 2.0, with the timm-derived ``local_timm`` helpers (Apache-2.0)
and ptlflow's RAFT-derived input padder (BSD-3-Clause). The pretrained weights
were trained within ptlflow and are released under CC BY-NC-SA 4.0
(non-commercial); LibreYOLO mirrors them with that licence.

Canonical checkpoint ``LibreDPFlowm-flow.pt`` is the FlyingThings3D stage
(``things``), the standard generalisation checkpoint. ``-sintel``,
``-kitti`` and ``-spring`` are the benchmark fine-tunes.

The iteration schedule is fixed at upstream's test setting (4 refinement
iterations per pyramid level). Warm start across frames (upstream's
``prev_preds``) is not used: every pair is estimated independently.
Training is out of scope (see :meth:`LibreDPFlow.train`).
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
from ._vendor.dpflow import DPFlow
from .utils import preprocess_numpy

logger = logging.getLogger(__name__)

_WEIGHT_LICENSE_NOTICE = (
    "DPFlow weights are licensed CC BY-NC-SA 4.0 (non-commercial, share-alike) "
    "by their ptlflow author. You are responsible for complying with that "
    "licence; see https://huggingface.co/LibreYOLO/{repo}."
)


class LibreDPFlow(BaseModel):
    """DPFlow: frame pair -> dense optical flow on the original canvas."""

    FAMILY = "dpflow"
    FILENAME_PREFIX = "LibreDPFlow"
    WEIGHT_EXT = ".pt"
    # Nominal 1K reference (the size the pyramid rule is anchored on). Predict
    # always runs at the native frame size; this value is only metadata.
    INPUT_SIZES: ClassVar[Dict[str, int]] = {"m": 960}
    SUPPORTED_TASKS = ("flow",)
    DEFAULT_TASK = "flow"
    REQUIRE_TASK_SUFFIX = True
    WEIGHT_VARIANTS: ClassVar[tuple[str, ...]] = ("sintel", "kitti", "spring")
    FLOW_CONTEXT: ClassVar[tuple[int, int]] = (0, 0)
    # Native-size frames are not stackable across images of different size,
    # and test-time augmentation has no flow contract.
    SUPPORTS_BATCHED_PREDICT = False
    TTA_ENABLED = False

    # Upstream test schedule: refinement iterations per pyramid level.
    ITERS_PER_LEVEL: ClassVar[int] = 4

    # ====================================================================
    # Checkpoint detection
    # ====================================================================

    _SIGNATURE_KEYS: ClassVar[tuple[str, ...]] = (
        "fnet.forward_gru.convz.weight",
        "fnet.backward_gru.convz.weight",
        "fnet.lowres_stem.conv1.weight",
        "fnet.out_merge_conv.weight",
    )

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        # The bidirectional GRU pair, the low-resolution stem and the output
        # merge conv of the dual-pyramid encoder are unique to DPFlow; RAFT-style
        # families share fnet.* and update_block.* but not these.
        return all(k in weights_dict for k in cls._SIGNATURE_KEYS) and any(
            k.startswith("update_block.gru_list.") for k in weights_dict
        )

    @classmethod
    def detect_size(cls, weights_dict: dict) -> Optional[str]:
        return "m" if cls.can_load(weights_dict) else None

    @classmethod
    def detect_nb_classes(cls, weights_dict: dict) -> Optional[int]:
        return 1

    @classmethod
    def get_download_notice(cls, filename: str, url: str) -> Optional[str]:
        repo = Path(filename).name.rsplit(cls.WEIGHT_EXT, 1)[0]
        return _WEIGHT_LICENSE_NOTICE.format(repo=repo)

    # ====================================================================
    # Construction
    # ====================================================================

    def __init__(
        self,
        model_path,
        size: str = "m",
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
        self.names = {0: "flow"}

    def _init_model(self) -> nn.Module:
        return DPFlow(iters_per_level=self.ITERS_PER_LEVEL)

    def _get_available_layers(self) -> Dict[str, nn.Module]:
        return {
            "fnet": self.model.fnet,
            "update_block": self.model.update_block,
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
        # input_size is accepted for the runner contract but not applied: the
        # dual pyramid adapts to the native resolution (see module docstring).
        img = ImageLoader.load(image, color_format=color_format)
        orig_w, orig_h = img.size
        chw, ratio = preprocess_numpy(np.asarray(img.convert("RGB")))
        return torch.from_numpy(chw).unsqueeze(0), img, (orig_w, orig_h), ratio

    def _forward_pair(self, image1: torch.Tensor, image2: torch.Tensor) -> torch.Tensor:
        """Flow ``(B, 2, H, W)`` from ``image1`` to ``image2`` (RGB in [0, 1])."""
        return self.model(image1, image2)

    def _forward_flow(self, window: torch.Tensor) -> torch.Tensor:
        return self._forward_pair(window[0:1], window[1:2])

    def _forward(self, input_tensor: torch.Tensor) -> Any:
        if input_tensor.dim() == 4 and input_tensor.shape[0] == 2:
            return self._forward_flow(input_tensor)
        raise ValueError(
            "DPFlow estimates optical flow between two frames; pass a sequence "
            "of at least two images (a list, a directory or a video), not a "
            "single image."
        )

    def _postprocess(
        self,
        output: Any,
        conf_thres: float,
        iou_thres: float,
        original_size: Tuple[int, int],
        max_det: int = 300,
        **kwargs,
    ) -> Dict:
        flow = output
        if isinstance(flow, dict):
            flow = flow["flow"]
        flow = flow.float()
        orig_w, orig_h = original_size
        in_h, in_w = flow.shape[-2:]
        if (in_h, in_w) != (orig_h, orig_w):
            flow = F.interpolate(
                flow, size=(orig_h, orig_w), mode="bilinear", align_corners=True
            )
            scale = flow.new_tensor([orig_w / in_w, orig_h / in_h]).view(1, 2, 1, 1)
            flow = flow * scale
        return {"flow": flow[0].permute(1, 2, 0).contiguous().cpu()}

    # ====================================================================
    # Out of scope
    # ====================================================================

    def train(self, *args, **kwargs):
        raise NotImplementedError(
            "Training and fine-tuning DPFlow are not supported by LibreYOLO; "
            "the flow task is inference-only. Use a pretrained LibreYOLO "
            "checkpoint or convert a ptlflow checkpoint with "
            "weights/convert_dpflow_weights.py."
        )
