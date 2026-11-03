"""LibreYOLO wrapper for MEMFOF three-frame optical flow.

MEMFOF (https://github.com/msu-video-group/memfof, ICCV 2025 Highlight) is a
memory-efficient multi-frame RAFT descendant: it reads frames ``t-1, t, t+1``
and predicts the backward (``t`` to ``t-1``) and forward (``t`` to ``t+1``)
flow of the middle frame in one pass. LibreYOLO's ``flow`` task reports flow
from frame ``i`` to ``i+1``, so the family declares ``FLOW_CONTEXT = (1, 1)``
(one frame of context on each side, edge-padded by the runner at sequence
boundaries) and emits the forward flow as ``Results.flow``. The backward flow
is computed by the network but kept out of the task contract for now.

Frames run at their native resolution, as upstream does, with upstream's
replicate padding to a multiple of 16 inside the network. Frames whose long
side exceeds ``imgsz`` (default 1920, MEMFOF's Full HD target) are downscaled
first and the flow is resized back and its vectors rescaled. Inference is
fp32 with 8 refinement iterations, upstream's defaults.

The vendored upstream code is BSD-3-Clause (``_vendor/LICENSE``). The upstream
checkpoints carry no separate licence statement; LibreYOLO mirrors them under
the code repository's BSD-3-Clause licence. Inference-only.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, ClassVar, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from ...utils.image_loader import ImageInput, ImageLoader
from ..base.model import BaseModel
from ._vendor import MEMFOF
from .utils import MIN_SIDE, flow_to_original, preprocess_numpy

logger = logging.getLogger(__name__)


class LibreMEMFOF(BaseModel):
    """MEMFOF: frames (t-1, t, t+1) to the forward optical flow of frame t."""

    FAMILY = "memfof"
    FILENAME_PREFIX = "LibreMEMFOF"
    WEIGHT_EXT = ".pt"
    # One released architecture (ResNet-34 FPN, dim 512). ``imgsz`` is the cap
    # on the long side, not a fixed square: smaller frames run natively.
    INPUT_SIZES: ClassVar[Dict[str, int]] = {"m": 1920}
    SUPPORTED_TASKS = ("flow",)
    DEFAULT_TASK = "flow"
    REQUIRE_TASK_SUFFIX = True
    # Bare LibreMEMFOFm-flow.pt is upstream's MEMFOF-Tartan-T-TSKH, which the
    # authors recommend for real-world video. The variants are the other
    # published stages and benchmark fine-tunes of the same network.
    WEIGHT_VARIANTS: ClassVar[tuple[str, ...]] = (
        "tartan-t",
        "tartan",
        "kitti",
        "sintel",
        "spring",
    )
    # Frames before and after the reference frame the network reads.
    FLOW_CONTEXT: ClassVar[Tuple[int, int]] = (1, 1)
    # Per-image native canvases differ, so batches are not stacked.
    SUPPORTS_BATCHED_PREDICT = False
    TTA_ENABLED = False

    # Upstream default refinement iterations (MEMFOF.forward iters=8).
    DEFAULT_ITERS: ClassVar[int] = 8

    # ====================================================================
    # Checkpoint detection
    # ====================================================================

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        # The context net reads three stacked RGB frames (9 input channels) and
        # the GMA attention lives at the top level as ``att.to_qk``; together
        # they separate MEMFOF from two-frame RAFT descendants (SEA-RAFT, RAFT).
        conv1 = weights_dict.get("cnet.resnet.conv1.weight")
        if conv1 is None or getattr(conv1, "ndim", 0) != 4 or int(conv1.shape[1]) != 9:
            return False
        return (
            "att.to_qk.weight" in weights_dict
            and "update_block.aggregator.to_v.weight" in weights_dict
            and "fnet.final_conv.weight" in weights_dict
        )

    @classmethod
    def detect_size(cls, weights_dict: dict) -> Optional[str]:
        return "m" if cls.can_load(weights_dict) else None

    @classmethod
    def detect_nb_classes(cls, weights_dict: dict) -> Optional[int]:
        # The single class slot is a schema placeholder; flow has no classes.
        return 1

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
        iters: int | None = None,
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
        self.iters = int(iters) if iters is not None else self.DEFAULT_ITERS
        if self.iters < 1:
            raise ValueError(f"MEMFOF iters must be at least 1; got {self.iters}.")
        self.model.eval()
        if model_path is not None and isinstance(model_path, (str, Path)):
            self._load_weights(str(model_path))
        self.nb_classes = 1
        self.names = {0: "flow"}

    def _init_model(self) -> nn.Module:
        return MEMFOF()

    def _get_available_layers(self) -> Dict[str, nn.Module]:
        return {
            "cnet": self.model.cnet,
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
    ) -> Tuple[torch.Tensor, Image.Image, Tuple[int, int], Tuple[float, float]]:
        max_side = int(input_size) if input_size is not None else self._get_input_size()
        img = ImageLoader.load(image, color_format=color_format)
        orig_w, orig_h = img.size
        chw, ratio = preprocess_numpy(np.asarray(img), max_side)
        if min(chw.shape[1:]) < MIN_SIDE:
            raise ValueError(
                f"MEMFOF needs frames of at least {MIN_SIDE} px on each side at the "
                f"network canvas; got {chw.shape[2]}x{chw.shape[1]}."
            )
        return torch.from_numpy(chw).unsqueeze(0), img, (orig_w, orig_h), ratio

    @staticmethod
    def _stack_window(frames: Any) -> torch.Tensor:
        """Return a ``(B, 3, 3, H, W)`` window from a tensor or a sequence of three frames."""
        if isinstance(frames, torch.Tensor):
            if frames.ndim == 5:
                window = frames
            elif frames.ndim == 4 and frames.shape[0] == 3:
                window = frames.unsqueeze(0)
            else:
                raise ValueError(
                    "MEMFOF expects a (B, 3, 3, H, W) window or three (3, H, W) "
                    f"frames; got shape {tuple(frames.shape)}."
                )
        else:
            frames = list(frames)
            parts = [f if f.ndim == 4 else f.unsqueeze(0) for f in frames]
            window = torch.stack(parts, dim=1)
        if window.shape[1] != 3 or window.shape[2] != 3:
            raise ValueError(
                "MEMFOF reads exactly three RGB frames (t-1, t, t+1); got a window "
                f"of shape {tuple(window.shape)}."
            )
        return window

    def _forward_window(self, frames: Any) -> torch.Tensor:
        """Run one (t-1, t, t+1) window; return the forward flow ``(B, 2, H, W)`` of frame t."""
        return self._forward(frames)["flow"]

    def _forward(self, input_tensor: Any) -> Dict[str, torch.Tensor]:
        window = self._stack_window(input_tensor)
        param = next(self.model.parameters())
        window = window.to(device=param.device, dtype=param.dtype)
        out = self.model(window, iters=self.iters)
        # out["flow"][-1]: (B, 2 directions, 2, H, W); index 0 is t to t-1,
        # index 1 is t to t+1 (upstream demo: ``[:, 1]`` is forward).
        flow = out["flow"][-1]
        return {"flow": flow[:, 1], "flow_backward": flow[:, 0]}

    def _postprocess(
        self,
        output: Any,
        conf_thres: float,
        iou_thres: float,
        original_size: Tuple[int, int],
        max_det: int = 300,
        **kwargs,
    ) -> Dict:
        flow = output["flow"] if isinstance(output, dict) else output
        if flow.ndim == 4:
            flow = flow[0]
        orig_w, orig_h = original_size
        return {"flow": flow_to_original(flow, orig_h, orig_w).cpu()}

    # ====================================================================
    # Out of scope
    # ====================================================================

    def train(self, *args, **kwargs):
        raise NotImplementedError(
            "Training MEMFOF is not supported by LibreYOLO; the flow task is "
            "inference-only. Use a pretrained LibreMEMFOF checkpoint."
        )
