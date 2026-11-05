"""LibreYOLO wrapper for RAFT optical flow.

RAFT (Teed and Deng, "RAFT: Recurrent All-Pairs Field Transforms for Optical
Flow", ECCV 2020, https://github.com/princeton-vl/RAFT) estimates a dense
flow field between two frames. The vendored upstream code is BSD-3-Clause
(``_vendor/``); the released checkpoints ship under the same repository.

Sizes: ``s`` is raft-small, ``l`` is raft (canonical weights: raft-things).
Weight variants of ``l``: ``-sintel`` (raft-sintel) and ``-kitti``
(raft-kitti). Inference runs at native resolution with upstream InputPadder
padding to a multiple of 8 and a fixed 20 refinement iterations (upstream
``demo.py``). Training is out of scope (see :meth:`LibreRAFT.train`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Dict, Optional, Tuple

import torch
import torch.nn as nn

from ...utils.image_loader import ImageInput
from ..base.flow_sequence import NATIVE_FLOW_IMGSZ
from ..base.model import BaseModel
from .nn import DEFAULT_ITERS, LibreRAFTNet
from .utils import postprocess_flow, preprocess_image, preprocess_numpy

_KEY_PREFIXES = ("fnet.", "cnet.", "update_block.")


class LibreRAFT(BaseModel):
    """RAFT: a pair of RGB frames in, a dense ``(H, W, 2)`` flow field out."""

    FAMILY = "raft"
    FILENAME_PREFIX = "LibreRAFT"
    WEIGHT_EXT = ".pt"
    # Nominal square size recorded in checkpoints and used by fixed-size
    # export. Predict runs at native resolution (padded to a multiple of 8)
    # unless an explicit ``imgsz`` caps the long side.
    # ``imgsz`` is a cap on the long side; the default means native resolution.
    INPUT_SIZES: ClassVar[Dict[str, int]] = {"s": NATIVE_FLOW_IMGSZ, "l": NATIVE_FLOW_IMGSZ}
    SUPPORTED_TASKS = ("flow",)
    DEFAULT_TASK = "flow"
    REQUIRE_TASK_SUFFIX = True
    WEIGHT_VARIANTS: ClassVar[Tuple[str, ...]] = ("sintel", "kitti")
    # Frames of context before and after the pair (two-frame model).
    FLOW_CONTEXT: ClassVar[Tuple[int, int]] = (0, 0)
    FLOW_ITERS: ClassVar[int] = DEFAULT_ITERS
    # Native-resolution inputs vary per image, so no stacked batched predict.
    SUPPORTS_BATCHED_PREDICT = False
    TTA_ENABLED = False

    # ====================================================================
    # Checkpoint detection
    # ====================================================================

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        # Every RAFT tensor lives under fnet / cnet / update_block, and the
        # update block carries RAFT's motion encoder plus a ConvGRU (small) or
        # SepConvGRU (large). SEA-RAFT and other RAFT descendants add top-level
        # modules, so the closed prefix set keeps them out.
        keys = list(weights_dict)
        if not keys or not all(k.startswith(_KEY_PREFIXES) for k in keys):
            return False
        required = (
            "fnet.conv1.weight",
            "cnet.conv1.weight",
            "update_block.encoder.convc1.weight",
            "update_block.flow_head.conv1.weight",
        )
        return all(k in weights_dict for k in required) and cls.detect_size(weights_dict) is not None

    @classmethod
    def detect_size(cls, weights_dict: dict) -> Optional[str]:
        if "update_block.gru.convz1.weight" in weights_dict and "update_block.mask.0.weight" in weights_dict:
            return "l"
        if "update_block.gru.convz.weight" in weights_dict:
            return "s"
        return None

    @classmethod
    def detect_nb_classes(cls, weights_dict: dict) -> Optional[int]:
        return 1 if cls.can_load(weights_dict) else None

    @classmethod
    def detect_checkpoint_task(cls, state_dict: dict) -> Optional[str]:
        return "flow" if cls.can_load(state_dict) else None

    # ====================================================================
    # Construction
    # ====================================================================

    def __init__(
        self,
        model_path=None,
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
        self.names = {0: "flow"}

    def _init_model(self) -> nn.Module:
        return LibreRAFTNet(small=self.size == "s", iters=self.FLOW_ITERS)

    def _get_available_layers(self) -> Dict[str, nn.Module]:
        return {
            "fnet": self.model.fnet,
            "cnet": self.model.cnet,
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
    ) -> Tuple[torch.Tensor, Any, Tuple[int, int], float]:
        # Native resolution by default; an explicit ``imgsz`` caps the long side.
        max_side = input_size if input_size is not None else self._get_input_size()
        return preprocess_image(image, color_format=color_format, max_side=max_side)

    def _forward_pair(self, image1: torch.Tensor, image2: torch.Tensor) -> torch.Tensor:
        """Flow from ``image1`` to ``image2``; both ``(B, 3, H, W)`` from ``_preprocess``."""
        return self.model(image1, image2)

    def _forward_flow(self, window: torch.Tensor) -> torch.Tensor:
        """Runner hook: ``window`` is ``(2, 3, H, W)`` (frame t, frame t+1)."""
        return self._forward_pair(window[0:1], window[1:2])

    def _forward(self, input_tensor: Any) -> Any:
        """Accept a frame pair as a 2-sequence, ``(B, 2, 3, H, W)`` or ``(B, 6, H, W)``."""
        if isinstance(input_tensor, (list, tuple)):
            if len(input_tensor) != 2:
                raise ValueError(f"RAFT expects exactly 2 frames, got {len(input_tensor)}.")
            return self._forward_pair(input_tensor[0], input_tensor[1])
        if input_tensor.ndim == 5 and input_tensor.shape[1] == 2:
            return self._forward_pair(input_tensor[:, 0], input_tensor[:, 1])
        if input_tensor.ndim == 4 and input_tensor.shape[1] == 6:
            return self._forward_pair(input_tensor[:, :3], input_tensor[:, 3:])
        raise ValueError(
            "RAFT computes flow between two frames; pass a pair of images "
            f"(got a tensor of shape {tuple(input_tensor.shape)})."
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
        del conf_thres, iou_thres, max_det, kwargs
        flow = output.get("flow") if isinstance(output, dict) else output
        work_hw = ratio.get("work_hw") if isinstance(ratio, dict) else None
        return {"flow": postprocess_flow(flow, original_size, work_hw)}

    # ====================================================================
    # Out of scope
    # ====================================================================

    def train(self, *args, **kwargs):
        raise NotImplementedError(
            "Training RAFT is not supported by LibreYOLO; the flow task is "
            "inference-only. Use a pretrained LibreRAFT checkpoint or convert "
            "an upstream one with weights/convert_raft_weights.py."
        )
