"""LibreYOLO wrapper for SEA-RAFT optical flow.

SEA-RAFT (https://github.com/princeton-vl/SEA-RAFT, Wang, Lipson and Deng,
ECCV 2024) is a RAFT descendant: a ResNet context and feature encoder, an
all-pairs correlation pyramid and a ConvNeXt update block that refines a
directly regressed initial flow, trained with a mixture-of-Laplace loss. It
predicts LibreYOLO's ``flow`` primitive (``Results.flow``): a dense ``(H, W, 2)``
displacement field from frame ``t`` to frame ``t + 1``.

Sizes follow upstream: ``s`` (ResNet18 encoders, 4 iterations), ``m``
(ResNet34, 4 iterations) and ``l`` (the ``m`` checkpoint with 12 iterations;
upstream ships no separate L weights). The canonical checkpoints are the
Tartan-C-T-TSKH-spring540x960 ones. The predicted mixture variance becomes the
optional ``FlowField.confidence`` map (see :func:`.nn.info_to_confidence`).

Code and weights are BSD-3-Clause (``_vendor/LICENSE``). Inference only.
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
from .nn import SEARAFT_CONFIGS, LibreSEARAFTNet, info_to_confidence
from .utils import compute_padding, flow_to_original, map_to_original, preprocess_numpy, resized_hw

logger = logging.getLogger(__name__)


class LibreSEARAFT(BaseModel):
    """SEA-RAFT: a frame pair -> dense optical flow field."""

    FAMILY = "searaft"
    FILENAME_PREFIX = "LibreSEARAFT"
    WEIGHT_EXT = ".pt"
    # Long-side cap, not a fixed canvas: frames up to this size run at native
    # resolution (padded to a multiple of 8), larger ones are downscaled.
    # 960 is the training crop width of the canonical spring540x960 weights.
    INPUT_SIZES: ClassVar[Dict[str, int]] = {"s": 960, "m": 960, "l": 960}
    SUPPORTED_TASKS = ("flow",)
    DEFAULT_TASK = "flow"
    REQUIRE_TASK_SUFFIX = True
    # Two frames per prediction, no extra context frames.
    FLOW_CONTEXT: ClassVar[Tuple[int, int]] = (0, 0)
    # Native-resolution preprocessing gives variable per-image sizes.
    SUPPORTS_BATCHED_PREDICT = False

    flow_imgsz_divisor = 8

    # ====================================================================
    # Checkpoint detection
    # ====================================================================

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        # SEA-RAFT's context encoder reads the stacked pair (6 input channels)
        # and its flow head emits flow + mixture weights + log scales (6
        # channels). RAFT has neither ``init_conv`` nor a 6-channel ``cnet``.
        cnet_conv = weights_dict.get("cnet.conv1.weight")
        head = weights_dict.get("flow_head.2.weight")
        return (
            cnet_conv is not None
            and getattr(cnet_conv, "ndim", 0) == 4
            and int(cnet_conv.shape[1]) == 6
            and head is not None
            and int(head.shape[0]) == 6
            and "init_conv.weight" in weights_dict
            and "upsample_weight.2.weight" in weights_dict
        )

    @staticmethod
    def restore_shared_keys(weights_dict: dict) -> dict:
        """Re-add ``downsample.1.*`` aliases that safetensors drops.

        In ``BasicBlock`` the projection's BatchNorm is registered twice
        (``bn3`` and ``downsample.1``). ``torch.save`` keeps both names, but the
        upstream Hugging Face ``model.safetensors`` files store each shared
        tensor once, under ``bn3``.
        """
        out = dict(weights_dict)
        for key, value in weights_dict.items():
            if ".bn3." in key:
                alias = key.replace(".bn3.", ".downsample.1.")
                out.setdefault(alias, value)
        return out

    @classmethod
    def convert_upstream_state_dict(cls, state_dict: dict) -> Optional[dict]:
        if not cls.can_load(state_dict):
            return None
        return cls.restore_shared_keys(state_dict)

    @classmethod
    def detect_size(cls, weights_dict: dict) -> Optional[str]:
        # ResNet18 has 2 blocks in layer1, ResNet34 has 3. ``l`` shares the
        # ``m`` weights, so tensors alone say ``m``; checkpoint metadata
        # carries ``l`` for the 12-iteration variant.
        blocks = {
            int(k.split(".")[2])
            for k in weights_dict
            if k.startswith("cnet.layer1.") and k.endswith(".conv1.weight")
        }
        if not blocks:
            return None
        return {2: "s", 3: "m"}.get(max(blocks) + 1)

    @classmethod
    def detect_nb_classes(cls, weights_dict: dict) -> Optional[int]:
        return 1  # schema placeholder, flow has no classes

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
        return LibreSEARAFTNet(self.size)

    def _get_available_layers(self) -> Dict[str, nn.Module]:
        return {
            "cnet": self.model.cnet,
            "fnet": self.model.fnet,
            "update_block": self.model.update_block,
            "flow_head": self.model.flow_head,
        }

    @property
    def iters(self) -> int:
        return int(SEARAFT_CONFIGS[self.size]["iters"])

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
    ) -> Tuple[torch.Tensor, Image.Image, Tuple[int, int], dict]:
        """One frame -> padded ``[0, 1]`` tensor ``(1, 3, Hp, Wp)``.

        Returns ``(tensor, pil_image, (orig_w, orig_h), geometry)``; the runner
        hands ``geometry`` back to :meth:`_postprocess` as ``ratio``.
        """
        img = ImageLoader.load(image, color_format=color_format)
        orig_w, orig_h = img.size
        tensor, meta = preprocess_numpy(np.asarray(img), input_size)
        return tensor, img, (orig_w, orig_h), meta

    def _forward_pair(
        self, image1: torch.Tensor, image2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Padded ``[0, 1]`` frames -> ``(flow, info)`` at the padded canvas."""
        if image1.shape != image2.shape:
            raise ValueError(
                f"SEA-RAFT needs two frames of the same size, got "
                f"{tuple(image1.shape)} and {tuple(image2.shape)}"
            )
        return self.model(image1, image2)

    def _forward_flow(self, window: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Runner hook: ``window`` is ``(2, 3, Hp, Wp)``, frame t then frame t+1."""
        return self._forward_pair(window[0:1], window[1:2])

    def _forward(self, input_tensor: Any) -> Any:
        """Accept a ``(image1, image2)`` pair or a ``(B, 6, H, W)`` stack."""
        if isinstance(input_tensor, (tuple, list)) and len(input_tensor) == 2:
            return self._forward_pair(input_tensor[0], input_tensor[1])
        if isinstance(input_tensor, torch.Tensor) and input_tensor.ndim == 4 and input_tensor.shape[1] == 6:
            return self._forward_pair(input_tensor[:, :3], input_tensor[:, 3:])
        raise ValueError(
            "SEA-RAFT predicts flow from a frame pair; pass two frames "
            "(a list, a directory, or a video), not a single image."
        )

    @staticmethod
    def _geometry(original_size: Tuple[int, int], input_size: Optional[int]) -> dict:
        orig_w, orig_h = original_size
        work_h, work_w = resized_hw(orig_h, orig_w, input_size)
        return {
            "orig_hw": (orig_h, orig_w),
            "work_hw": (work_h, work_w),
            "pad": compute_padding(work_h, work_w),
        }

    def _postprocess(
        self,
        output: Any,
        conf_thres: float,
        iou_thres: float,
        original_size: Tuple[int, int],
        max_det: int = 300,
        input_size: Optional[int] = None,
        ratio: Optional[dict] = None,
        **kwargs,
    ) -> Dict:
        """``(flow, info)`` at the padded canvas -> flow on the original canvas.

        Returns ``{"flow": (H, W, 2) float32, "flow_confidence": (H, W)}`` with
        vectors in original-canvas pixels.
        """
        flow, info = output
        # ``ratio`` is the geometry dict our ``_preprocess`` returned for frame t.
        geom = ratio if isinstance(ratio, dict) else self._geometry(original_size, input_size)
        flow = flow_to_original(flow.float(), geom["pad"], geom["orig_hw"])
        conf = info_to_confidence(
            map_to_original(info.float(), geom["pad"], geom["orig_hw"])
        )
        return {
            "flow": flow[0].permute(1, 2, 0).contiguous().cpu(),
            "flow_confidence": conf[0].cpu(),
        }

    # ====================================================================
    # Out of scope
    # ====================================================================

    def train(self, *args, **kwargs):
        raise NotImplementedError(
            "Training SEA-RAFT is not supported by LibreYOLO; the flow task is "
            "inference-only. Use a pretrained LibreSEARAFT checkpoint or convert "
            "an upstream one with weights/convert_searaft_weights.py."
        )
