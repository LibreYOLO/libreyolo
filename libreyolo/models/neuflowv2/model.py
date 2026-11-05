"""LibreYOLO wrapper for NeuFlow v2 optical flow.

NeuFlow v2 (https://github.com/neufieldrobotics/NeuFlow_v2, arXiv 2408.10161)
is a 9M-parameter flow network built for edge devices: a shallow CNN encoder,
two cross-attention layers and a global softmax match at 1/16 scale, then a
light convolutional refiner at 1/16 and 1/8 scale and a learned 8x convex
upsampler. It predicts the dense displacement from frame ``t`` to frame
``t + 1``, which is LibreYOLO's ``flow`` task primitive (``Results.flow``,
ADR 0032).

The vendored model code is Apache-2.0 (``_vendor/``). Upstream ships three
checkpoints in its repository root under the same licence:
``neuflow_mixed.pth`` (trained on several datasets, the canonical
``LibreNeuFlowv2s-flow.pt``), ``neuflow_sintel.pth`` (``-flow-sintel``) and
``neuflow_things.pth`` (``-flow-things``).

Geometry: both frames are stretch-resized to the fixed canvas (432x768 by
default, the upstream ``infer.py`` size, any multiple of 16 via ``imgsz``),
the network runs there with one refinement step at 1/16 and eight at 1/8 (the
upstream inference defaults), and the field is resized back to the original
canvas with each vector component multiplied by its axis resize ratio.

Training is out of scope (see :meth:`LibreNeuFlowv2.train`).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from ...utils.image_loader import ImageInput, ImageLoader
from ..base.model import BaseModel
from .nn import NEUFLOWV2_STRIDE, LibreNeuFlowv2Net
from .utils import preprocess_numpy

logger = logging.getLogger(__name__)


def _input_size_hw(input_size) -> tuple[int, int]:
    if isinstance(input_size, int):
        return input_size, input_size
    if isinstance(input_size, (tuple, list)) and len(input_size) == 2:
        return int(input_size[0]), int(input_size[1])
    raise ValueError(
        f"NeuFlow v2 imgsz must be an int or (height, width), got {input_size!r}"
    )


class LibreNeuFlowv2(BaseModel):
    """NeuFlow v2: a pair of frames -> dense optical flow."""

    FAMILY = "neuflowv2"
    FILENAME_PREFIX = "LibreNeuFlowv2"
    WEIGHT_EXT = ".pt"
    # One released architecture. The canvas is (height, width), the upstream
    # infer.py size; both sides must be multiples of 16.
    INPUT_SIZES: ClassVar[dict[str, tuple[int, int]]] = {"s": (432, 768)}
    IMGSZ_STRIDE: ClassVar[int] = NEUFLOWV2_STRIDE
    SUPPORTED_TASKS = ("flow",)
    DEFAULT_TASK = "flow"
    REQUIRE_TASK_SUFFIX = True
    # Training-data variants of the same network. The bare filename is the
    # upstream "mixed" checkpoint.
    WEIGHT_VARIANTS: ClassVar[tuple[str, ...]] = ("sintel", "things")
    # Frames of context before and after the pair (ADR 0032): a plain pair.
    FLOW_CONTEXT: ClassVar[tuple[int, int]] = (0, 0)
    SUPPORTS_BATCHED_PREDICT = False
    TTA_ENABLED = False

    # Keys unique to NeuFlow v2: the 1/16 cross-attention, the two refiners
    # and the 8x convex upsampler. All four prefixes must be present.
    _SIGNATURE_PREFIXES: ClassVar[tuple[str, ...]] = (
        "backbone.block_8_1.",
        "cross_attn_s16.layers.",
        "refine_s8.conv1.",
        "upsample_s8.conv3.",
    )

    # ====================================================================
    # Checkpoint detection
    # ====================================================================

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        keys = list(weights_dict)
        return all(
            any(k.startswith(prefix) for k in keys)
            for prefix in cls._SIGNATURE_PREFIXES
        )

    @classmethod
    def detect_size(cls, weights_dict: dict) -> str | None:
        return "s" if cls.can_load(weights_dict) else None

    @classmethod
    def detect_nb_classes(cls, weights_dict: dict) -> int | None:
        # The single class slot is a schema placeholder; flow has no classes.
        return 1

    # ====================================================================
    # Construction
    # ====================================================================

    def __init__(
        self,
        model_path=None,
        size: str = "s",
        nb_classes: int = 1,
        device: str = "auto",
        task: str | None = None,
        **kwargs,
    ):
        super().__init__(
            model_path=model_path,
            size=size,
            nb_classes=1,  # always 1 ("flow"); ignore any caller-provided value
            device=device,
            task=task,
            **kwargs,
        )
        self.model.eval()
        if model_path is not None and isinstance(model_path, (str, Path)):
            self._load_weights(str(model_path))
        # Enforce the single flow slot after any loading path (a raw upstream
        # .pth through the autoconverter would otherwise get class_0).
        self.nb_classes = 1
        self.names = {0: "flow"}

    def _init_model(self) -> nn.Module:
        return LibreNeuFlowv2Net()

    def _get_available_layers(self) -> dict[str, nn.Module]:
        return {
            "backbone": self.model.backbone,
            "cross_attn_s16": self.model.cross_attn_s16,
            "refine_s16": self.model.refine_s16,
            "refine_s8": self.model.refine_s8,
            "upsample_s8": self.model.upsample_s8,
        }

    # ====================================================================
    # Inference surface
    # ====================================================================

    @staticmethod
    def _get_preprocess_numpy():
        return preprocess_numpy

    def _canvas_hw(self, input_size=None) -> tuple[int, int]:
        effective = input_size if input_size is not None else self._get_input_size()
        h, w = _input_size_hw(effective)
        if h % NEUFLOWV2_STRIDE or w % NEUFLOWV2_STRIDE:
            raise ValueError(
                f"NeuFlow v2 imgsz={effective} must have both sides divisible "
                f"by {NEUFLOWV2_STRIDE} (encoder stride product)."
            )
        return h, w

    def _preprocess(
        self,
        image: ImageInput,
        color_format: str = "auto",
        input_size=None,
    ) -> tuple[torch.Tensor, Image.Image, tuple[int, int], tuple[float, float]]:
        """Load one frame and stretch it to the canvas.

        Returns ``(tensor (1, 3, H, W) in [0, 1], PIL image, (orig_w, orig_h),
        (sx, sy))``. The flow runner calls it once per frame and stacks the
        tensors into the window passed to :meth:`_forward_flow`.
        """
        input_h, input_w = self._canvas_hw(input_size)
        img = ImageLoader.load(image, color_format=color_format)
        orig_w, orig_h = img.size
        chw, ratio = preprocess_numpy(np.asarray(img), (input_h, input_w))
        return torch.from_numpy(chw).unsqueeze(0), img, (orig_w, orig_h), ratio

    def _forward_pair(self, image1: torch.Tensor, image2: torch.Tensor) -> torch.Tensor:
        """Flow from ``image1`` to ``image2``: ``(B, 2, H, W)`` canvas pixels."""
        return self.model(image1, image2)

    def _forward(self, input_tensor: Any) -> Any:
        """Run a pair given as ``(image1, image2)``, ``(B, 2, 3, H, W)`` or ``(B, 6, H, W)``."""
        if isinstance(input_tensor, (tuple, list)):
            if len(input_tensor) != 2:
                raise ValueError(
                    f"NeuFlow v2 consumes exactly two frames, got {len(input_tensor)}."
                )
            return self._forward_pair(input_tensor[0], input_tensor[1])
        if isinstance(input_tensor, torch.Tensor):
            if input_tensor.ndim == 5 and input_tensor.shape[1] == 2:
                return self._forward_pair(input_tensor[:, 0], input_tensor[:, 1])
            if input_tensor.ndim == 4 and input_tensor.shape[1] == 6:
                return self._forward_pair(input_tensor[:, :3], input_tensor[:, 3:])
        raise ValueError(
            "Optical flow needs a pair of frames; a single image has no motion. "
            "Pass two or more images (a list, a directory or a video)."
        )

    def _forward_flow(self, window: torch.Tensor) -> torch.Tensor:
        """Runner hook (ADR 0032): ``window`` is ``(2, 3, H, W)``, frames t and t+1."""
        return self._forward_pair(window[0:1], window[1:2])

    def _postprocess(
        self,
        output: Any,
        conf_thres: float,
        iou_thres: float,
        original_size: tuple[int, int],
        max_det: int = 300,
        **kwargs,
    ) -> dict:
        flow = output
        if isinstance(flow, dict):
            flow = flow.get("flow", flow.get("predictions"))
        if isinstance(flow, (tuple, list)):
            flow = flow[-1]
        if flow.ndim == 3:
            flow = flow.unsqueeze(0)
        orig_w, orig_h = original_size
        flow = flow[:1].float()
        in_h, in_w = int(flow.shape[-2]), int(flow.shape[-1])
        if (in_h, in_w) != (orig_h, orig_w):
            flow = F.interpolate(
                flow, size=(orig_h, orig_w), mode="bilinear", align_corners=False
            )
            scale = flow.new_tensor([orig_w / in_w, orig_h / in_h]).view(1, 2, 1, 1)
            flow = flow * scale
        return {"flow": flow[0].permute(1, 2, 0).contiguous().cpu()}

    # ====================================================================
    # Out of scope
    # ====================================================================

    def train(self, *args, **kwargs):
        raise NotImplementedError(
            "Training NeuFlow v2 is not supported by LibreYOLO; the flow task "
            "is inference-only. Use LibreNeuFlowv2s-flow.pt (or the -sintel / "
            "-things variants), or convert an upstream checkpoint with "
            "weights/convert_neuflowv2_weights.py."
        )
