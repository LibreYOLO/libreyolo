"""LibreYOLO wrapper for FlowSeek optical flow.

FlowSeek (Poggi and Tosi, ICCV 2025, https://github.com/mattpoggi/flowseek)
is a SEA-RAFT style recurrent flow network that adds a frozen Depth Anything
V2 encoder and a set of depth-derived motion bases. It predicts the dense
displacement from one frame to the next, LibreYOLO's ``flow`` task
(``Results.flow``, a ``FlowField`` on the original canvas of the first frame).

Sizes follow the four upstream evaluation configs. Upstream trains two
networks; ``s`` and ``l`` reuse the ``t`` and ``m`` weights with more refinement
iterations, exactly as upstream ``config/eval/flowseek-[T,S,M,L].json`` does:

====  ===========  ===============  ==========  =============================
size  context net  DAv2 encoder     iterations  checkpoint
====  ===========  ===============  ==========  =============================
t     ResNet-18    ViT-S (Small)    4           flowseek_T_TartanCT_TSKH.pth
s     ResNet-18    ViT-S (Small)    12          flowseek_T_TartanCT_TSKH.pth
m     ResNet-34    ViT-B (Base)     4           flowseek_M_TartanCT_TSKH.pth
l     ResNet-34    ViT-B (Base)     12          flowseek_M_TartanCT_TSKH.pth
====  ===========  ===============  ==========  =============================

The hosted checkpoints are the final TartanAir, Chairs, Things, then
Sintel+KITTI+HD1K+Things mixed fine-tune (``TartanCT_TSKH``), the generalist
upstream evaluates on Spring and LayeredFlow.

Licences: the code is Apache-2.0 (FlowSeek) with BSD-3-Clause parts from
SEA-RAFT. Each checkpoint embeds its Depth Anything V2 encoder (``dav2.*``):
the ViT-S encoder of ``t`` and ``s`` is Apache-2.0, the ViT-B encoder of ``m``
and ``l`` is CC BY-NC 4.0, so the ``m`` and ``l`` checkpoints are for
non-commercial use only.

Inference runs at the native frame resolution, padded to a multiple of 8
inside the network (upstream ``InputPadder``), so the field needs no vector
rescaling. Training is out of scope.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from ...utils.image_loader import ImageInput, ImageLoader
from ..base.flow_sequence import NATIVE_FLOW_IMGSZ, downscale_rgb
from ..base.model import BaseModel
from ._vendor.flowseek import FlowSeek

# Upstream config/eval/flowseek-[T,S,M,L].json.
FLOWSEEK_CONFIGS: Dict[str, Dict[str, Any]] = {
    "t": {"pretrain": "resnet18", "da_size": "vits", "iters": 4},
    "s": {"pretrain": "resnet18", "da_size": "vits", "iters": 12},
    "m": {"pretrain": "resnet34", "da_size": "vitb", "iters": 4},
    "l": {"pretrain": "resnet34", "da_size": "vitb", "iters": 12},
}
_SHARED_CONFIG = {
    "initial_dim": 64,
    "block_dims": [64, 128, 256],
    "radius": 4,
    "dim": 128,
    "num_blocks": 2,
}

_NON_COMMERCIAL_SIZES = ("m", "l")


def build_flowseek(size: str) -> FlowSeek:
    cfg = dict(_SHARED_CONFIG)
    cfg["block_dims"] = list(cfg["block_dims"])
    cfg.update(FLOWSEEK_CONFIGS[size])
    return FlowSeek(SimpleNamespace(**cfg))


def preprocess_numpy(img_rgb_hwc: np.ndarray, input_size: int | None = None):
    """RGB uint8 ``(H, W, 3)`` to float32 ``(3, H, W)`` in ``[0, 255]``.

    FlowSeek normalises inside the network and pads to a multiple of 8 itself,
    so the frame keeps its native size. The ratio is always ``1.0``.
    """
    chw = np.ascontiguousarray(np.asarray(img_rgb_hwc, dtype=np.float32).transpose(2, 0, 1))
    return chw, 1.0


# FlowSeek builds on SEA-RAFT: features at 1/8 resolution with a four-level
# correlation pyramid on top, so each side of the padded working canvas must be
# at least 128 px. Probed with the tiny model: 96x96 and 200x48 fail inside the
# network with a zero-sized upsample, 128x128 runs (RAFT and SEA-RAFT are the same).
MIN_SIDE = 128


class LibreFlowSeek(BaseModel):
    """FlowSeek: frame pair to dense optical flow."""

    FAMILY = "flowseek"
    FILENAME_PREFIX = "LibreFlowSeek"
    WEIGHT_EXT = ".pt"
    # ``imgsz`` is a cap on the long side; the default means native resolution
    # (the frame runs at its own size, padded to a multiple of 8 by the network).
    INPUT_SIZES: ClassVar[Dict[str, int]] = {
        size: NATIVE_FLOW_IMGSZ for size in ("t", "s", "m", "l")
    }
    SUPPORTED_TASKS = ("flow",)
    DEFAULT_TASK = "flow"
    REQUIRE_TASK_SUFFIX = True
    # Frames of context before and after the pair: FlowSeek uses the pair only.
    FLOW_CONTEXT: ClassVar[Tuple[int, int]] = (0, 0)
    SUPPORTS_BATCHED_PREDICT = False

    ITERATIONS: ClassVar[Dict[str, int]] = {k: v["iters"] for k, v in FLOWSEEK_CONFIGS.items()}
    _EMBED_DIM_TO_SIZE: ClassVar[Dict[int, str]] = {384: "t", 768: "m"}

    # ====================================================================
    # Checkpoint detection
    # ====================================================================

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        # The motion-basis encoder (bnet), the DAv2 feature merge head and the
        # embedded DAv2 encoder together are unique to FlowSeek.
        keys = weights_dict.keys()
        return (
            "bnet.conv1.weight" in keys
            and "merge_head.0.weight" in keys
            and "dav2.pretrained.cls_token" in keys
            and "update_block.encoder.convc1.weight" in keys
        )

    @classmethod
    def detect_size(cls, weights_dict: dict) -> Optional[str]:
        # t/s (and m/l) share weights; the state dict alone gives the smaller
        # iteration count. The filename or checkpoint metadata picks s or l.
        cls_token = weights_dict.get("dav2.pretrained.cls_token")
        if cls_token is not None and getattr(cls_token, "ndim", 0) >= 1:
            return cls._EMBED_DIM_TO_SIZE.get(int(cls_token.shape[-1]))
        return None

    @classmethod
    def detect_nb_classes(cls, weights_dict: dict) -> Optional[int]:
        return 1

    @classmethod
    def get_download_notice(cls, filename: str, url: str) -> Optional[str]:
        size = cls.detect_size_from_filename(filename)
        if size in _NON_COMMERCIAL_SIZES:
            return (
                f"{cls.FILENAME_PREFIX}{size}-flow embeds the Depth Anything V2 "
                "Base encoder, licensed CC BY-NC 4.0: non-commercial use only."
            )
        return None

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
        if isinstance(model_path, (str, Path)):
            self._load_weights(str(model_path))
        self.nb_classes = 1
        self.names = {0: "flow"}

    def _init_model(self) -> nn.Module:
        return build_flowseek(self.size)

    def _get_available_layers(self) -> Dict[str, nn.Module]:
        return {
            "dav2": self.model.dav2,
            "cnet": self.model.cnet,
            "bnet": self.model.bnet,
            "fnet": self.model.fnet,
            "update_block": self.model.update_block,
        }

    @property
    def iterations(self) -> int:
        return self.ITERATIONS[self.size]

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
        # Native resolution by default; an explicit ``imgsz`` caps the long side
        # and ``_postprocess`` rescales the flow back to the original canvas.
        max_side = input_size if input_size is not None else self._get_input_size()
        img = ImageLoader.load(image, color_format=color_format)
        orig_w, orig_h = img.size
        frame, ratio = downscale_rgb(np.asarray(img.convert("RGB")), max_side)
        work_h, work_w = frame.shape[:2]
        # The network pads to a multiple of 8 before the 1/8 feature pyramid, so
        # the padded canvas is what must reach the minimum.
        padded_h, padded_w = ((work_h + 7) // 8) * 8, ((work_w + 7) // 8) * 8
        if min(padded_h, padded_w) < MIN_SIDE:
            cause = (
                f" after capping the long side at imgsz={max_side}"
                if (work_h, work_w) != (orig_h, orig_w)
                else ""
            )
            raise ValueError(
                f"FlowSeek needs frames of at least {MIN_SIDE}x{MIN_SIDE} pixels, got "
                f"{work_w}x{work_h}{cause}: its correlation pyramid and the "
                "upsampling head fail inside the network on smaller frames."
            )
        chw, _ = preprocess_numpy(frame)
        return torch.from_numpy(chw).unsqueeze(0), img, (orig_w, orig_h), ratio

    def _forward_pair(self, image1: torch.Tensor, image2: torch.Tensor) -> torch.Tensor:
        """Flow from ``image1`` to ``image2``.

        Both are ``(B, 3, H, W)`` RGB floats in ``[0, 255]``; returns the
        ``(B, 2, H, W)`` flow in pixels at the input size, with the size's
        iteration count.
        """
        if image1.shape != image2.shape:
            raise ValueError(
                "FlowSeek needs two frames of the same size; got "
                f"{tuple(image1.shape)} and {tuple(image2.shape)}."
            )
        return self.model(image1, image2, iters=self.iterations)

    def _forward_flow(self, window: torch.Tensor) -> torch.Tensor:
        """Runner hook: ``window`` is ``(2, 3, H, W)`` (FLOW_CONTEXT is (0, 0))."""
        before = self.FLOW_CONTEXT[0]
        return self._forward_pair(window[before : before + 1], window[before + 1 : before + 2])

    def _forward(self, input_tensor: torch.Tensor) -> Any:
        raise ValueError(
            "LibreFlowSeek predicts optical flow from a pair of frames; pass two "
            "or more images (a list, a directory or a video), not a single image."
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
        flow = output["flow"] if isinstance(output, dict) else output
        flow = flow.float()
        orig_w, orig_h = original_size
        in_h, in_w = flow.shape[-2:]
        if (in_h, in_w) != (orig_h, orig_w):
            flow = F.interpolate(flow, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
            scale = torch.tensor([orig_w / in_w, orig_h / in_h], device=flow.device, dtype=flow.dtype)
            flow = flow * scale.view(1, 2, 1, 1)
        return {"flow": flow[0].permute(1, 2, 0).contiguous().cpu()}

    # ====================================================================
    # Out of scope
    # ====================================================================

    def train(self, *args, **kwargs):
        raise NotImplementedError(
            "Training FlowSeek is not supported by LibreYOLO; the flow task is "
            "inference-only. Use a pretrained LibreFlowSeek checkpoint."
        )
