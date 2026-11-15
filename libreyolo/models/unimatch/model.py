"""LibreYOLO wrapper for UniMatch / GMFlow optical flow.

UniMatch (https://github.com/autonomousvision/unimatch, TPAMI 2023) unifies
flow, stereo and depth with one transformer matcher; GMFlow is its flow model.
This family ships the optical flow task only:

- ``s``: GMFlow scale1 (global matching at 1/8, convex upsampling), the
  ``gmflow-scale1-things`` checkpoint (FlyingChairs then FlyingThings3D).
- ``l``: GMFlow+ scale2 with 6 local regression refinement steps, the
  ``gmflow-scale2-regrefine6-mixdata`` checkpoint upstream recommends for
  real-world data. Variants ``sintel`` and ``kitti`` load the Sintel and
  KITTI-2015 finetunes of the same architecture.

The vendored model code (``_vendor/``) is MIT and copied verbatim, including
the stereo and depth paths, so those tasks can be wired later. Upstream
publishes the checkpoints without a separate weight licence; LibreYOLO hosts
them under the releasing project's MIT licence, an implied grant, not a
publisher-confirmed one.

Preprocessing reproduces upstream ``inference_flow``: portrait frames are
transposed to landscape, both frames are bilinearly resized
(``align_corners=True``) to each side rounded up to the size's padding factor
(16 for ``s``, 32 for ``l``), and the network normalizes ``[0, 255]`` input
with ImageNet statistics. Postprocess resizes the field back and multiplies
``dx`` by ``W_orig / W_in`` and ``dy`` by ``H_orig / H_in``. One deliberate
deviation: for portrait frames upstream transposes the field back without
swapping its two channels, which leaves ``dx`` and ``dy`` exchanged; this
wrapper swaps them so the field follows the flow contract.

Training is out of scope (see :meth:`LibreUniMatch.train`).
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
from ._vendor.unimatch.unimatch import UniMatch
from .utils import inference_size, preprocess_numpy

logger = logging.getLogger(__name__)


class LibreUniMatch(BaseModel):
    """UniMatch / GMFlow: a frame pair to a dense optical flow field."""

    FAMILY = "unimatch"
    FILENAME_PREFIX = "LibreUniMatch"
    WEIGHT_EXT = ".pt"
    # Nominal canvas for fixed-size consumers (export, validation). Predict
    # keeps upstream's native geometry: each side rounded up to the padding
    # factor of the size.
    INPUT_SIZES: ClassVar[Dict[str, int]] = {"s": 640, "l": 640}
    SUPPORTED_TASKS = ("flow",)
    DEFAULT_TASK = "flow"
    REQUIRE_TASK_SUFFIX = True
    # Sintel and KITTI-2015 finetunes of the ``l`` architecture.
    WEIGHT_VARIANTS: ClassVar[tuple[str, ...]] = ("sintel", "kitti")
    UNPUBLISHED_WEIGHTS: ClassVar[Dict[str, str]] = {
        "LibreUniMatchs-flow-sintel": (
            "Only the l size has Sintel and KITTI finetunes; use "
            "LibreUniMatchl-flow-sintel.pt."
        ),
        "LibreUniMatchs-flow-kitti": (
            "Only the l size has Sintel and KITTI finetunes; use "
            "LibreUniMatchl-flow-kitti.pt."
        ),
    }
    # Two-frame model: no frames of context before or after the pair.
    FLOW_CONTEXT: ClassVar[Tuple[int, int]] = (0, 0)
    # Native geometry varies per input, so frames are not stacked.
    SUPPORTS_BATCHED_PREDICT = False

    # Architecture and the upstream inference arguments, per size, taken from
    # upstream scripts/gmflow_demo.sh and main_flow.py defaults.
    _CONFIGS: ClassVar[Dict[str, Dict[str, Any]]] = {
        "s": {
            "num_scales": 1,
            "upsample_factor": 8,
            "reg_refine": False,
            "padding_factor": 16,
            "attn_type": "swin",
            "attn_splits_list": [2],
            "corr_radius_list": [-1],
            "prop_radius_list": [-1],
            "num_reg_refine": 1,
        },
        "l": {
            "num_scales": 2,
            "upsample_factor": 4,
            "reg_refine": True,
            "padding_factor": 32,
            "attn_type": "swin",
            "attn_splits_list": [2, 8],
            "corr_radius_list": [-1, 4],
            "prop_radius_list": [-1, 1],
            "num_reg_refine": 6,
        },
    }

    # ====================================================================
    # Checkpoint detection
    # ====================================================================

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        # GMFlow's self-attention flow propagation and its 6-layer
        # cross-attention transformer together with the ResNet-style CNN
        # encoder. ``feature_flow_attn`` is unique to the GMFlow lineage.
        keys = weights_dict.keys()
        return (
            "feature_flow_attn.q_proj.weight" in keys
            and "backbone.conv1.weight" in keys
            and any(k.startswith("transformer.layers.") for k in keys)
        )

    @classmethod
    def detect_size(cls, weights_dict: dict) -> Optional[str]:
        keys = weights_dict.keys()
        has_trident = "backbone.trident_conv.weight" in keys
        if "refine_proj.weight" in keys and has_trident:
            return "l"
        if "upsampler.0.weight" in keys and not has_trident:
            # Only the 1/8 convex upsampler (8 * 8 * 9 mask channels) is the
            # scale1 graph shipped as size s.
            up = weights_dict.get("upsampler.2.weight")
            if up is None or int(up.shape[0]) == 8 * 8 * 9:
                return "s"
        # scale2 without refinement is not a shipped size.
        return None

    @classmethod
    def detect_nb_classes(cls, weights_dict: dict) -> Optional[int]:
        # Schema placeholder; flow has no classes.
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
        cfg = self._CONFIGS[self.size]
        return UniMatch(
            num_scales=cfg["num_scales"],
            feature_channels=128,
            upsample_factor=cfg["upsample_factor"],
            num_head=1,
            ffn_dim_expansion=4,
            num_transformer_layers=6,
            reg_refine=cfg["reg_refine"],
            task="flow",
        )

    def _get_available_layers(self) -> Dict[str, nn.Module]:
        layers = {
            "backbone": self.model.backbone,
            "transformer": self.model.transformer,
            "feature_flow_attn": self.model.feature_flow_attn,
        }
        if hasattr(self.model, "refine"):
            layers["refine"] = self.model.refine
        if hasattr(self.model, "upsampler"):
            layers["upsampler"] = self.model.upsampler
        return layers

    @property
    def inference_args(self) -> Dict[str, Any]:
        """Upstream forward arguments baked in for this size."""
        cfg = self._CONFIGS[self.size]
        return {
            "attn_type": cfg["attn_type"],
            "attn_splits_list": list(cfg["attn_splits_list"]),
            "corr_radius_list": list(cfg["corr_radius_list"]),
            "prop_radius_list": list(cfg["prop_radius_list"]),
            "num_reg_refine": cfg["num_reg_refine"],
            "task": "flow",
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
        """Load one frame and bring it to upstream's inference geometry.

        Returns a ``(1, 3, H_in, W_in)`` float tensor in ``[0, 255]``,
        transposed when the frame is portrait (upstream trains with
        width >= height). Both frames of a pair go through this independently;
        the geometry depends only on the frame size, so equal-size frames
        land on the same canvas.
        """
        img = ImageLoader.load(image, color_format=color_format)
        orig_w, orig_h = img.size
        return self.preprocess_array(np.asarray(img.convert("RGB"))), img, (orig_w, orig_h), 1.0

    def preprocess_array(self, rgb: np.ndarray) -> torch.Tensor:
        """Upstream geometry for an ``(H, W, 3)`` uint8 RGB array."""
        tensor = torch.from_numpy(np.array(rgb[..., :3], dtype=np.uint8)).permute(2, 0, 1)
        tensor = tensor.float().unsqueeze(0)
        if tensor.shape[-2] > tensor.shape[-1]:
            tensor = tensor.transpose(-2, -1)
        h, w = tensor.shape[-2:]
        size = inference_size(h, w, self._CONFIGS[self.size]["padding_factor"])
        if (h, w) != size:
            tensor = F.interpolate(tensor, size=size, mode="bilinear", align_corners=True)
        return tensor

    def _forward_pair(self, image1: torch.Tensor, image2: torch.Tensor) -> torch.Tensor:
        """Flow from ``image1`` to ``image2`` on the inference canvas.

        Inputs are ``(B, 3, H_in, W_in)`` outputs of :meth:`_preprocess`.
        Returns ``(B, 2, H_in, W_in)`` flow in inference-canvas pixels.
        """
        if image1.shape != image2.shape:
            raise ValueError(
                "UniMatch needs both frames of a pair at the same size, got "
                f"{tuple(image1.shape)} and {tuple(image2.shape)}."
            )
        device = next(self.model.parameters()).device
        out = self.model(
            image1.to(device=device, dtype=torch.float32),
            image2.to(device=device, dtype=torch.float32),
            **self.inference_args,
        )
        return out["flow_preds"][-1]

    def _forward_flow(self, window: torch.Tensor) -> torch.Tensor:
        """Runner hook: ``window`` is ``(2, 3, H_in, W_in)``, frames t and t+1."""
        return self._forward_pair(window[0:1], window[1:2])

    def _forward(self, input_tensor: Any) -> Any:
        # A pair arrives either as a (image1, image2) sequence or as a
        # (B, 2, 3, H, W) stacked tensor.
        if isinstance(input_tensor, (tuple, list)):
            image1, image2 = input_tensor
        elif isinstance(input_tensor, torch.Tensor) and input_tensor.ndim == 5:
            image1, image2 = input_tensor[:, 0], input_tensor[:, 1]
        else:
            raise ValueError(
                "UniMatch predicts optical flow from a pair of frames; pass two "
                "or more images (a list, a directory or a video), not one image."
            )
        return self._forward_pair(image1, image2)

    def flow_to_original(
        self, flow: torch.Tensor, original_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Map ``(B, 2, H_in, W_in)`` canvas flow to ``(B, 2, H, W)`` original pixels."""
        orig_w, orig_h = original_size
        transposed = orig_h > orig_w
        # Work in the (possibly transposed) frame upstream predicted in.
        ori = (orig_w, orig_h) if transposed else (orig_h, orig_w)
        flow = flow.float()
        in_h, in_w = flow.shape[-2:]
        if (in_h, in_w) != ori:
            flow = F.interpolate(flow, size=ori, mode="bilinear", align_corners=True)
            flow = torch.stack(
                (flow[:, 0] * ori[1] / in_w, flow[:, 1] * ori[0] / in_h), dim=1
            )
        if transposed:
            # Transposing the canvas exchanges the axes, so dx and dy swap too.
            flow = flow.transpose(-2, -1).flip(1)
        return flow

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
            flow = flow["flow_preds"][-1] if "flow_preds" in flow else flow["flow"]
        flow = self.flow_to_original(flow, original_size)
        return {"flow": flow[0].permute(1, 2, 0).contiguous().cpu()}

    # ====================================================================
    # Out of scope
    # ====================================================================

    def train(self, *args, **kwargs):
        raise NotImplementedError(
            "Training and fine-tuning UniMatch are not supported by LibreYOLO. "
            "Use a pretrained LibreYOLO checkpoint, or convert an upstream "
            "GMFlow checkpoint with weights/convert_unimatch_weights.py."
        )
