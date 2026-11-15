"""LibreYOLO wrapper for WAFT optical flow.

WAFT, Warping-Alone Field Transforms (Wang and Deng, ICLR 2026,
https://github.com/princeton-vl/WAFT), replaces RAFT's cost volume with
high-resolution feature warping and a ViT refinement loop. Upstream code and
the released checkpoints are BSD-3-Clause (``_vendor/LICENSE``).

Sizes map the released zero-shot checkpoints (TartanAir, then Chairs, then
Things) by parameter count; the backbone is the weight variant:

====  ==============================  =========  ==============================
size  upstream checkpoint             params     LibreYOLO file
====  ==============================  =========  ==============================
t     a2, DINOv3 ViT-S/16 encoder     56.5 M     ``LibreWAFTt-flow-dinov3.pt``
s     a2, Depth Anything V2 ViT-S     56.9 M     ``LibreWAFTs-flow-dav2.pt``
m     a1, Depth Anything V2 ViT-S     64.2 M     ``LibreWAFTm-flow.pt``
l     a2, Twins-SVT-Large encoder     136.0 M    ``LibreWAFTl-flow-twins.pt``
====  ==============================  =========  ==============================

``m`` is the a1 adaptation the upstream README recommends for downstream use
(``a1/tar-c-t.pth``), so it carries no variant suffix.

Licensing of the sub-components:

- size ``t`` builds its encoder from the DINOv3 code vendored with DEIMv2
  (``libreyolo/models/deimv2/engine/backbone/dinov3``, Meta DINOv3 License,
  see its ``LICENSE.md`` and ``NOTICE``). The ``t`` weights contain a DINOv3
  ViT-S/16 fine-tuned by the WAFT authors, so the DINOv3 License terms flow
  through to that checkpoint; the other sizes do not depend on DINOv3.
- sizes ``s`` and ``m`` reuse the Depth Anything V2 encoder code vendored in
  ``libreyolo/models/depth_anything`` (Apache-2.0).
- size ``l`` uses a plain-torch Twins copy derived from timm (Apache-2.0).

Prediction runs at native resolution by default: the network pads each frame
pair with upstream's ``Padder`` (to a multiple of 112, 32 or 16 depending on
the encoder) and crops the flow back. ``imgsz`` is a cap on the long side
(default 8192, in practice no cap): frames above it are downscaled before the
network and the flow is rescaled back to the original canvas. Five refinement
iterations are baked in (upstream evaluation default). Inference only.
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
from ..base.flow_sequence import NATIVE_FLOW_IMGSZ, downscale_rgb
from ..base.model import BaseModel
from ._vendor.waft import WAFTA1, WAFTA2

logger = logging.getLogger(__name__)

# size -> (algorithm, feature encoder, published weight variant or None)
WAFT_ARCHS: Dict[str, Tuple[str, str, Optional[str]]] = {
    "t": ("a2", "dinov3", "dinov3"),
    "s": ("a2", "dav2", "dav2"),
    "m": ("a1", "dav2", None),
    "l": ("a2", "twins", "twins"),
}
WAFT_VARIANTS = ("dav2", "dinov3", "twins")
WAFT_ITERS = 5


def _unpublished() -> Dict[str, str]:
    out: Dict[str, str] = {}
    for size, (_, _, published) in WAFT_ARCHS.items():
        for variant in (None,) + WAFT_VARIANTS:
            if variant == published:
                continue
            name = f"LibreWAFT{size}-flow" + (f"-{variant}" if variant else "")
            want = f"LibreWAFT{size}-flow" + (f"-{published}" if published else "")
            out[name] = f"WAFT size {size} is published only as {want}.pt."
    return out


def preprocess_numpy(img: np.ndarray) -> Tuple[np.ndarray, float]:
    """HWC uint8 RGB -> CHW float32 in [0, 255] (normalisation is in-graph)."""
    return np.ascontiguousarray(img.transpose(2, 0, 1).astype(np.float32)), 1.0


class LibreWAFT(BaseModel):
    """WAFT optical flow: frame pair -> dense (dx, dy) field."""

    FAMILY = "waft"
    FILENAME_PREFIX = "LibreWAFT"
    WEIGHT_EXT = ".pt"
    # Nominal canvas only (export and metadata); predict keeps native size.
    # ``imgsz`` is a cap on the long side; the default means native resolution.
    INPUT_SIZES: ClassVar[Dict[str, int]] = {
        size: NATIVE_FLOW_IMGSZ for size in ("t", "s", "m", "l")
    }
    SUPPORTED_TASKS = ("flow",)
    DEFAULT_TASK = "flow"
    REQUIRE_TASK_SUFFIX = True
    WEIGHT_VARIANTS: ClassVar[Tuple[str, ...]] = WAFT_VARIANTS
    UNPUBLISHED_WEIGHTS: ClassVar[Dict[str, str]] = _unpublished()
    FLOW_CONTEXT: ClassVar[tuple[int, int]] = (0, 0)
    TRAIN_CONFIG = None
    SUPPORTS_BATCHED_PREDICT = False
    TTA_ENABLED = False

    # ====================================================================
    # Checkpoint detection
    # ====================================================================

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        return (
            "warp_linear.weight" in weights_dict
            and "refine_transform.weight" in weights_dict
            and "refine_net.blks.0.attn.qkv.weight" in weights_dict
            and cls.detect_size(weights_dict) is not None
        )

    @classmethod
    def detect_size(cls, weights_dict: dict) -> Optional[str]:
        if "da_feature.depth_anything.pretrained.cls_token" in weights_dict:
            return "m"
        if "encoder.backbone.model.patch_embeds.0.proj.weight" in weights_dict:
            return "l"
        if "encoder.encoder.storage_tokens" in weights_dict:
            return "t"
        if "encoder.encoder.pos_embed" in weights_dict:
            return "s"
        return None

    @classmethod
    def detect_nb_classes(cls, weights_dict: dict) -> Optional[int]:
        return 1

    # ====================================================================
    # Construction
    # ====================================================================

    def __init__(
        self,
        model_path=None,
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
        algorithm, encoder, _ = WAFT_ARCHS[self.size]
        if algorithm == "a1":
            return WAFTA1("vits", "vits", iters=WAFT_ITERS)
        return WAFTA2(encoder, "vits", iters=WAFT_ITERS)

    def _get_available_layers(self) -> Dict[str, nn.Module]:
        net = self.model
        layers = {"fnet": net.fnet, "refine_net": net.refine_net}
        layers["encoder"] = net.da_feature if hasattr(net, "da_feature") else net.encoder
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
        input_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Image.Image, Tuple[int, int], float]:
        # Native resolution by default; an explicit ``imgsz`` caps the long side
        # (the frame is downscaled here, the network pads internally and crops
        # the flow back, ``_postprocess`` rescales it to the original canvas).
        max_side = input_size if input_size is not None else self._get_input_size()
        img = ImageLoader.load(image, color_format=color_format)
        orig_w, orig_h = img.size
        frame, ratio = downscale_rgb(np.asarray(img.convert("RGB")), max_side)
        chw, _ = preprocess_numpy(frame)
        return torch.from_numpy(chw).unsqueeze(0), img, (orig_w, orig_h), ratio

    def _forward(self, input_tensor: torch.Tensor) -> Any:
        raise ValueError(
            "WAFT is an optical flow model: pass at least two frames (a list of "
            "images, a directory, or a video), not a single image."
        )

    def _forward_pair(self, image1: torch.Tensor, image2: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Raw WAFT output for ``image1`` -> ``image2`` (RGB 0-255, B3HW)."""
        return self.model(image1, image2)

    def _forward_flow(self, window: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self._forward_pair(window[0:1], window[1:2])

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
        h, w = flow.shape[-2:]
        if (h, w) != (orig_h, orig_w):
            flow = F.interpolate(flow, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
            scale = flow.new_tensor([orig_w / w, orig_h / h]).view(1, 2, 1, 1)
            flow = flow * scale
        return {"flow": flow[0].permute(1, 2, 0).contiguous().cpu()}

    # ====================================================================
    # Out of scope
    # ====================================================================

    def train(self, *args, **kwargs):
        raise NotImplementedError(
            "Training WAFT is not supported by LibreYOLO; the family is "
            "inference-only. Use a pretrained LibreWAFT checkpoint."
        )
