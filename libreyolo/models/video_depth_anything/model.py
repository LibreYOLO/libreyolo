"""LibreYOLO wrapper for Video Depth Anything.

Video Depth Anything (https://github.com/DepthAnything/Video-Depth-Anything,
CVPR 2025) puts a temporal DPT head on the Depth Anything V2 encoder so depth
stays consistent across the frames of a video. It predicts relative inverse
depth like Depth Anything V2, so it implements the shared ``depth`` task
(``Results.depth_map``) and its dataset, validator and checkpoint schema.

Multi-frame input goes through the shared sequence runner
(``SEQUENCE_INPUT = True``, ``BaseModel._forward_sequence``): any multi-image
source (a list of frames, a directory sorted by name, a video, a stream) is
one sequence, and ``N`` frames yield ``N`` results, one depth map per frame on
that frame's canvas. Inside, frames run in upstream's overlapping 32-frame
windows, and every window is scale/shift aligned to the ones before it, so the
whole sequence shares one relative scale. A single image is a one-frame
sequence (upstream pads it to a window with copies of itself).

The vendored code is Apache-2.0 (``_vendor/LICENSE``). Weights: Small is
Apache-2.0; Base and Large are CC-BY-NC-4.0 (non-commercial). Inference-only.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, ClassVar, Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from ...utils.image_loader import ImageInput, ImageLoader
from ..base.model import BaseModel
from .nn import INFER_LEN, LibreVideoDepthAnythingNet
from .utils import check_not_metric, preprocess_frame, stitch_video_depth

logger = logging.getLogger(__name__)

_NON_COMMERCIAL_NOTICE = (
    "Video Depth Anything {size} weights are licensed CC-BY-NC-4.0 "
    "(non-commercial) by their authors. You are responsible for complying "
    "with that licence; see https://huggingface.co/LibreYOLO/{repo}."
)


class LibreVideoDepthAnything(BaseModel):
    """Video Depth Anything: frame sequence -> temporally consistent depth."""

    FAMILY = "video_depth_anything"
    FILENAME_PREFIX = "LibreVideoDepthAnything"
    WEIGHT_EXT = ".pt"
    # Upstream default ``input_size`` (37 * 14) for every encoder; wider than
    # 16:9 frames run proportionally smaller, as upstream.
    INPUT_SIZES: ClassVar[Dict[str, int]] = {"s": 518, "b": 518, "l": 518}
    SUPPORTED_TASKS = ("depth",)
    DEFAULT_TASK = "depth"
    REQUIRE_TASK_SUFFIX = True
    # Every source is a frame sequence; see BaseModel._forward_sequence.
    SEQUENCE_INPUT: ClassVar[bool] = True
    SUPPORTS_BATCHED_PREDICT = False
    TTA_ENABLED = False

    depth_imgsz_divisor = 14
    depth_resize_mode = "letterbox"

    _SIZE_TO_ENCODER: ClassVar[Dict[str, str]] = {"s": "vits", "b": "vitb", "l": "vitl"}
    _EMBED_DIM_TO_SIZE: ClassVar[Dict[int, str]] = {384: "s", 768: "b", 1024: "l"}
    _NON_COMMERCIAL_SIZES: ClassVar[Dict[str, str]] = {"b": "Base", "l": "Large"}

    # ====================================================================
    # Checkpoint detection
    # ====================================================================

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        # DINOv2 under ``pretrained.*`` plus the temporal head's motion
        # modules. Depth Anything V2 names its head ``depth_head.*`` and has
        # no motion modules, so neither family claims the other's files.
        has_encoder = "pretrained.cls_token" in weights_dict
        has_temporal = any(
            k.startswith("head.motion_modules.") for k in weights_dict
        )
        has_dpt = any(k.startswith("head.scratch.") for k in weights_dict)
        return has_encoder and has_temporal and has_dpt

    @classmethod
    def detect_size(cls, weights_dict: dict) -> Optional[str]:
        cls_token = weights_dict.get("pretrained.cls_token")
        if cls_token is not None and getattr(cls_token, "ndim", 0) >= 1:
            return cls._EMBED_DIM_TO_SIZE.get(int(cls_token.shape[-1]))
        return None

    @classmethod
    def detect_nb_classes(cls, weights_dict: dict) -> Optional[int]:
        return 1

    @classmethod
    def upstream_checkpoint_metadata(cls, loaded: Any, *, source) -> dict:
        """Refuse upstream metric checkpoints during runtime auto-conversion."""
        check_not_metric(source)
        return {}

    @classmethod
    def get_download_notice(cls, filename: str, url: str) -> Optional[str]:
        size = cls.detect_size_from_filename(Path(filename).name)
        if size not in cls._NON_COMMERCIAL_SIZES:
            return None
        repo = Path(filename).name.rsplit(cls.WEIGHT_EXT, 1)[0]
        return _NON_COMMERCIAL_NOTICE.format(
            size=cls._NON_COMMERCIAL_SIZES[size], repo=repo
        )

    # ====================================================================
    # Construction
    # ====================================================================

    def __init__(
        self,
        model_path,
        size: str = "s",
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
        self.names = {0: "depth"}

    def _init_model(self) -> nn.Module:
        return LibreVideoDepthAnythingNet(self._SIZE_TO_ENCODER[self.size])

    def _get_available_layers(self) -> Dict[str, nn.Module]:
        return {"pretrained": self.model.pretrained, "head": self.model.head}

    # ====================================================================
    # Inference surface
    # ====================================================================

    @staticmethod
    def _get_preprocess_numpy():
        return preprocess_frame

    def _preprocess(
        self,
        image: ImageInput,
        color_format: str = "auto",
        input_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Image.Image, Tuple[int, int], float]:
        effective_res = int(input_size) if input_size is not None else self._get_input_size()
        if effective_res % self.depth_imgsz_divisor:
            raise ValueError(
                f"Video Depth Anything imgsz={effective_res} must be divisible by "
                f"{self.depth_imgsz_divisor} (DINOv2 patch grid)."
            )
        img = ImageLoader.load(image, color_format=color_format)
        orig_w, orig_h = img.size
        chw = preprocess_frame(np.asarray(img.convert("RGB")), effective_res)
        return torch.from_numpy(chw).unsqueeze(0), img, (orig_w, orig_h), 1.0

    def _run_window(
        self, window: torch.Tensor, original_size: Tuple[int, int]
    ) -> List[np.ndarray]:
        """One ``(1, 32, 3, h, w)`` window to 32 depth maps on the original canvas."""
        orig_w, orig_h = original_size
        param = next(self.model.parameters())
        window = window.to(device=param.device, dtype=param.dtype)
        with torch.no_grad():
            depth = self.model(window)  # (1, T, h, w)
            depth = depth.to(window.dtype)
            depth = F.interpolate(
                depth.flatten(0, 1).unsqueeze(1),
                size=(orig_h, orig_w),
                mode="bilinear",
                align_corners=True,
            )
        return [depth[i][0].float().cpu().numpy() for i in range(depth.shape[0])]

    def _forward_sequence(
        self, frames: Iterable[Tuple[torch.Tensor, Tuple[int, int]]]
    ) -> Iterator[Dict[str, torch.Tensor]]:
        """Runner hook: one aligned depth map per frame, in order."""
        for depth in stitch_video_depth(frames, self._run_window):
            yield {"depth": torch.from_numpy(depth)}

    def _forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Independent images ``(B, 3, h, w)`` to ``(B, 1, h, w)`` depth.

        Used by validation and any caller with unrelated images: each image is
        its own one-frame sequence, which upstream pads to a full window with
        copies of the frame. Costs one 32-frame forward per image.
        """
        outputs = []
        for index in range(input_tensor.shape[0]):
            clip = input_tensor[index : index + 1].unsqueeze(1)
            clip = clip.expand(-1, INFER_LEN, -1, -1, -1)
            outputs.append(self.model(clip)[:, :1])
        return torch.cat(outputs, dim=0)

    def _postprocess(
        self,
        output: Any,
        conf_thres: float,
        iou_thres: float,
        original_size: Tuple[int, int],
        max_det: int = 300,
        **kwargs,
    ) -> Dict:
        orig_w, orig_h = original_size
        if isinstance(output, dict):
            # Sequence path: already on the original canvas.
            depth = output["depth"]
            if tuple(depth.shape[-2:]) != (orig_h, orig_w):
                raise ValueError(
                    f"Sequence depth map has shape {tuple(depth.shape)}; expected "
                    f"({orig_h}, {orig_w})."
                )
            return {"depth": depth.float().cpu()}
        depth = F.interpolate(
            output.float(), size=(orig_h, orig_w), mode="bilinear", align_corners=True
        )
        return {"depth": depth[0, 0].cpu()}

    # ====================================================================
    # Out of scope
    # ====================================================================

    def train(self, *args, **kwargs):
        raise NotImplementedError(
            "Training and fine-tuning Video Depth Anything are not supported by "
            "LibreYOLO. Use a pretrained checkpoint, or convert an upstream one "
            "with weights/convert_video_depth_anything_weights.py."
        )

    def export(self, format: str = "onnx", **kwargs) -> str:
        raise NotImplementedError(
            "Video Depth Anything export is not supported: the network reads a "
            "32-frame window and stitches windows on the host, which the "
            "single-image depth export contract does not describe."
        )
