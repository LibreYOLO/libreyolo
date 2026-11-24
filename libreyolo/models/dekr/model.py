"""LibreDEKR: bottom-up multi-person pose estimation (inference only).

DEKR ("Disentangled Keypoint Regression", arXiv:2104.02300) predicts a dense
heatmap per keypoint type plus a person-centre heatmap, and a dense vector from
each centre location to every keypoint. It needs no person detector, which makes
it the first genuinely bottom-up pose family in LibreYOLO.

The shipped model is DEKR-W32-NO-DC: an HRNet-W32 trunk with the paper's
deformable offset head replaced by standard convolutions at dilation 5. That
substitution is what lets the graph export without custom operators, and it is a
different architecture from the original deformable DEKR-W32 -- the two
checkpoints are not interchangeable. LibreYOLO's public size is ``w32`` and the
export-friendly variant is recorded as ``variant="no_dc"`` in checkpoint
metadata, per ``docs/nomenclature.md``.

Weights: the released checkpoint is served from Deci's public CDN. No
per-artifact redistribution grant was found for it, so LibreYOLO links to the
CDN rather than mirroring it on the LibreYOLO HuggingFace org -- the same
treatment YOLO-NAS gets. See ``THIRD_PARTY_NOTICES.txt``.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, ClassVar, Optional

import numpy as np
import torch
from torch import nn

from ...data.pose_metadata import (
    COCO17_FLIP_IDX,
    COCO17_KEYPOINT_NAMES,
    COCO17_OKS_SIGMAS,
    COCO17_SKELETON,
    default_oks_sigmas,
)
from ...postprocess.dekr import (
    DEKR_KEYPOINT_THRESHOLD,
    DEKR_MAX_NUM_PEOPLE,
    DEKR_NMS_NUM_THRESHOLD,
    DEKR_NMS_THRESHOLD,
    DEKR_OUTPUT_STRIDE,
    postprocess_dekr,
)
from ...preprocess.dekr import preprocess_image, preprocess_numpy
from ..base import BaseModel
from .nn import LibreDEKRModel
from .utils import strip_module_prefix, unwrap_dekr_checkpoint

logger = logging.getLogger(__name__)

# Architecture signatures used by can_load / detect_*. Each is unique to the
# DEKR head layout: no other family carries a `transition_heatmap` next to an
# indexed `offset_final_layer` stack.
_HEATMAP_HEAD_KEY = "head_heatmap.1.weight"
_HEATMAP_TRANSITION_KEY = "transition_heatmap.0.weight"
_OFFSET_TRANSITION_KEY = "transition_offset.0.weight"
_OFFSET_FINAL_RE = re.compile(r"^offset_final_layer\.(\d+)\.weight$")

# Channel counts fixed by DEKR-W32-NO-DC.
_CONCAT_CHANNELS = 480
_HEATMAP_CHANNELS = 32
_OFFSET_CHANNELS_PER_KEYPOINT = 15
_BRANCH_WIDTHS = (32, 64, 128, 256)


class LibreDEKR(BaseModel):
    """DEKR-W32-NO-DC bottom-up pose estimator."""

    FAMILY = "dekr"
    FILENAME_PREFIX = "LibreDEKR"
    INPUT_SIZES: ClassVar[dict[str, int]] = {"w32": 640}
    SUPPORTED_TASKS = ("pose",)
    DEFAULT_TASK = "pose"
    REQUIRE_TASK_SUFFIX = True
    TRAIN_CONFIG = None
    POSE_NUM_KEYPOINTS = 17
    ARCH_VARIANT = "no_dc"
    OUTPUT_STRIDE = DEKR_OUTPUT_STRIDE

    _DECI_CDN_BASE = "https://d2gjn4b69gu75n.cloudfront.net/models"
    _SOURCE_FILENAME = "dekr_w32_no_dc_coco_pose.pth"
    # SHA-256 of the released DEKR-W32-NO-DC checkpoint, observed 2026-08-10 at
    # 357,227,441 bytes. Auto-downloaded third-party pickles are verified
    # against this pin before unpickling, so a tampered CDN object fails closed.
    _SOURCE_SHA256 = "e5c4797205ddabd5efcebee470ee669c657e6b62f03948d57996e7d9f4022a6b"

    # ---- family recognition -------------------------------------------------

    @classmethod
    def _offset_branch_count(cls, weights_dict: dict) -> Optional[int]:
        """Return K when offset heads are a complete 0..K-1 run of 15->2 convs."""
        indices: set[int] = set()
        for key, value in weights_dict.items():
            match = _OFFSET_FINAL_RE.match(key)
            if match is None:
                continue
            shape = getattr(value, "shape", None)
            if shape is None or len(shape) != 4:
                return None
            if int(shape[0]) != 2 or int(shape[1]) != _OFFSET_CHANNELS_PER_KEYPOINT:
                return None
            indices.add(int(match.group(1)))
        if not indices or indices != set(range(len(indices))):
            return None
        return len(indices)

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        heatmap = weights_dict.get(_HEATMAP_HEAD_KEY)
        transition = weights_dict.get(_HEATMAP_TRANSITION_KEY)
        offset_transition = weights_dict.get(_OFFSET_TRANSITION_KEY)
        if heatmap is None or transition is None or offset_transition is None:
            return False
        if getattr(heatmap, "ndim", 0) != 4 or getattr(transition, "ndim", 0) != 4:
            return False
        if getattr(offset_transition, "ndim", 0) != 4:
            return False
        # Concatenated multiresolution feature is 480 channels, squeezed to 32
        # for the heatmap branch.
        if tuple(transition.shape) != (
            _HEATMAP_CHANNELS,
            _CONCAT_CHANNELS,
            1,
            1,
        ):
            return False
        num_keypoints = cls._offset_branch_count(weights_dict)
        if num_keypoints is None:
            return False
        # Reject the original deformable DEKR: it carries an adapt_conv /
        # transform_matrix_conv offset head that this no-DC port cannot load.
        if any(
            "adapt_conv" in key
            or "transform_matrix_conv" in key
            or "translation_conv" in key
            for key in weights_dict
        ):
            return False
        if int(offset_transition.shape[0]) != num_keypoints * _OFFSET_CHANNELS_PER_KEYPOINT:
            return False
        return int(heatmap.shape[0]) == num_keypoints + 1

    @classmethod
    def detect_size(cls, weights_dict: dict) -> Optional[str]:
        """Infer the HRNet width from branch and transition shapes, not the name."""
        if not cls.can_load(weights_dict):
            return None
        widths = []
        for branch, expected in enumerate(_BRANCH_WIDTHS):
            key = f"stage4.0.branches.{branch}.0.conv1.weight"
            tensor = weights_dict.get(key)
            if tensor is None or getattr(tensor, "ndim", 0) != 4:
                return None
            widths.append(int(tensor.shape[0]))
        if tuple(widths) != _BRANCH_WIDTHS:
            return None
        return "w32"

    @classmethod
    def detect_nb_classes(cls, weights_dict: dict) -> Optional[int]:
        # DEKR is a single-category person model; it has no classification head.
        return 1 if cls.can_load(weights_dict) else None

    @classmethod
    def detect_num_keypoints(cls, weights_dict: dict) -> Optional[int]:
        return cls._offset_branch_count(weights_dict)

    @classmethod
    def detect_size_from_filename(cls, filename: str) -> Optional[str]:
        size = super().detect_size_from_filename(filename)
        if size is not None:
            return size
        # Also accept the native CDN filename so a locally staged upstream
        # checkpoint resolves. Shape validation still decides loadability.
        return "w32" if "dekr_w32_no_dc" in filename.lower() else None

    @classmethod
    def detect_task_from_filename(cls, filename: str) -> Optional[str]:
        if "dekr_w32_no_dc" in filename.lower():
            return "pose"
        return super().detect_task_from_filename(filename)

    @classmethod
    def get_download_url(cls, filename: str) -> Optional[str]:
        """Link to Deci's public CDN; DEKR weights are not mirrored by LibreYOLO."""
        if cls.detect_size_from_filename(filename) != "w32":
            return None
        return f"{cls._DECI_CDN_BASE}/{cls._SOURCE_FILENAME}"

    @classmethod
    def verify_downloaded_file(cls, local_path: str, source_url: str) -> None:
        """Checksum a freshly auto-downloaded DEKR pickle before it is loaded."""
        import hashlib
        from urllib.parse import urlparse

        name = Path(urlparse(source_url).path).name
        if name != cls._SOURCE_FILENAME:
            Path(local_path).unlink(missing_ok=True)
            raise RuntimeError(
                f"Refusing to auto-load DEKR checkpoint '{name}': no pinned "
                "checksum is known for it. Download it manually from a source "
                "you trust and pass its path instead."
            )
        digest = hashlib.sha256()
        with open(local_path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        actual = digest.hexdigest()
        if actual != cls._SOURCE_SHA256:
            Path(local_path).unlink(missing_ok=True)
            raise RuntimeError(
                f"Checksum mismatch for downloaded DEKR checkpoint '{name}': "
                f"expected {cls._SOURCE_SHA256}, got {actual}. Refusing to load "
                "a possibly tampered file."
            )

    # ---- construction -------------------------------------------------------

    def __init__(
        self,
        model_path=None,
        size: str = "w32",
        nb_classes: int = 1,
        device: str = "auto",
        task: str | None = None,
        num_keypoints: int | None = None,
        **kwargs,
    ) -> None:
        self.num_keypoints = int(num_keypoints or self.POSE_NUM_KEYPOINTS)
        super().__init__(
            model_path=model_path,
            size=size,
            nb_classes=1,
            device=device,
            task=task,
            **kwargs,
        )
        self.keypoint_dim = 3
        self.variant = self.ARCH_VARIANT
        # BaseModel.__init__ records the path but leaves loading to the family.
        if model_path is not None and isinstance(model_path, (str, Path)):
            self._load_weights(str(model_path))
        # The released head is COCO person-only; keep the semantic name even if
        # a metadata-less upstream file was wrapped with a generic fallback.
        self.names = {0: "person"}
        self._refresh_pose_metadata()
        self.model.eval()

    def _refresh_pose_metadata(self) -> None:
        """Publish keypoint metadata matching the current keypoint count."""
        if self.num_keypoints == self.POSE_NUM_KEYPOINTS:
            self.keypoint_names = list(COCO17_KEYPOINT_NAMES)
            self.flip_idx = list(COCO17_FLIP_IDX)
            self.skeleton = [list(edge) for edge in COCO17_SKELETON]
            self.oks_sigmas = list(COCO17_OKS_SIGMAS)
        else:
            self.keypoint_names = [f"keypoint_{i}" for i in range(self.num_keypoints)]
            self.flip_idx = list(range(self.num_keypoints))
            self.skeleton = []
            self.oks_sigmas = default_oks_sigmas(self.num_keypoints)

    def _init_model(self) -> nn.Module:
        return LibreDEKRModel(num_keypoints=self.num_keypoints)

    def _get_available_layers(self) -> dict[str, nn.Module]:
        return {
            "stem": self.model.conv1,
            "backbone": self.model.stage4,
            "heatmap_head": self.model.head_heatmap,
            "offset_head": self.model.offset_final_layer,
        }

    # ---- checkpoint handling ------------------------------------------------

    def _prepare_state_dict(self, state_dict: dict) -> dict:
        """Accept both a LibreYOLO checkpoint body and a raw upstream artifact.

        The released artifact is a dict whose model weights live under ``net``
        with a leading ``module.`` on every key, alongside optimizer and scaler
        blobs that are dropped here and never reach a LibreYOLO checkpoint.
        """
        prepared = strip_module_prefix(unwrap_dekr_checkpoint(state_dict))
        keypoints = self.detect_num_keypoints(prepared)
        if keypoints is not None and keypoints != self.num_keypoints:
            self.num_keypoints = keypoints
            self.model.replace_head(keypoints)
            self.model.to(self.device)
            self._refresh_pose_metadata()
        return prepared

    def _strict_loading(self) -> bool:
        # Every released tensor maps onto a native parameter; nothing is
        # regenerated or dropped, so a missing key is a real error.
        return True

    # ---- inference ----------------------------------------------------------

    @staticmethod
    def _get_preprocess_numpy():
        return preprocess_numpy

    def _preprocess(self, image, color_format="auto", input_size=None):
        return preprocess_image(
            image,
            input_size=input_size or self._get_input_size(),
            color_format=color_format,
        )

    def _forward(self, input_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model(input_tensor)

    def _postprocess(
        self,
        output,
        conf_thres,
        iou_thres,
        original_size,
        max_det: int = DEKR_MAX_NUM_PEOPLE,
        ratio: float = 1.0,
        **kwargs,
    ) -> dict:
        # DEKR suppresses by normalized joint distance, not box IoU: pose NMS
        # has no IoU knob to honour.
        del iou_thres
        return postprocess_dekr(
            output,
            conf_thres=float(conf_thres),
            original_size=original_size,
            ratio=float(ratio),
            # DEKR's decoder is defined at 30 people; a larger LibreYOLO-wide
            # max_det must not silently widen the family's top-k.
            max_det=min(int(max_det), DEKR_MAX_NUM_PEOPLE),
            keypoint_threshold=float(
                kwargs.get("keypoint_threshold", DEKR_KEYPOINT_THRESHOLD)
            ),
            nms_threshold=float(kwargs.get("nms_threshold", DEKR_NMS_THRESHOLD)),
            nms_num_threshold=int(
                kwargs.get("nms_num_threshold", DEKR_NMS_NUM_THRESHOLD)
            ),
            output_stride=self.OUTPUT_STRIDE,
        )

    def predict_batch_raw(self, images: list[np.ndarray]) -> list[dict]:
        """Decode a stacked batch, one result dict per image.

        Exists so the batch-safety of the decoder is reachable from the public
        surface and not just from tests.
        """
        if not images:
            return []
        input_size = self._get_input_size()
        tensors, scales, sizes = [], [], []
        for image in images:
            chw, scale = preprocess_numpy(image, input_size=input_size)
            tensors.append(torch.from_numpy(chw))
            scales.append(scale)
            sizes.append((image.shape[1], image.shape[0]))
        batch = torch.stack(tensors).to(self.device)
        with torch.no_grad():
            heatmap, offsets = self.model(batch)
        return [
            postprocess_dekr(
                (heatmap, offsets),
                conf_thres=0.0,
                original_size=sizes[i],
                ratio=scales[i],
                batch_index=i,
                output_stride=self.OUTPUT_STRIDE,
            )
            for i in range(len(images))
        ]

    # ---- scope --------------------------------------------------------------

    def train(self, *args, **kwargs):
        raise NotImplementedError(
            "LibreDEKR ships inference-only. Training DEKR needs its dense "
            "centre/joint heatmap and offset target generator, the quality-focal "
            "heatmap loss and masked SmoothL1 offset loss, and the source pose "
            "transform stack -- none of which are in this port. Track the "
            "training milestone rather than fine-tuning through this entry point."
        )

    def export(self, format: str = "onnx", **kwargs) -> str:
        supported = {"onnx", "torchscript", "openvino", "tensorrt"}
        if format.lower() not in supported:
            raise NotImplementedError(
                f"LibreDEKR export to {format!r} is not implemented. The raw "
                "two-output pose graph is validated for ONNX, TorchScript, "
                "OpenVINO and TensorRT only."
            )
        return super().export(format=format, **kwargs)

    def _get_export_metadata(self) -> dict[str, Any]:
        """Carry everything the exported-backend decoder needs."""
        metadata = {}
        parent = getattr(super(), "_get_export_metadata", None)
        if callable(parent):
            metadata.update(parent())
        metadata.update(
            {
                "model_family": self.FAMILY,
                "task": "pose",
                "variant": self.ARCH_VARIANT,
                "output_names": ["heatmap_logits", "offsets"],
                "output_stride": self.OUTPUT_STRIDE,
                "num_keypoints": self.num_keypoints,
                "keypoint_dim": 3,
                "max_num_people": DEKR_MAX_NUM_PEOPLE,
                "keypoint_threshold": DEKR_KEYPOINT_THRESHOLD,
                "nms_threshold": DEKR_NMS_THRESHOLD,
                "nms_num_threshold": DEKR_NMS_NUM_THRESHOLD,
                "apply_sigmoid": True,
                "pad_value": 127,
                "normalization": {
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225],
                },
                "skeleton": self.skeleton,
                "flip_idx": self.flip_idx,
                "oks_sigmas": self.oks_sigmas,
            }
        )
        return metadata
