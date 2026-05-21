"""LibreYOLO wrapper for L2CS-Net gaze estimation (inference only).

L2CS-Net (Ahmednull/L2CS-Net, MIT, IEEE-ICIP 2022) is a two-stage gaze
estimator: a face detector locates faces, and a ResNet trunk with two
parallel angle-bin classification heads predicts pitch and yaw per face.
LibreYOLO embeds only the gaze head and a small protocol for plugging in
face detectors. Training and ground-truth-dataset validation are deliberately
out of scope here — train upstream at L2CS-Net.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, ClassVar, Dict, Optional

import torch
import torch.nn as nn

from ..base.model import BaseModel
from .face import FaceDetector, resolve_face_detector
from .nn import build_l2cs, detect_size_from_state_dict


logger = logging.getLogger(__name__)


class LibreL2CS(BaseModel):
    """L2CS gaze estimator: image → (per-face face box + (pitch, yaw) radians)."""

    FAMILY = "l2cs"
    FILENAME_PREFIX = "LibreL2CS"
    WEIGHT_EXT = ".pt"
    # All L2CS variants take 448×448 face crops; the size code is the ResNet
    # depth (r18/r34/r50/r101/r152) matching upstream's ``arch`` enum.
    INPUT_SIZES: ClassVar[Dict[str, int]] = {
        "r18": 448,
        "r34": 448,
        "r50": 448,
        "r101": 448,
        "r152": 448,
    }
    SUPPORTED_TASKS = ("gaze",)
    DEFAULT_TASK = "gaze"
    NUM_BINS = 90

    # TTA, tiling, and validation all make no sense for two-stage gaze.
    TTA_ENABLED = False

    # State-dict fingerprint shared by every L2CS checkpoint.
    _SIGNATURE_KEYS = ("fc_yaw_gaze.weight", "fc_pitch_gaze.weight")

    # Upstream checkpoints include a vestigial ``fc_finetune`` layer that
    # is never invoked in forward. We drop those keys before loading and
    # therefore tolerate the absence of strict matching.
    _DROPPED_KEY_PREFIXES = ("fc_finetune.",)

    # =========================================================================
    # Detection of weights belonging to this family
    # =========================================================================

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        if not all(key in weights_dict for key in cls._SIGNATURE_KEYS):
            return False
        yaw = weights_dict["fc_yaw_gaze.weight"]
        pitch = weights_dict["fc_pitch_gaze.weight"]
        return yaw.shape == pitch.shape

    @classmethod
    def detect_size(cls, weights_dict: dict) -> Optional[str]:
        return detect_size_from_state_dict(weights_dict)

    @classmethod
    def detect_nb_classes(cls, weights_dict: dict) -> Optional[int]:
        # L2CS itself doesn't classify objects; report a single "face" class so
        # the surrounding factory plumbing has a sensible value.
        return 1

    # =========================================================================
    # Construction
    # =========================================================================

    def __init__(
        self,
        model_path,
        size: str = "r50",
        nb_classes: int = 1,
        device: str = "auto",
        task: str | None = None,
        num_bins: int = NUM_BINS,
        face_detector: Optional[FaceDetector] = None,
        **kwargs,
    ):
        self.num_bins = num_bins
        super().__init__(
            model_path=model_path,
            size=size,
            nb_classes=1,  # always 1 ("face"); ignore caller's nb_classes
            device=device,
            task=task,
            **kwargs,
        )
        self.names = {0: "face"}
        # Resolve the optional default face detector once at construction time.
        self.face_detector = (
            resolve_face_detector(face_detector) if face_detector is not None else None
        )

        # If we built with model_path=None, the BaseModel left the network in
        # training mode (intended for training-from-scratch flows). Gaze is
        # inference-only, so flip eval() unconditionally.
        self.model.eval()

        if model_path is not None and isinstance(model_path, (str, Path)):
            self._load_weights(str(model_path))

    # =========================================================================
    # BaseModel abstract surface — gaze uses GazeInferenceRunner instead
    # =========================================================================

    def _init_model(self) -> nn.Module:
        return build_l2cs(self.size, num_bins=self.num_bins)

    def _get_available_layers(self) -> Dict[str, nn.Module]:
        return {name: module for name, module in self.model.named_modules() if name}

    @staticmethod
    def _get_preprocess_numpy():
        # The standard detection-shaped numpy preprocess is not applicable —
        # gaze preprocessing operates on per-face crops inside the runner.
        raise NotImplementedError(
            "LibreL2CS preprocesses per face inside GazeInferenceRunner; "
            "see libreyolo.models.l2cs.utils.preprocess_face_crops."
        )

    def _preprocess(self, *args, **kwargs):
        raise NotImplementedError(
            "LibreL2CS does not use the detection-shaped _preprocess hook; "
            "GazeInferenceRunner orchestrates face detection and cropping."
        )

    def _forward(self, *args, **kwargs):
        raise NotImplementedError(
            "LibreL2CS does not use the detection-shaped _forward hook; "
            "GazeInferenceRunner calls the underlying ResNet directly."
        )

    def _postprocess(self, *args, **kwargs):
        raise NotImplementedError(
            "LibreL2CS does not use the detection-shaped _postprocess hook; "
            "see libreyolo.models.l2cs.utils.bin_logits_to_angles."
        )

    def _strict_loading(self) -> bool:
        return False

    def _prepare_state_dict(self, state_dict: dict) -> dict:
        return {
            k: v
            for k, v in state_dict.items()
            if not any(k.startswith(p) for p in self._DROPPED_KEY_PREFIXES)
        }

    # =========================================================================
    # Override the runner
    # =========================================================================

    @property
    def _runner(self):
        if getattr(self, "_runner_instance", None) is None:
            from .inference import GazeInferenceRunner
            self._runner_instance = GazeInferenceRunner(self)
        return self._runner_instance

    # =========================================================================
    # Train / val are explicitly out of scope
    # =========================================================================

    def train(self, *args, **kwargs):
        raise NotImplementedError(
            "Training is out of scope for LibreL2CS in LibreYOLO. "
            "Train upstream at https://github.com/Ahmednull/L2CS-Net and load the "
            "resulting state dict here."
        )

    def val(self, *args, **kwargs):
        raise NotImplementedError(
            "Validation against gaze ground-truth datasets (MPIIGaze, Gaze360) is "
            "out of scope for LibreL2CS in LibreYOLO. Evaluate upstream."
        )

    def export(self, format: str = "onnx", **kwargs) -> str:
        # ONNX export is planned but requires a small head wrapper. Block
        # other formats explicitly so users don't hit a confusing failure
        # inside the detection-shaped BaseExporter path.
        if format.lower() != "onnx":
            raise NotImplementedError(
                f"LibreL2CS export to {format!r} is not implemented. "
                "Only 'onnx' is planned."
            )
        raise NotImplementedError(
            "LibreL2CS ONNX export is not yet implemented. Track this on the "
            "gaze integration PR."
        )
