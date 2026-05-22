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

    # Per-checkpoint bin geometry: {num_bins: (bin_width_deg, offset_deg)}.
    # Gaze360 L2CS uses 90 bins of 4 deg spanning [-180, 180); MPIIGaze uses
    # 28 bins of 3 deg spanning [-42, 42). These decode constants are not
    # derivable from the bin count alone, so they are pinned per known
    # upstream training configuration.
    _BIN_GEOMETRY: ClassVar[Dict[int, tuple]] = {
        90: (4.0, -180.0),  # Gaze360
        28: (3.0, -42.0),   # MPIIGaze
    }

    # TTA, tiling, and validation all make no sense for two-stage gaze.
    TTA_ENABLED = False

    # State-dict fingerprint shared by every L2CS checkpoint.
    _SIGNATURE_KEYS = ("fc_yaw_gaze.weight", "fc_pitch_gaze.weight")

    # Bring-your-own weights. The L2CS Gaze360 checkpoint cannot be mirrored by
    # LibreYOLO — the Gaze360 dataset license forbids redistributing models
    # trained on it — so there is no auto-download. The user fetches it from
    # the official L2CS-Net distribution.
    _WEIGHTS_URL = (
        "https://drive.google.com/file/d/18S956r4jnHtSeT8z8t3z8AoJZjVnNqPJ/view"
    )
    _GAZE360_LICENSE_URL = (
        "https://github.com/erkil1452/gaze360/blob/master/LICENSE.md"
    )

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

    @staticmethod
    def _detect_num_bins(model_path) -> Optional[int]:
        """Infer the angle-bin count from a checkpoint's ``fc_yaw_gaze`` weight.

        Accepts a state-dict (the factory's pretrained path) or a file path.
        Returns None when the count cannot be determined.
        """
        state: Optional[dict] = None
        if isinstance(model_path, dict):
            state = model_path
        elif isinstance(model_path, (str, Path)):
            try:
                from ...utils.serialization import load_untrusted_torch_file

                resolved = BaseModel._resolve_weights_path(str(model_path))
                if not Path(resolved).exists():
                    return None
                loaded = load_untrusted_torch_file(
                    resolved, map_location="cpu", context="L2CS bin-count probe"
                )
                if isinstance(loaded, dict):
                    state = loaded.get("model", loaded.get("state_dict", loaded))
            except Exception:
                return None
        if not isinstance(state, dict):
            return None
        weight = state.get("fc_yaw_gaze.weight")
        if weight is not None and getattr(weight, "ndim", 0) >= 1:
            return int(weight.shape[0])
        return None

    @classmethod
    def get_download_url(cls, filename: str) -> None:
        # L2CS gaze weights are trained on the Gaze360 dataset, whose license
        # forbids redistributing models derived from it. They cannot be
        # mirrored on the LibreYOLO HuggingFace org, so there is intentionally
        # no auto-download URL — weights are bring-your-own. See _weights_help.
        return None

    @classmethod
    def _weights_help(cls, requested: str) -> str:
        """User-facing guidance shown when L2CS weights are missing."""
        return (
            f"L2CS gaze weights not found: {requested}\n\n"
            "LibreYOLO does not bundle or mirror L2CS weights. The model is "
            "trained on the Gaze360 dataset, whose license forbids "
            "redistributing derived models, so the checkpoint cannot be "
            "auto-downloaded.\n\n"
            "1. Download the official checkpoint 'L2CSNet_gaze360.pkl' "
            "(ResNet-50, Gaze360) from:\n"
            f"     {cls._WEIGHTS_URL}\n"
            "2. Pass its path explicitly, e.g.:\n"
            "     LibreL2CS(r'C:\\path\\to\\L2CSNet_gaze360.pkl')\n\n"
            "The Gaze360 weights are licensed for research / non-commercial "
            f"use only:\n     {cls._GAZE360_LICENSE_URL}"
        )

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
        num_bins: int | None = None,
        face_detector: Optional[FaceDetector] = None,
        **kwargs,
    ):
        # The angle-bin count must match the checkpoint's head width exactly,
        # so infer it from the weights when possible; fall back to an explicit
        # num_bins, then to the Gaze360 default.
        detected_bins = self._detect_num_bins(model_path)
        self.num_bins = (
            detected_bins
            if detected_bins is not None
            else (num_bins if num_bins is not None else self.NUM_BINS)
        )
        if self.num_bins in self._BIN_GEOMETRY:
            self.bin_width_deg, self.offset_deg = self._BIN_GEOMETRY[self.num_bins]
        else:
            self.bin_width_deg, self.offset_deg = self._BIN_GEOMETRY[self.NUM_BINS]
            logger.warning(
                "L2CS checkpoint has %d angle bins; no known decode geometry "
                "for that bin count. Falling back to Gaze360 (4 deg / -180 deg) "
                "— decoded angles may be wrong. Use a 90-bin Gaze360 or 28-bin "
                "MPIIGaze checkpoint for correct results.",
                self.num_bins,
            )
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
        # Upstream L2CS checkpoints carry a vestigial ``fc_finetune`` layer that
        # this port omits. Non-strict loading lets those unused keys be ignored
        # on load instead of raising.
        return False

    def _load_weights(self, model_path: str) -> None:
        # L2CS has no auto-download (see get_download_url). When the file is
        # missing, fail fast with actionable guidance — the official download
        # link and the license — instead of the generic "could not determine
        # download URL" error from the shared download path.
        path = Path(model_path)
        if not path.exists():
            alt = Path("weights") / path.name
            if alt.exists():
                path = alt
            else:
                raise FileNotFoundError(self._weights_help(model_path))
        super()._load_weights(str(path))

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
