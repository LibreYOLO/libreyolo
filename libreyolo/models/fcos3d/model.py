"""Native FCOS3D inference with camera calibration and standard Results."""

import hashlib
import logging
import time
from pathlib import Path
from typing import ClassVar

import numpy as np
import torch

from ...utils.image_loader import SUPPORTED_EXTENSIONS, ImageLoader
from ...utils.results import Results
from ...utils.serialization import load_untrusted_torch_file
from .nn import FCOS3DNetwork
from .utils import NAMES, calibration, decode, payloads, preprocess

logger = logging.getLogger(__name__)
HF_REPO = "LibreYOLO/LibreFCOS3D"
HF_REVISION = "af69c4abfb265f63695ecbaa68b4167f0c150605"
WEIGHT_FILE = (
    "fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune_"
    "20210717_095645-8d806dc2.pth"
)
WEIGHT_SHA256 = "8d806dc2ecae85bc8eaba1f16dccdf03459317ca6aed7984cfda33c2a2bc33a8"
TERMS_URL = "https://www.nuscenes.org/terms-of-use"


class LibreFCOS3D:
    """FCOS3D R101-DCN for the ten nuScenes classes, inference-only.

    ``model_path`` accepts an unchanged official checkpoint. Omitting it
    downloads LibreYOLO's byte-identical, revision-pinned mirror. Those
    weights are trained on nuScenes under non-commercial terms and are not
    covered by LibreYOLO's MIT license.
    Supply original-image pinhole intrinsics to predict.
    Images retain their resolution; padding is only on the right and bottom.
    Single images return Results; sequences/directories return lists and
    stream=True returns a generator. NumPy input color follows ImageLoader.
    The aligned 2D boxes are projected cuboid hulls, not a separate 2D head.
    Training, validation, tracking and export are not implemented.
    """

    FAMILY = "fcos3d"
    CLI_COMMAND = "fcos3d"
    FILENAME_PREFIX = "LibreFCOS3D"
    SUPPORTED_TASKS = ("detect3d",)
    DEFAULT_TASK = "detect3d"
    INPUT_SIZES: ClassVar[dict[str, int]] = {"r101": 1600}
    TASK_INPUT_SIZES: ClassVar[dict] = {}
    DEFAULT_CONF = 0.05
    DEFAULT_IOU = 0.8

    def __init__(self, model_path=None, *, device="auto"):
        self.model_path = self._resolve_checkpoint(model_path)
        if isinstance(device, int) or str(device).isdigit():
            device = f"cuda:{device}"
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError(
                "FCOS3D supports cpu or cuda; torchvision deform_conv2d does not support MPS."
            )
        checkpoint = load_untrusted_torch_file(self.model_path, map_location="cpu")
        if not isinstance(checkpoint, dict) or not isinstance(
            checkpoint.get("state_dict"), dict
        ):
            raise TypeError("Expected an official FCOS3D checkpoint with a state_dict.")
        metadata = checkpoint.get("meta", {})
        if not isinstance(metadata, dict):
            raise TypeError("FCOS3D checkpoint meta must be a dictionary.")
        classes = metadata.get("CLASSES")
        if classes is not None and tuple(classes) != tuple(NAMES.values()):
            raise ValueError(
                "FCOS3D checkpoint classes do not match the supported nuScenes class order."
            )
        self.model = FCOS3DNetwork()
        self.model.load_state_dict(checkpoint["state_dict"], strict=True)
        self.model.to(self.device).eval()
        self.names = NAMES.copy()
        self.task = self.DEFAULT_TASK
        self.size = "r101"

    @classmethod
    def get_download_url(cls, filename):
        """Return the immutable byte-identical checkpoint mirror URL."""
        if filename not in (None, WEIGHT_FILE):
            return None
        return f"https://huggingface.co/{HF_REPO}/resolve/{HF_REVISION}/{WEIGHT_FILE}"

    @classmethod
    def get_download_notice(cls, filename, url):
        """Announce the non-commercial terms before an automatic download."""
        return (
            "FCOS3D weights are trained on nuScenes, whose terms are "
            "non-commercial, and are not covered by LibreYOLO's MIT license. "
            f"See {TERMS_URL}"
        )

    @classmethod
    def _resolve_checkpoint(cls, model_path):
        if model_path is not None:
            candidate = Path(model_path).expanduser()
            if candidate.is_file():
                return candidate.resolve()
            if str(model_path) != WEIGHT_FILE:
                raise FileNotFoundError(
                    f"FCOS3D requires a local official R101 nuScenes checkpoint: {candidate}"
                )
        return cls._download_checkpoint()

    @classmethod
    def _download_checkpoint(cls):
        logger.warning(
            "FCOS3D weights are trained on nuScenes under non-commercial terms "
            "(%s) and are not covered by LibreYOLO's MIT license.",
            TERMS_URL,
        )
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise ImportError(
                "FCOS3D automatic weights require huggingface_hub. Install "
                "with: pip install huggingface_hub"
            ) from exc
        path = Path(
            hf_hub_download(
                repo_id=HF_REPO,
                filename=WEIGHT_FILE,
                revision=HF_REVISION,
            )
        )
        cls._verify_mirrored_checkpoint(path)
        return path.resolve()

    @classmethod
    def _verify_mirrored_checkpoint(cls, path):
        digest = hashlib.sha256()
        with Path(path).open("rb") as checkpoint:
            for chunk in iter(lambda: checkpoint.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != WEIGHT_SHA256:
            raise ValueError(
                "Downloaded FCOS3D checkpoint failed its pinned SHA-256 check."
            )

    @staticmethod
    def _threshold(value, name):
        if isinstance(value, bool) or not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be a finite number in [0, 1].")
        return float(value)

    def predict(
        self,
        source,
        *,
        intrinsics,
        conf=DEFAULT_CONF,
        iou=DEFAULT_IOU,
        max_det=200,
        stream=False,
        save=False,
        output_path=None,
        color_format="auto",
    ):
        """Predict metric cuboids with known original-image camera calibration."""
        k = calibration(intrinsics)
        conf = self._threshold(conf, "conf")
        iou = self._threshold(iou, "iou")
        if isinstance(max_det, bool) or not isinstance(max_det, int) or max_det <= 0:
            raise ValueError("max_det must be a positive integer.")
        many = isinstance(source, (list, tuple))
        if isinstance(source, (str, Path)) and Path(source).is_dir():
            source = sorted(
                p
                for p in Path(source).iterdir()
                if p.suffix.lower() in SUPPORTED_EXTENSIONS
            )
            many = True
        sources = list(source) if many else [source]
        if output_path is not None and (not save or len(sources) != 1):
            raise ValueError("output_path requires save=True with a single image.")

        def generate():
            save_dir = None
            for index, item in enumerate(sources):
                start = time.perf_counter()
                image = ImageLoader.load(item, color_format=color_format)
                shape = (image.height, image.width)
                tensor = preprocess(image).to(self.device)
                ready = time.perf_counter()
                with torch.inference_mode():
                    raw = self.model(tensor)
                    if self.device.type == "cuda":
                        torch.cuda.synchronize(self.device)
                    inferred = time.perf_counter()
                    boxes, scores, labels = decode(raw, k, conf, iou, max_det)
                boxes2d, boxes3d = payloads(boxes, scores, labels, k, shape)
                result = Results(
                    boxes2d,
                    shape,
                    path=str(item) if isinstance(item, (str, Path)) else None,
                    names=self.names.copy(),
                    boxes3d=boxes3d,
                    speed={
                        "preprocess": (ready - start) * 1000,
                        "inference": (inferred - ready) * 1000,
                        "postprocess": (time.perf_counter() - inferred) * 1000,
                    },
                )
                if save:
                    if output_path:
                        target = Path(output_path)
                    else:
                        if save_dir is None:
                            from ...utils.general import increment_path

                            save_dir = increment_path(
                                Path("runs/detect3d/predict"), mkdir=True
                            )
                        target = save_dir / f"image_{index}.png"
                    target.parent.mkdir(parents=True, exist_ok=True)
                    result.plot(image).save(target)
                yield result

        output = generate()
        return output if stream else (list(output) if many else next(output))

    __call__ = predict

    def train(self, *args, **kwargs):
        raise NotImplementedError(
            "FCOS3D training requires a dedicated 3D dataset and trainer."
        )

    def val(self, *args, **kwargs):
        raise NotImplementedError(
            "Use the upstream nuScenes evaluator for FCOS3D accuracy; 2D mAP is not 3D evaluation."
        )

    def export(self, *args, **kwargs):
        raise NotImplementedError(
            "FCOS3D export has not been implemented or validated."
        )

    def track(self, *args, **kwargs):
        raise NotImplementedError("FCOS3D tracking is not implemented.")
