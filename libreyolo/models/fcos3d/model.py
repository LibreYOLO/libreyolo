"""Native FCOS3D inference with camera calibration and standard Results."""

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


class LibreFCOS3D:
    """FCOS3D R101-DCN for the ten nuScenes classes, inference-only.

    Supply a local official checkpoint and original-image pinhole intrinsics.
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

    def __init__(self, model_path, *, device="auto"):
        self.model_path = Path(model_path).expanduser()
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"FCOS3D requires a local official R101 nuScenes checkpoint: {self.model_path}"
            )
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
            raise TypeError(
                "Expected an official FCOS3D checkpoint with a state_dict."
            )
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
        """No hosted checkpoint until redistribution terms are established."""
        return None

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
