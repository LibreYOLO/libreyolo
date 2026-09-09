"""Independent MIT adapter using WildDet3D's documented public interface.

No upstream implementation is bundled or derived here. Install the separately
licensed runtime using its installation guide. CUDA can expose it on
``PYTHONPATH``; macOS passes its checkout and interpreter through
``runtime_path`` and ``runtime_python``. This sibling API uses upstream
checkpoints unchanged, outside LibreYOLO's state-dict factory. See
docs/adr/0021-detect3d-task-contract.md.
"""

from __future__ import annotations

import importlib
import logging
import os
from pathlib import Path
from typing import ClassVar

import numpy as np
import torch

from ...utils.image_loader import SUPPORTED_EXTENSIONS, ImageLoader
from ...utils.results import Boxes, Boxes3D, Results

logger = logging.getLogger(__name__)
UPSTREAM_REVISION = "1b8aa52b6ff3f00d0ebfa07175efc0c0c440964a"
INSTALL_URL = "https://github.com/allenai/WildDet3D#installation"


class LibreWildDet3D:
    """Promptable camera-frame 3D detection through an optional upstream runtime.

    ``model_path`` is a user-supplied upstream full checkpoint, unchanged.
    ``intrinsics`` in predict is required: original-image pixel calibration.
    Inputs and outputs use original-image pixels; 3D geometry uses metres.
    A single source returns Results, a list/directory returns a list, and
    stream=True yields Results. Shared prompts/calibration apply to every
    image in a multi-source call. On macOS, ``device="auto"`` selects the
    isolated CPU worker. MPS is rejected because the upstream image path mixes
    CPU and MPS tensors. Training, tracking and export are deferred.
    """

    FAMILY = "wilddet3d"
    CLI_COMMAND = "wilddet3d"
    FILENAME_PREFIX = "LibreWildDet3D"
    SUPPORTED_TASKS = ("detect3d",)
    DEFAULT_TASK = "detect3d"
    INPUT_SIZES: ClassVar[dict[str, int]] = {"l": 1008}
    TASK_INPUT_SIZES: ClassVar[dict[str, dict[str, int]]] = {}
    DEFAULT_CONF = 0.3
    DEFAULT_CONF3D = 0.1
    DEFAULT_IOU = 0.6

    def __init__(
        self,
        model_path,
        *,
        device="auto",
        conf=DEFAULT_CONF,
        conf3d=DEFAULT_CONF3D,
        iou=DEFAULT_IOU,
        use_depth=False,
        runtime_path=None,
        runtime_python=None,
    ):
        self.model_path = Path(model_path).expanduser().resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"WildDet3D checkpoint not found: {self.model_path}"
            )
        if isinstance(device, int) or str(device).isdigit():
            device = f"cuda:{device}"
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if self.device.type == "mps":
            raise ValueError(
                "WildDet3D does not support MPS: its upstream image path mixes "
                "CPU and MPS tensors. Use device='cpu' on macOS."
            )
        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError("WildDet3D supports cpu or cuda devices.")
        self.conf = self._threshold(conf, "conf")
        self.conf3d = self._threshold(conf3d, "conf3d")
        self.iou = self._threshold(iou, "iou")
        if not isinstance(use_depth, bool):
            raise TypeError("use_depth must be a boolean.")
        self.use_depth = use_depth
        self.task = self.DEFAULT_TASK
        self._runtime = None
        self._predictor = None
        self._backend = None
        self._runtime_path = runtime_path or os.environ.get("WILDDET3D_PATH")
        self._runtime_python = runtime_python or os.environ.get("WILDDET3D_PYTHON")
        if self._runtime_path:
            self._runtime_path = Path(self._runtime_path).expanduser().resolve()
            if not self._runtime_path.is_dir():
                raise FileNotFoundError(
                    f"WildDet3D runtime checkout not found: {self._runtime_path}"
                )
        if self._runtime_python:
            # Do not resolve this path: virtualenv interpreters are symlinks to
            # the base Python, and resolving one silently drops the venv.
            self._runtime_python = Path(self._runtime_python).expanduser().absolute()
            if not self._runtime_python.is_file():
                raise FileNotFoundError(
                    f"WildDet3D runtime interpreter not found: {self._runtime_python}"
                )
        self._classes = None

    @staticmethod
    def _threshold(value, name):
        value = float(value)
        if not np.isfinite(value) or value < 0 or (name != "conf" and value > 1):
            bounds = "nonnegative" if name == "conf" else "in [0, 1]"
            raise ValueError(f"{name} must be finite and {bounds}.")
        return value

    @staticmethod
    def _texts(text):
        values = [text] if isinstance(text, str) else list(text)
        if not values or any(not isinstance(v, str) or not v.strip() for v in values):
            raise ValueError("text must contain nonempty category names.")
        return [v.strip() for v in values]

    @classmethod
    def get_download_url(cls, filename):
        """Upstream checkpoints are user supplied; no implicit download route."""
        return

    def set_classes(self, names):
        """Set the vocabulary used when predict omits text and geometric prompts."""
        self._classes = self._texts(names)
        return self

    def _load(self):
        if self._predictor is not None or self._backend is not None:
            return
        if self.device.type != "cuda" or self._runtime_path or self._runtime_python:
            from .runtime import RuntimeWorker

            logger.warning(
                "WildDet3D runtime and weights retain their upstream SAM License "
                "terms; they are not covered by LibreYOLO's MIT license."
            )
            self._backend = RuntimeWorker(
                config=self._runtime_config(),
                runtime_path=self._runtime_path,
                runtime_python=self._runtime_python,
            )
            return
        if not torch.cuda.is_available():
            raise RuntimeError("The selected CUDA device is not available.")
        try:
            runtime = importlib.import_module("wilddet3d")
        except ImportError as exc:
            raise ImportError(
                "Install the optional WildDet3D runtime and its dependencies using "
                f"{INSTALL_URL}, then add its checkout to PYTHONPATH. "
                f"Documented API revision: {UPSTREAM_REVISION}."
            ) from exc
        if not all(
            callable(getattr(runtime, name, None))
            for name in ("build_model", "preprocess")
        ):
            raise RuntimeError(
                f"WildDet3D runtime lacks its documented public API ({UPSTREAM_REVISION})."
            )
        logger.warning(
            "WildDet3D runtime and weights retain their upstream SAM "
            "License terms; they are not covered by LibreYOLO's MIT license."
        )
        predictor = runtime.build_model(**self._runtime_config())
        if not callable(predictor):
            raise TypeError(
                "WildDet3D build_model did not return a callable predictor."
            )
        self._runtime, self._predictor = runtime, predictor

    def _runtime_config(self):
        return {
            "checkpoint": str(self.model_path),
            "device": str(self.device),
            "skip_pretrained": True,
            "score_threshold": self.conf,
            "score_3d_threshold": self.conf3d,
            "iou_threshold": self.iou,
            "use_depth_input_test": self.use_depth,
            "use_predicted_intrinsics": False,
        }

    def close(self):
        """Release the cached runtime; a later predict call loads it again."""
        if getattr(self, "_backend", None) is not None:
            self._backend.close()
            self._backend = None
        self._predictor = None
        self._runtime = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def predict(
        self,
        source,
        *,
        intrinsics,
        text=None,
        bboxes=None,
        points=None,
        labels=None,
        prompt_mode="geometric",
        depth=None,
        stream=False,
        save=False,
        output_path=None,
        color_format="auto",
    ):
        """Detect with text, xyxy boxes, or grouped xy points plus binary labels.

        Geometric prompts lift each box/point group to one cuboid. Visual
        prompts find similar objects. These modes retain upstream score
        semantics: geometric prompts are not confidence-filtered. Confidence
        and IoU thresholds are constructor arguments, matching upstream.
        ``depth`` is an original-resolution (H,W) metric array and requires
        use_depth=True at construction. All images in a list share this input.
        """
        k = np.array(intrinsics, dtype=np.float32, copy=True)
        # Reuse the public calibration validator without loading the model.
        Boxes3D(np.empty((0, 14), dtype=np.float32), intrinsics=k)
        if prompt_mode not in ("geometric", "visual"):
            raise ValueError("prompt_mode must be 'geometric' or 'visual'.")
        if text is None and bboxes is None and points is None:
            text = self._classes
        if sum(v is not None for v in (text, bboxes, points)) != 1:
            raise ValueError("Supply exactly one of text, bboxes, or points.")
        if labels is not None and points is None:
            raise ValueError("labels requires points.")
        if text is not None:
            if prompt_mode != "geometric":
                raise ValueError("prompt_mode applies only to box or point prompts.")
            names = self._texts(text)
            prompt = {"input_texts": names}
        elif bboxes is not None:
            boxes = np.asarray(bboxes, dtype=np.float32)
            if boxes.ndim == 1:
                boxes = boxes[None]
            if (
                boxes.ndim != 2
                or boxes.shape[1] != 4
                or not len(boxes)
                or not np.isfinite(boxes).all()
                or (boxes[:, 2:] <= boxes[:, :2]).any()
            ):
                raise ValueError("bboxes must be nonempty finite (N, 4) xyxy boxes.")
            names = [f"prompt_{i}" for i in range(len(boxes))]
            prompt = {"input_boxes": boxes.tolist(), "prompt_text": prompt_mode}
        else:
            xy = np.asarray(points, dtype=np.float32)
            if xy.ndim == 2:
                xy = xy[None]
            if (
                xy.ndim != 3
                or xy.shape[2] != 2
                or not xy.shape[0]
                or not xy.shape[1]
                or not np.isfinite(xy).all()
            ):
                raise ValueError("points must be finite (N, 2) or (groups, N, 2).")
            lab = np.ones(xy.shape[:2]) if labels is None else np.asarray(labels)
            if lab.ndim == 1:
                lab = lab[None]
            if lab.shape != xy.shape[:2] or not np.isin(lab, [0, 1]).all():
                raise ValueError("labels must match point groups and contain only 0/1.")
            if not (lab == 1).any(axis=1).all():
                raise ValueError("Each point group needs a positive point.")
            names = [f"prompt_{i}" for i in range(len(xy))]
            prompt = {
                "input_points": np.concatenate((xy, lab[..., None]), axis=2).tolist(),
                "prompt_text": prompt_mode,
            }
        if self.use_depth != (depth is not None):
            raise ValueError("depth must be supplied exactly when use_depth=True.")
        is_many = isinstance(source, (list, tuple))
        if isinstance(source, (str, Path)) and Path(source).is_dir():
            source = sorted(
                p
                for p in Path(source).iterdir()
                if p.suffix.lower() in SUPPORTED_EXTENSIONS
            )
            is_many = True
        sources = list(source) if is_many else [source]
        if output_path is not None and not save:
            raise ValueError("output_path requires save=True.")
        if save and len(sources) > 1 and output_path is not None:
            raise ValueError("An explicit output_path supports a single image only.")

        def generate():
            save_dir = None
            for index, item in enumerate(sources):
                image = ImageLoader.load(item, color_format=color_format)
                h, w = image.height, image.width
                depth_array = None
                if depth is not None:
                    depth_array = np.asarray(depth, dtype=np.float32)
                    if (
                        depth_array.shape != (h, w)
                        or not np.isfinite(depth_array).all()
                        or (depth_array < 0).any()
                    ):
                        raise ValueError(
                            "depth must be finite nonnegative (H, W) metres."
                        )
                self._load()
                if self._backend is not None:
                    outputs = self._backend.predict(
                        np.asarray(image), k, prompt, depth=depth_array
                    )
                else:
                    data = self._runtime.preprocess(
                        np.asarray(image, dtype=np.float32), k, depth=depth_array
                    )
                    call = dict(
                        images=data["images"].to(self.device),
                        intrinsics=data["intrinsics"].to(self.device)[None],
                        input_hw=[data["input_hw"]],
                        original_hw=[data["original_hw"]],
                        padding=[data["padding"]],
                        **prompt,
                    )
                    if depth_array is not None:
                        call["depth_gt"] = data["depth_gt"].to(self.device)
                    with torch.inference_mode():
                        outputs = self._predictor(**call)
                result = self._result(
                    outputs,
                    (h, w),
                    k,
                    names,
                    str(item) if isinstance(item, (str, Path)) else None,
                )
                if save:
                    if output_path is None:
                        if save_dir is None:
                            from ...utils.general import increment_path

                            save_dir = increment_path(
                                Path("runs/detect3d/predict"), mkdir=True
                            )
                        destination = save_dir / f"image_{index}.png"
                    else:
                        destination = Path(output_path)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    result.plot(image).save(destination)
                yield result

        if stream:
            return generate()
        results = list(generate())
        return results if is_many else results[0]

    __call__ = predict

    @staticmethod
    def _result(outputs, shape, intrinsics, names, path):
        if not isinstance(outputs, (tuple, list)) or len(outputs) != 7:
            raise ValueError("Unexpected WildDet3D output: expected seven fields.")
        fields = outputs[:6]
        if any(not isinstance(f, (tuple, list)) for f in fields):
            raise ValueError("WildDet3D output fields must be per-prompt lists.")
        if len({len(f) for f in fields}) != 1:
            raise ValueError("WildDet3D output lists have different lengths.")
        boxes_rows, cuboid_rows = [], []
        for group in zip(*fields):
            boxes, cuboids, scores, scores2d, scores3d, ids = [
                torch.as_tensor(value).detach().cpu().float() for value in group
            ]
            n = len(boxes)
            if boxes.shape != (n, 4) or cuboids.shape != (n, 10):
                raise ValueError("WildDet3D returned malformed box geometry.")
            if any(v.shape != (n,) for v in (scores, scores2d, scores3d, ids)):
                raise ValueError("WildDet3D returned misaligned scores/classes.")
            if not torch.isfinite(boxes).all() or (boxes[:, 2:] < boxes[:, :2]).any():
                raise ValueError("WildDet3D returned invalid 2D boxes.")
            if (ids >= len(names)).any():
                raise ValueError(
                    "WildDet3D returned class ids outside the prompt vocabulary."
                )
            boxes_rows.append(torch.cat((boxes, scores[:, None], ids[:, None]), dim=1))
            cuboid_rows.append(
                torch.cat(
                    (
                        cuboids,
                        scores[:, None],
                        ids[:, None],
                        scores2d[:, None],
                        scores3d[:, None],
                    ),
                    dim=1,
                )
            )
        boxes = torch.cat(boxes_rows) if boxes_rows else torch.empty((0, 6))
        cuboids = torch.cat(cuboid_rows) if cuboid_rows else torch.empty((0, 14))
        return Results(
            Boxes(boxes[:, :4], boxes[:, 4], boxes[:, 5], orig_shape=shape),
            shape,
            path=path,
            names=dict(enumerate(names)),
            boxes3d=Boxes3D(cuboids, shape, torch.as_tensor(intrinsics).clone()),
        )

    def train(self, *args, **kwargs):
        raise NotImplementedError(
            "WildDet3D training is not integrated; use the upstream recipe."
        )

    def val(self, *args, **kwargs):
        raise NotImplementedError(
            "WildDet3D validation requires the upstream 3D benchmark evaluator."
        )

    def export(self, *args, **kwargs):
        raise NotImplementedError(
            "WildDet3D export is not supported by this optional adapter."
        )

    def track(self, *args, **kwargs):
        raise NotImplementedError("WildDet3D tracking is not integrated.")
