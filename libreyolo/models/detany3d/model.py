"""DetAny3D integration through a separately installed upstream runtime.

The runtime contains UniDepth under CC BY-NC 4.0. That implementation is not
bundled in LibreYOLO. This adapter uses the upstream Apache-2.0 inference
interface and converts its outputs to the existing camera-frame contract.
"""

import hashlib
import logging
import os
from pathlib import Path
from typing import ClassVar

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from ...utils.image_loader import SUPPORTED_EXTENSIONS, ImageLoader
from ...utils.results import Boxes, Boxes3D, Results
from .runtime import RuntimeWorker

logger = logging.getLogger(__name__)
HF_REPO = "LibreYOLO/LibreDetAny3D"
HF_REVISION = "eeb89d1b1d37ec5a361b1ec79b26ea0620d9f4c3"
WEIGHT_FILE = "detany3d.pth"
WEIGHT_SHA256 = "cd5f737ddbf3ceb64f969141672d04c0a3d785ad3b8a72668619a530840cd217"


class LibreDetAny3D:
    """Box-, point- and text-prompted 3D detection with predicted calibration.

    ``model_path`` accepts an unchanged official full checkpoint. Omitting it
    downloads LibreYOLO's byte-identical, revision-pinned mirror. Upstream
    documents its depth branch as initialized from UniDepth v2 under
    CC BY-NC 4.0, so the mirrored weights are non-commercial and are not
    covered by LibreYOLO's MIT license.
    Supply the separately installed runtime.
    Point arrays (N,2) describe one object; (G,N,2) describes G objects. All
    points are positive prompts. Text can be combined with box prompts, while
    point prompts are exclusive. conf/text_threshold configure the text detector;
    geometric prompts retain one prediction per prompt group.
    """

    FAMILY = "detany3d"
    CLI_COMMAND = "detany3d"
    FILENAME_PREFIX = "LibreDetAny3D"
    SUPPORTED_TASKS = ("detect3d",)
    DEFAULT_TASK = "detect3d"
    INPUT_SIZES: ClassVar[dict[str, int]] = {"h": 896}
    TASK_INPUT_SIZES: ClassVar[dict] = {}
    DEFAULT_CONF = 0.37
    DEFAULT_TEXT_THRESHOLD = 0.25

    def __init__(
        self,
        model_path=None,
        *,
        runtime_path=None,
        runtime_python=None,
        device="auto",
        conf=DEFAULT_CONF,
        text_threshold=DEFAULT_TEXT_THRESHOLD,
        grounding_checkpoint=None,
        grounding_config=None,
    ):
        checkpoint = self._resolve_checkpoint(model_path)
        runtime_path = runtime_path or os.environ.get("DETANY3D_PATH")
        if not runtime_path:
            raise ImportError(
                "Pass runtime_path= or set DETANY3D_PATH to a separately installed DetAny3D checkout."
            )
        root = Path(runtime_path).expanduser().resolve()
        if not (root / "wrap_model.py").is_file():
            raise FileNotFoundError(f"DetAny3D runtime has no wrap_model.py: {root}")
        if isinstance(device, int) or str(device).isdigit():
            device = f"cuda:{device}"
        if device != "auto" and torch.device(device).type not in {"cpu", "cuda"}:
            raise ValueError(
                "DetAny3D supports CPU or CUDA; MPS has not been validated."
            )
        for name, value in (("conf", conf), ("text_threshold", text_threshold)):
            if isinstance(value, bool) or not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1].")
        self.model_path = checkpoint
        self._root = root
        self._python = runtime_python or os.environ.get("DETANY3D_PYTHON")
        self._config = {
            "checkpoint": str(checkpoint),
            "runtime_path": str(root),
            "device": str(device),
            "conf": float(conf),
            "text_threshold": float(text_threshold),
            "grounding_checkpoint": str(
                Path(grounding_checkpoint).expanduser().resolve()
            )
            if grounding_checkpoint
            else None,
            "grounding_config": str(Path(grounding_config).expanduser().resolve())
            if grounding_config
            else None,
        }
        self._backend = None
        self._classes = None
        self.names = {}
        self._label_ids = {}
        self.task = "detect3d"
        self.size = "h"
        self._ensure_backend()

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
            "DetAny3D weights are non-commercial: upstream documents the depth "
            "branch as initialized from UniDepth v2 under CC BY-NC 4.0. They "
            "are not covered by LibreYOLO's MIT license."
        )

    @classmethod
    def _resolve_checkpoint(cls, model_path):
        if model_path is not None:
            candidate = Path(model_path).expanduser()
            if candidate.is_file():
                return candidate.resolve()
            if str(model_path) != WEIGHT_FILE:
                raise FileNotFoundError(
                    f"DetAny3D checkpoint not found: {candidate.resolve()}"
                )
        return cls._download_checkpoint()

    @classmethod
    def _download_checkpoint(cls):
        logger.warning(
            "DetAny3D weights are non-commercial (UniDepth v2 depth lineage, "
            "CC BY-NC 4.0) and are not covered by LibreYOLO's MIT license."
        )
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise ImportError(
                "DetAny3D automatic weights require huggingface_hub. Install "
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
                "Downloaded DetAny3D checkpoint failed its pinned SHA-256 check."
            )

    def _ensure_backend(self):
        if self._backend is None or self._backend.closed:
            self._backend = RuntimeWorker(
                config=self._config,
                runtime_path=self._root,
                runtime_python=self._python,
            )
            self.device = torch.device(self._backend.device)
        return self._backend

    @staticmethod
    def _text(value):
        if isinstance(value, str):
            value = [value]
        if (
            not isinstance(value, (list, tuple))
            or not value
            or any(not isinstance(x, str) or not x.strip() for x in value)
        ):
            raise ValueError(
                "text must be a nonempty string or list of nonempty strings."
            )
        return [x.strip() for x in value]

    def _register_labels(self, labels):
        for label in labels:
            key = label.lower()
            if key not in self._label_ids:
                index = len(self.names)
                self._label_ids[key] = index
                self.names[index] = key

    def set_classes(self, names):
        self._classes = self._text(names)
        self.names = {}
        self._label_ids = {}
        self._register_labels(self._classes)
        return self

    def predict(
        self,
        source,
        *,
        text=None,
        bboxes=None,
        points=None,
        stream=False,
        save=False,
        output_path=None,
        color_format="auto",
    ):
        if text is None and bboxes is None and points is None:
            text = self._classes
        if text is None and bboxes is None and points is None:
            raise ValueError("Supply text, bboxes, or points.")
        if points is not None and (bboxes is not None or text is not None):
            raise ValueError("Point prompts cannot be combined with boxes or text.")
        prompt = {}
        if text is not None:
            prompt["text"] = self._text(text)
            self._register_labels(prompt["text"])
        if bboxes is not None:
            boxes = np.asarray(bboxes, np.float32)
            if boxes.ndim == 1:
                boxes = boxes[None]
            if (
                boxes.ndim != 2
                or boxes.shape[1] != 4
                or not len(boxes)
                or not np.isfinite(boxes).all()
                or (boxes[:, 2:] <= boxes[:, :2]).any()
            ):
                raise ValueError("bboxes must be a nonempty finite (N,4) xyxy array.")
            prompt["bboxes"] = boxes.tolist()
        if points is not None:
            xy = np.asarray(points, np.float32)
            if xy.ndim == 1:
                xy = xy[None]
            if xy.ndim == 2:
                xy = xy[None]
            if (
                xy.ndim != 3
                or xy.shape[2] != 2
                or not xy.shape[0]
                or not xy.shape[1]
                or not np.isfinite(xy).all()
            ):
                raise ValueError(
                    "points must be finite (N,2) or (G,N,2) pixel coordinates."
                )
            prompt["points"] = xy.tolist()
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
            raise ValueError("output_path requires save=True with one image.")

        def generate():
            save_dir = None
            for index, item in enumerate(sources):
                image = ImageLoader.load(item, color_format=color_format)
                try:
                    arrays, labels = self._ensure_backend().predict(
                        np.asarray(image, dtype=np.uint8), prompt
                    )
                    self._register_labels(labels)
                    result = self._result(
                        arrays,
                        labels,
                        (image.height, image.width),
                        str(item) if isinstance(item, (str, Path)) else None,
                        names=self.names.copy(),
                    )
                except Exception:
                    self.close()
                    raise
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

        iterator = generate()
        return iterator if stream else (list(iterator) if many else next(iterator))

    __call__ = predict

    @staticmethod
    def _result(arrays, labels, shape, path, names=None):
        centers, dims, rotations = (
            arrays["centers"],
            arrays["dimensions"],
            arrays["rotations"],
        )
        n = len(centers)
        if (
            centers.shape != (n, 3)
            or dims.shape != (n, 3)
            or rotations.shape != (n, 3, 3)
            or len(labels) != n
        ):
            raise ValueError("DetAny3D returned misaligned geometry.")
        if names is None:
            names = dict(enumerate(dict.fromkeys(label.lower() for label in labels)))
        lookup = {label.lower(): index for index, label in names.items()}
        ids = np.array([lookup[label.lower()] for label in labels], np.int64)
        scores = arrays["scores"].reshape(-1)
        if len(scores) != n or arrays["boxes"].shape != (n, 4):
            raise ValueError("DetAny3D returned misaligned boxes or scores.")
        transform = arrays["view_to_original"]
        k = transform @ arrays["intrinsics"]
        xywh = arrays["boxes"]
        xyxy = np.concatenate(
            (xywh[:, :2] - xywh[:, 2:] / 2, xywh[:, :2] + xywh[:, 2:] / 2), 1
        )
        xyxy[:, [0, 2]] = xyxy[:, [0, 2]] * transform[0, 0] + transform[0, 2]
        xyxy[:, [1, 3]] = xyxy[:, [1, 3]] * transform[1, 1] + transform[1, 2]
        data = np.zeros((n, 14), np.float32)
        data[:, :3] = centers
        data[:, 3:6] = dims[:, [0, 2, 1]]  # upstream w,h,l -> canonical w,l,h
        if n:
            # Canonical local +x is length; upstream local +z is length.
            basis = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], np.float32)
            q = Rotation.from_matrix(rotations @ basis).as_quat()
            data[:, 6:10] = q[:, [3, 0, 1, 2]]
        data[:, 10] = scores
        data[:, 11] = ids
        data[:, 12] = scores
        data[:, 13] = (
            1.0  # upstream exports the input detection score, not box-IoU logits
        )
        return Results(
            Boxes(xyxy, scores, ids, orig_shape=shape),
            shape,
            path=path,
            names=names,
            boxes3d=Boxes3D(data, shape, k),
        )

    def close(self):
        if self._backend is not None:
            self._backend.close()
            self._backend = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def train(self, *args, **kwargs):
        raise NotImplementedError("DetAny3D training is not integrated.")

    def val(self, *args, **kwargs):
        raise NotImplementedError("Use upstream evaluation for DetAny3D 3D accuracy.")

    def export(self, *args, **kwargs):
        raise NotImplementedError("DetAny3D export is not integrated.")

    def track(self, *args, **kwargs):
        raise NotImplementedError("DetAny3D tracking is not integrated.")
