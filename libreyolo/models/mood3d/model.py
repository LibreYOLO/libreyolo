"""3D-MOOD open-set monocular 3D detection adapter.

The Apache-2.0 upstream runtime stays separately installed. LibreYOLO calls
its public model components at pinned revision 41bb290 and does not bundle the
runtime or its undeclared-license CUDA extension. See the family NOTICE and
ADR 0021.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import ClassVar

import numpy as np
import torch

from ...utils.image_loader import SUPPORTED_EXTENSIONS, ImageLoader
from ...utils.results import Boxes, Boxes3D, DepthMap, Results

logger = logging.getLogger(__name__)
UPSTREAM_REPO = "cvg/3D-MOOD"
UPSTREAM_REVISION = "41bb2904932d91507338e75ae4c802d67616ca1b"
INSTALL_URL = "https://github.com/cvg/3D-MOOD#installation"
UPSTREAM_HF_REPO = "RoyYang0714/3D-MOOD"
UPSTREAM_HF_REVISION = "3d1fab552189f1a62fdb60ebb606d20625a30b90"
WEIGHTS = {
    "t": {
        "repo_revision": "f1e163b249b43a8bb7b8b4ef4f446828270e35ef",
        "filename": "Libre3DMOODt.pt",
        "upstream_filename": "gdino3d_swin-t_120e_omni3d_699f69.pt",
        "sha256": "699f69454625d1fb3ad7d3669d8d662d52c349c99b2aba28da1215b8e7e4a555",
    },
    "b": {
        "repo_revision": "a4f6115d285a6983439f0e54f20c44b0ce87c261",
        "filename": "Libre3DMOODb.pt",
        "upstream_filename": "gdino3d_swin-b_120e_omni3d_834c97.pt",
        "sha256": "834c976df385610bba105c7d4ea3e2d7b57dabd69dee2b61397ae398de67a677",
    },
}


class Libre3DMOOD:
    """Text-conditioned camera-frame 3D detection with 3D-MOOD.

    The official Swin-T and Swin-B checkpoints are used unchanged. Known
    original-image camera intrinsics are required. Predictions include aligned
    2D boxes, metric 3D cuboids, and the model's metric depth map. A single
    source returns ``Results``; multiple sources return a list; ``stream=True``
    returns a generator.

    3D-MOOD exposes one language-conditioned detection score and no independent
    3D confidence. ``Boxes3D.conf`` and ``conf2d`` contain that score;
    ``conf3d`` is the neutral value 1.0.
    """

    FAMILY = "3dmood"
    CLI_COMMAND = "3dmood"
    FILENAME_PREFIX = "Libre3DMOOD"
    SUPPORTED_TASKS = ("detect3d",)
    DEFAULT_TASK = "detect3d"
    INPUT_SIZES: ClassVar[dict[str, tuple[int, int]]] = {
        "t": (800, 1333),
        "b": (800, 1333),
    }
    TASK_INPUT_SIZES: ClassVar[dict[str, dict[str, int]]] = {}
    DEFAULT_CONF = 0.1
    DEFAULT_IOU = 0.5
    DEFAULT_MAX_DET = 100

    def __init__(
        self,
        model_path=None,
        size=None,
        *,
        device="auto",
        conf=DEFAULT_CONF,
        iou=DEFAULT_IOU,
        max_det=DEFAULT_MAX_DET,
        runtime_path=None,
        runtime_python=None,
    ):
        size = self._resolve_size(model_path, size)
        if size not in WEIGHTS:
            raise ValueError("3D-MOOD size must be 't' or 'b'.")
        self.size = size
        self.model_path = self._resolve_checkpoint(model_path, size)
        if isinstance(device, int) or str(device).isdigit():
            device = f"cuda:{device}"
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)
        if self.device.type not in {"cpu", "cuda", "mps"}:
            raise ValueError("3D-MOOD supports cpu, mps, or cuda devices.")
        self.conf = self._probability(conf, "conf")
        self.iou = self._probability(iou, "iou")
        if isinstance(max_det, bool) or int(max_det) != max_det or int(max_det) <= 0:
            raise ValueError("max_det must be a positive integer.")
        self.max_det = int(max_det)
        self.task = self.DEFAULT_TASK
        self._runtime_path = runtime_path or os.environ.get("MOOD3D_PATH")
        self._runtime_python = runtime_python or os.environ.get("MOOD3D_PYTHON")
        if self._runtime_path:
            self._runtime_path = Path(self._runtime_path).expanduser().resolve()
            if not self._runtime_path.is_dir():
                raise FileNotFoundError(
                    f"3D-MOOD runtime checkout not found: {self._runtime_path}"
                )
        if self._runtime_python:
            self._runtime_python = Path(self._runtime_python).expanduser().absolute()
            if not self._runtime_python.is_file():
                raise FileNotFoundError(
                    f"3D-MOOD runtime interpreter not found: {self._runtime_python}"
                )
        self._backend = None
        self._classes = None

    @staticmethod
    def _resolve_size(model_path, size):
        derived = None
        if model_path is not None:
            name = Path(str(model_path)).name
            for candidate, weight in WEIGHTS.items():
                if name in {weight["filename"], weight["upstream_filename"]}:
                    derived = candidate
                    break
        if size is None:
            return derived or "t"
        size = str(size).lower()
        if size not in WEIGHTS:
            raise ValueError("3D-MOOD size must be 't' or 'b'.")
        if derived is not None and derived != size:
            raise ValueError(
                f"3D-MOOD checkpoint {Path(str(model_path)).name!r} is size "
                f"{derived!r}, not {size!r}."
            )
        return size

    @staticmethod
    def _probability(value, name):
        value = float(value)
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be finite and in [0, 1].")
        return value

    @staticmethod
    def _texts(text):
        values = [text] if isinstance(text, str) else list(text)
        if not values or any(not isinstance(v, str) or not v.strip() for v in values):
            raise ValueError("text must contain nonempty category names.")
        return [v.strip() for v in values]

    @classmethod
    def get_download_url(cls, filename):
        for weight in WEIGHTS.values():
            if filename == weight["filename"]:
                return (
                    f"https://huggingface.co/LibreYOLO/{weight['filename'][:-3]}"
                    f"/resolve/{weight['repo_revision']}/{weight['filename']}"
                )
        return None

    @classmethod
    def _resolve_checkpoint(cls, model_path, size):
        expected = WEIGHTS[size]["filename"]
        if model_path is not None:
            candidate = Path(model_path).expanduser()
            if candidate.is_file():
                return candidate.resolve()
            if str(model_path) != expected:
                raise FileNotFoundError(f"3D-MOOD checkpoint not found: {candidate}")
        return cls._download_checkpoint(size)

    @classmethod
    def _download_checkpoint(cls, size):
        weight = WEIGHTS[size]
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise ImportError(
                "3D-MOOD automatic weights require huggingface_hub. Install "
                "with: pip install huggingface_hub"
            ) from exc
        path = Path(
            hf_hub_download(
                repo_id=f"LibreYOLO/{weight['filename'][:-3]}",
                filename=weight["filename"],
                revision=weight["repo_revision"],
            )
        )
        cls._verify_checkpoint(path, size)
        return path.resolve()

    @classmethod
    def _verify_checkpoint(cls, path, size):
        expected = WEIGHTS[size]["sha256"]
        if expected is None:
            raise RuntimeError(f"3D-MOOD {size} checkpoint digest is not configured.")
        digest = hashlib.sha256()
        with Path(path).open("rb") as checkpoint:
            for chunk in iter(lambda: checkpoint.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise ValueError(
                f"Downloaded 3D-MOOD {size} checkpoint failed its pinned SHA-256 check."
            )

    def set_classes(self, names):
        self._classes = self._texts(names)
        return self

    def _load(self):
        if self._backend is not None:
            return
        from .runtime import RuntimeWorker

        try:
            self._backend = RuntimeWorker(
                config={
                    "checkpoint": str(self.model_path),
                    "size": self.size,
                    "device": str(self.device),
                    "conf": self.conf,
                    "iou": self.iou,
                    "max_det": self.max_det,
                },
                runtime_path=self._runtime_path,
                runtime_python=self._runtime_python,
            )
        except RuntimeError as exc:
            if "opendet3d" not in str(exc):
                raise
            raise ImportError(
                "Install the optional 3D-MOOD runtime and its dependencies using "
                f"{INSTALL_URL}, then pass runtime_path and runtime_python. "
                f"Documented revision: {UPSTREAM_REVISION}."
            ) from exc

    def close(self):
        if getattr(self, "_backend", None) is not None:
            self._backend.close()
            self._backend = None

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
        stream=False,
        save=False,
        output_path=None,
        color_format="auto",
    ):
        """Detect text categories and return 2D boxes, 3D boxes, and depth."""
        calibration = np.array(intrinsics, dtype=np.float32, copy=True)
        Boxes3D(np.empty((0, 14), dtype=np.float32), intrinsics=calibration)
        names = self._texts(self._classes if text is None else text)
        is_many = isinstance(source, (list, tuple))
        if isinstance(source, (str, Path)) and Path(source).is_dir():
            source = sorted(
                path
                for path in Path(source).iterdir()
                if path.suffix.lower() in SUPPORTED_EXTENSIONS
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
                self._load()
                outputs = self._backend.predict(
                    np.asarray(image), calibration, {"text": names}
                )
                result = self._result(
                    outputs,
                    (image.height, image.width),
                    calibration,
                    names,
                    str(item) if isinstance(item, (str, Path)) else None,
                )
                if save:
                    if output_path is None:
                        if save_dir is None:
                            from ...utils.general import increment_path

                            save_dir = increment_path(
                                Path("runs/detect3d/3dmood"), mkdir=True
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
        if not isinstance(outputs, (tuple, list)) or len(outputs) != 6:
            raise ValueError("Unexpected 3D-MOOD output: expected six fields.")
        boxes, cuboids, scores, ids, depth, _ = outputs
        boxes = torch.as_tensor(boxes).detach().cpu().float()
        cuboids = torch.as_tensor(cuboids).detach().cpu().float()
        scores = torch.as_tensor(scores).detach().cpu().float()
        ids = torch.as_tensor(ids).detach().cpu().float()
        depth = torch.as_tensor(depth).detach().cpu().float()
        count = len(boxes)
        if boxes.shape != (count, 4) or cuboids.shape != (count, 10):
            raise ValueError("3D-MOOD returned malformed box geometry.")
        if scores.shape != (count,) or ids.shape != (count,):
            raise ValueError("3D-MOOD returned misaligned scores/classes.")
        if depth.shape != torch.Size(shape):
            raise ValueError("3D-MOOD returned depth outside the original image canvas.")
        if not all(torch.isfinite(value).all() for value in (boxes, cuboids, scores, ids, depth)):
            raise ValueError("3D-MOOD returned non-finite predictions.")
        if (boxes[:, 2:] < boxes[:, :2]).any() or (cuboids[:, 3:6] <= 0).any():
            raise ValueError("3D-MOOD returned invalid box geometry.")
        if ((scores < 0) | (scores > 1)).any():
            raise ValueError("3D-MOOD returned scores outside [0, 1].")
        if ((ids < 0) | (ids != ids.floor()) | (ids >= len(names))).any():
            raise ValueError("3D-MOOD returned class ids outside the vocabulary.")
        boxes_payload = torch.cat((boxes, scores[:, None], ids[:, None]), dim=1)
        cuboid_payload = torch.cat(
            (
                cuboids,
                scores[:, None],
                ids[:, None],
                scores[:, None],
                torch.ones_like(scores[:, None]),
            ),
            dim=1,
        )
        return Results(
            Boxes(
                boxes_payload[:, :4],
                boxes_payload[:, 4],
                boxes_payload[:, 5],
                orig_shape=shape,
            ),
            shape,
            path=path,
            names=dict(enumerate(names)),
            boxes3d=Boxes3D(cuboid_payload, shape, torch.as_tensor(intrinsics).clone()),
            depth_map=DepthMap(depth, shape),
        )

    def train(self, *args, **kwargs):
        raise NotImplementedError(
            "3D-MOOD training is not integrated; use the upstream multi-dataset recipe."
        )

    def val(self, *args, **kwargs):
        raise NotImplementedError(
            "3D-MOOD validation requires its upstream Omni3D/open-set evaluator."
        )

    def export(self, *args, **kwargs):
        raise NotImplementedError("3D-MOOD export is not supported by this adapter.")

    def track(self, *args, **kwargs):
        raise NotImplementedError("3D-MOOD tracking is not integrated.")
