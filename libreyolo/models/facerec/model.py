"""LibreYOLO face-embedding model (task ``embed`` / facial-recognition).

Inference-only, two-stage and ONNX-backed — structurally a sibling of the gaze
(L2CS) task: a face detector locates faces (and ideally 5 landmarks), each face
is aligned to a canonical 112x112 crop, and a recognition head produces an
L2-normalized identity embedding. Verification/identification is then a cosine
similarity (== dot product of unit vectors).

The recognition head is consumed as an opaque ONNX graph via onnxruntime, so no
third-party architecture code is ported. Training and dataset validation are out
of scope here (exactly like L2CS) — embeddings are an inference product.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import torch

from ..l2cs.face import FaceBox, FaceDetector, resolve_face_detector
from .preprocess import PreprocCfg, l2_normalize, preprocess_aligned, resolve_preproc

logger = logging.getLogger(__name__)


class LibreFaceEmbedder:
    """Face-embedding model: image -> per-face (box + L2-normalized embedding).

    Args:
        model_path: Path to a face-recognition ONNX model that maps an
            ``(N, 3, 112, 112)`` aligned-face batch to ``(N, D)`` embeddings.
        preproc: A ``PreprocCfg``, a preset name (``"arcface"`` | ``"raw_bgr"``),
            or ``None`` (defaults to the ArcFace convention: RGB,
            ``(x-127.5)/127.5``).
        device: ``"auto"`` | ``"cpu"`` | ``"cuda"``.
        face_detector: Optional default detector (a callable, a LibreYOLO
            detection model, or a ``FaceDetector``). May also be supplied per
            call, or bypassed entirely with ``face_boxes=...``.
    """

    FAMILY = "facerec"
    SUPPORTED_TASKS = ("embed",)
    DEFAULT_TASK = "embed"
    # This class does not derive from BaseModel, so it declares the batching
    # contract itself: callers that read the flag off an arbitrary model
    # default to True when it is absent, and predict() takes a single image.
    # Without this, predict([img_a, img_b]) reached ImageLoader and raised
    # "Unsupported image type: list", which reads as a broken model rather
    # than an unsupported call shape.
    SUPPORTS_BATCHED_PREDICT = False

    def __init__(
        self,
        model_path: str,
        *,
        preproc: "PreprocCfg | str | None" = None,
        device: str = "auto",
        face_detector: Any = None,
        names: Optional[dict] = None,
        task: str | None = "embed",
    ):
        try:
            import onnxruntime as ort
        except ImportError as e:  # pragma: no cover - env dependent
            raise ImportError(
                "Face embedding requires onnxruntime. "
                'Install with: pip install "libreyolo[onnx]"'
            ) from e

        from ...backends.axelera import is_axelera_artifact

        axelera_head = Path(model_path).exists() and is_axelera_artifact(model_path)
        if not Path(model_path).exists():
            from .weights import is_facerec_weight_name, resolve_facerec_weight

            if is_facerec_weight_name(model_path):
                model_path = resolve_facerec_weight(model_path)
            else:
                raise FileNotFoundError(
                    f"Face-embedding ONNX model not found: {model_path}"
                )

        if axelera_head:
            # A recognition head exported with export(format="axelera"): the
            # head runs on the Metis AIPU; detection and alignment stay on the host.
            self.session = _AxeleraHeadSession(model_path)
            self.device = "axelera"
            if preproc is None:
                preproc = self.session.preproc
        else:
            available = ort.get_available_providers()
            if device in ("auto", "cuda", "gpu") and "CUDAExecutionProvider" in available:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                self.device = "cuda"
            else:
                providers = ["CPUExecutionProvider"]
                self.device = "cpu"

            so = ort.SessionOptions()
            so.log_severity_level = 3  # silence "initializer in graph inputs" spam
            self.session = ort.InferenceSession(model_path, sess_options=so, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

        self.model_path = model_path
        self.family = self.FAMILY
        self.task = "embed"
        self.cfg = resolve_preproc(preproc)
        self.names = names or {0: "face"}
        self.face_detector = (
            resolve_face_detector(face_detector) if face_detector is not None else None
        )
        self._default_detector_instance = None

        out_dim = self.session.get_outputs()[0].shape[-1]
        self._dim = int(out_dim) if isinstance(out_dim, int) else None

    # ------------------------------------------------------------------
    @property
    def dim(self) -> Optional[int]:
        """Embedding dimension (``None`` if the ONNX graph declares it dynamic)."""
        return self._dim

    def embed_aligned(self, aligned_crops: List[np.ndarray]) -> np.ndarray:
        """List of aligned HxWx3 RGB uint8 crops -> ``(N, D)`` L2-normalized."""
        if not aligned_crops:
            return np.zeros((0, self._dim or 0), dtype=np.float32)
        outs = []
        for crop in aligned_crops:
            blob = preprocess_aligned(crop, self.cfg)
            out = self.session.run(None, {self.input_name: blob})[0]
            outs.append(np.asarray(out, dtype=np.float32).reshape(1, -1))
        emb = np.concatenate(outs, axis=0)
        self._dim = emb.shape[1]
        return l2_normalize(emb, axis=1)

    def default_face_detector(self):
        """The family's default detector (auto-downloaded on first use)."""
        if self._default_detector_instance is None:
            from .weights import default_face_detector

            self._default_detector_instance = default_face_detector()
        return self._default_detector_instance

    # ------------------------------------------------------------------
    @property
    def _runner(self):
        if getattr(self, "_runner_instance", None) is None:
            from .inference import FaceEmbedRunner

            self._runner_instance = FaceEmbedRunner(self)
        return self._runner_instance

    def __call__(self, source=None, **kwargs):
        return self._runner(source, **kwargs)

    def predict(self, *args, **kwargs):
        return self(*args, **kwargs)

    def embed(self, source=None, **kwargs) -> torch.Tensor:
        """Return all face-region rows across ``source`` as ``(N_total, D)``."""
        from ...utils.results import stack_result_embeddings

        prediction = (
            [self.predict(item, **kwargs) for item in source]
            if isinstance(source, (list, tuple))
            else self.predict(source, **kwargs)
        )
        return stack_result_embeddings(prediction)

    def verify(
        self,
        image_a,
        image_b,
        *,
        threshold: float = 0.4,
        face_detector: Any = None,
        face_boxes_a: Any = None,
        face_boxes_b: Any = None,
    ) -> dict:
        """Compare the most prominent face in two images.

        Returns ``{'similarity', 'same_person', 'threshold'}`` where similarity
        is cosine similarity in ``[-1, 1]``.
        """
        ra = self(image_a, face_detector=face_detector, face_boxes=face_boxes_a)
        rb = self(image_b, face_detector=face_detector, face_boxes=face_boxes_b)
        ea = _top_embedding(ra)
        eb = _top_embedding(rb)
        if ea is None or eb is None:
            raise RuntimeError(
                "verify() needs at least one face in each image "
                f"(image_a faces={0 if ea is None else 1}, image_b faces={0 if eb is None else 1})."
            )
        sim = float(np.dot(ea, eb))
        return {"similarity": sim, "same_person": sim >= threshold, "threshold": threshold}

    # ------------------------------------------------------------------
    def train(self, *args, **kwargs):
        raise NotImplementedError(
            "Training is out of scope for LibreFaceEmbedder (inference-only). "
            "Train a recognition head upstream and export it to ONNX."
        )

    def val(self, *args, **kwargs):
        raise NotImplementedError(
            "Dataset validation is out of scope for LibreFaceEmbedder. Evaluate "
            "verification accuracy with the `compare` API on labeled pairs."
        )

    #: Axelera calibration: aligned faces found by the family's detector in the
    #: calibration dataset's images, at most this many per image and in total.
    AXELERA_CALIBRATION_FACES_PER_IMAGE = 4
    AXELERA_CALIBRATION_FACES = 128
    #: Measured on 890 aligned COCO faces vs the FP32 head: embedding cosine 0.9933
    #: (per_tensor_min_max 0.9738), every same/different decision at 0.4 kept
    #: (ledger facerec/embed/l/2026-11-20/3).
    AXELERA_QUANTIZATION_SCHEME = "hybrid_per_tensor_per_channel"

    def export(
        self,
        format: str = "onnx",
        *,
        output_path: str | Path | None = None,
        data: str | Path = "coco128.yaml",
        fraction: float = 1.0,
        quantization_scheme: str | None = None,
        verify: bool = False,
        **kwargs,
    ) -> str:
        """Compile the recognition head for an Axelera Metis (``format="axelera"``).

        The head keeps its contract: an aligned, normalized face
        ``[1, 3, 112, 112]`` -> one embedding ``[1, D]``. Face detection,
        alignment and L2 normalization stay on the host, as with the ONNX head.
        INT8 calibration uses faces that the family's detector finds in
        ``data``'s images, aligned exactly as ``predict()`` aligns them. Load
        the result with ``LibreYOLO("<stem>_axelera_model")`` (a
        ``librefacerec-*`` stem) or ``LibreFaceEmbedder(<dir>)``.
        """
        if str(format).lower() != "axelera":
            raise NotImplementedError(
                "LibreFaceEmbedder already wraps an ONNX graph; re-export is not "
                "supported except format='axelera' (the head compiled for Metis)."
            )
        if kwargs:
            raise TypeError(f"Unsupported face-embedding export arguments: {', '.join(sorted(kwargs))}")
        if isinstance(self.session, _AxeleraHeadSession):
            raise ValueError("This embedder already runs an Axelera export; export from the ONNX head.")
        if not 0.0 < float(fraction) <= 1.0:
            raise ValueError("fraction must be in (0, 1].")
        return _export_axelera_head(
            self,
            Path(output_path or f"{Path(self.model_path).stem}_axelera_model"),
            data=data,
            fraction=float(fraction),
            quantization_scheme=quantization_scheme or self.AXELERA_QUANTIZATION_SCHEME,
            verify=verify,
        )


def _top_embedding(result) -> Optional[np.ndarray]:
    """Highest-confidence face embedding from a Results, or None."""
    emb = getattr(result, "embeddings", None)
    if emb is None or len(emb) == 0:
        return None
    data = emb.numpy().data if hasattr(emb, "numpy") else emb.data
    data = np.asarray(data, dtype=np.float32)
    boxes = result.boxes
    if boxes is not None and len(boxes) == len(data):
        conf = np.asarray(boxes.numpy().conf, dtype=np.float32)
        idx = int(np.argmax(conf)) if conf.size else 0
    else:
        idx = 0
    return l2_normalize(data[idx])


class OpenCVFaceDetector:
    """Face detector + 5 landmarks via OpenCV's bundled ``cv2.FaceDetectorYN``.

    Returns ``FaceBox`` with 5-point landmarks, ready for canonical alignment.
    The detector API ships with OpenCV (a core dependency, Apache-2.0); the
    ``.onnx`` model file is supplied by the caller. Lazy — cv2/model are only
    touched on first use.
    """

    def __init__(self, model_path: str, score_threshold: float = 0.6, nms_threshold: float = 0.3):
        self.model_path = str(model_path)
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self._impl = None

    def _load(self):
        if self._impl is None:
            import cv2

            if not Path(self.model_path).exists():
                raise FileNotFoundError(f"Face-detector model not found: {self.model_path}")
            self._impl = cv2.FaceDetectorYN.create(
                self.model_path, "", (320, 320),
                score_threshold=self.score_threshold, nms_threshold=self.nms_threshold,
            )

    def __call__(self, image_rgb: np.ndarray) -> List[FaceBox]:
        self._load()
        bgr = np.ascontiguousarray(image_rgb[:, :, ::-1])
        h, w = bgr.shape[:2]
        self._impl.setInputSize((w, h))
        _, faces = self._impl.detect(bgr)
        if faces is None:
            return []
        out: List[FaceBox] = []
        for f in faces:
            x, y, bw, bh = f[:4]
            out.append(
                FaceBox(
                    xyxy=(float(x), float(y), float(x + bw), float(y + bh)),
                    score=float(f[-1]),
                    landmarks=f[4:14].reshape(5, 2).astype(np.float32),
                )
            )
        return out


class _AxeleraHeadSession:
    """The ``onnxruntime.InferenceSession`` subset ``LibreFaceEmbedder`` uses,
    backed by a recognition head compiled with ``export(format="axelera")``."""

    class _Arg:
        def __init__(self, name, shape):
            self.name, self.shape = name, shape

    def __init__(self, directory: str | Path):
        import yaml

        from ...backends.axelera import (
            AxeleraBackend,
            require_axelera_rt,
            resolve_axelera_artifact,
        )

        axm, metadata_path = resolve_axelera_artifact(directory)
        meta = yaml.safe_load(Path(metadata_path).read_text()) or {}
        if meta.get("model_family") != "facerec":
            raise ValueError(f"{directory} is not a face-embedding (facerec) Axelera export.")
        require_axelera_rt()
        self.metadata = meta
        size = int(meta.get("imgsz", [112, 112])[0])
        dim = int(meta.get("embedding_dim", 0)) or None
        self._inputs = [self._Arg(meta.get("input_names", ["data"])[0], [1, 3, size, size])]
        self._outputs = [self._Arg("embedding", [1, dim])]
        self.preproc = PreprocCfg(**meta["preprocess"]) if meta.get("preprocess") else None
        self._model = AxeleraBackend._load(axm)

    def get_inputs(self):
        return self._inputs

    def get_outputs(self):
        return self._outputs

    def run(self, output_names, feeds):
        from ...backends.axelera import _no_device_error

        (blob,) = feeds.values()
        try:
            out = self._model(np.ascontiguousarray(blob, dtype=np.float32))
        except RuntimeError as exc:
            raise _no_device_error(exc) from exc
        out = out[0] if isinstance(out, (list, tuple)) else out
        return [np.asarray(out, dtype=np.float32).reshape(1, -1)]


def _axelera_calibration_faces(embedder, data, fraction: float, split: str, limit: int) -> list:
    """Aligned, normalized ``[1, 3, S, S]`` faces from a dataset's images (predict()'s own steps)."""
    import cv2

    from ...data.utils import get_img_files, load_data_config
    from .align import align_face

    config = load_data_config(str(data), autodownload=True)
    root = Path(config.get("path", "."))
    source = config.get(split) or config.get("train") or config.get("val")
    if source is None:
        raise ValueError(f"Dataset config {data!r} has no {split!r} or 'val' split.")
    images = get_img_files(source, prefix=str(root))
    images = images[: max(1, int(len(images) * fraction))]
    detector = embedder.face_detector or embedder.default_face_detector()
    size = embedder.cfg.size
    faces = []
    for image_path in images:
        bgr = cv2.imread(str(image_path))
        if bgr is None:
            continue
        rgb = np.ascontiguousarray(bgr[:, :, ::-1])
        found = 0
        for face in detector(rgb):
            if found >= embedder.AXELERA_CALIBRATION_FACES_PER_IMAGE or len(faces) >= limit:
                break
            try:
                crop = align_face(rgb, face.xyxy, face.landmarks, image_size=size)
            except ValueError:
                continue
            faces.append(preprocess_aligned(crop, embedder.cfg))
            found += 1
        if len(faces) >= limit:
            break
    return faces


def _export_axelera_head(embedder, destination: Path, *, data, fraction, quantization_scheme, verify) -> str:
    import dataclasses
    import tempfile

    import onnx
    import torch

    from ...export.axelera import export_axelera

    if embedder.cfg.layout != "NCHW":
        raise ValueError("Axelera exports take NCHW inputs; use an NCHW preprocessing preset.")
    calibration = _axelera_calibration_faces(
        embedder, data, fraction, "train", embedder.AXELERA_CALIBRATION_FACES
    )
    if not calibration:
        raise ValueError(f"The face detector found no faces to calibrate on in {data!r}.")
    samples = _axelera_calibration_faces(embedder, data, 1.0, "val", 32) if verify else None
    size = embedder.cfg.size
    session = embedder.session
    input_name = embedder.input_name

    class _Reference(torch.nn.Module):
        output_names = ["embedding"]

        def forward(self, x):
            out = session.run(None, {input_name: x.detach().cpu().numpy().astype(np.float32)})[0]
            return (torch.from_numpy(np.asarray(out, dtype=np.float32).reshape(1, -1)),)

    with tempfile.TemporaryDirectory(prefix="facerec_axelera_") as work:
        graph = onnx.load(embedder.model_path)
        (graph_input,) = [i for i in graph.graph.input if i.name == input_name]
        dims = graph_input.type.tensor_type.shape.dim
        dims[0].Clear()
        dims[0].dim_value = 1  # compiled at batch 1
        static = Path(work) / "head.onnx"
        onnx.save(graph, str(static))
        dim = embedder.dim
        return export_axelera(
            onnx_path=str(static),
            output_path=str(destination),
            stem=destination.name.removesuffix("_axelera_model") or "model",
            metadata={
                "model_family": "facerec",
                "size": Path(embedder.model_path).stem.removeprefix("librefacerec-"),
                "task": "embed",
                "imgsz": [size, size],
                "input_names": [input_name],
                "embedding_dim": int(dim) if dim else None,
                "preprocess": dataclasses.asdict(embedder.cfg),
                "contract": f"aligned face [1,3,{size},{size}] (preprocess) -> embedding [1,D]; "
                "face detection, alignment and L2 normalization in Python",
            },
            calibration=calibration,
            calibration_info={
                "data": str(data),
                "num_images": len(calibration),
                "fraction": float(fraction),
                "batch": 1,
                "preprocess": "facerec aligned faces (family detector, 5-point alignment)",
            },
            quantization_scheme=quantization_scheme,
            # The embedding Gemm reads the flattened 7x7x512 map (25088 inputs, 392
            # reduction blocks): no MVM mapping, so it runs as row convolutions, as VGG's fc6.
            graph_passes=("fc_rowconv",),
            verify=verify,
            reference_model=_Reference() if verify else None,
            verify_samples=samples,
            verify_source="dataset faces",
        )
