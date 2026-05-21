"""Face-detector layer for L2CS gaze inference.

L2CS-Net needs face bounding boxes upstream of the gaze head. Rather than
bundling a specific face detector as a hard dependency, this module defines
a small ``FaceDetector`` protocol and three adapters:

* ``CallableFaceDetector`` — wraps any ``image -> list[FaceBox]`` callable.
* ``LibreYOLOFaceDetector`` — adapts an existing LibreYOLO detector (e.g.
  a YOLO9 model fine-tuned on faces) into the protocol.
* ``RetinaFaceAdapter`` — optional, lazy-imports ``face_detection`` for
  parity with upstream L2CS-Net's pipeline. Behind the ``gaze-retinaface``
  optional extra; never imported eagerly.

Callers may also pre-compute face boxes and pass them directly to
``LibreL2CS(...)``, bypassing the protocol entirely (the cleanest path
for composition with external detectors).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Iterable, List, Optional, Protocol, Sequence

import numpy as np

if TYPE_CHECKING:
    from PIL import Image


@dataclass
class FaceBox:
    """A single face detection.

    Attributes
    ----------
    xyxy : tuple
        ``(x1, y1, x2, y2)`` in pixel coordinates of the source image.
    score : float
        Detector confidence in [0, 1]. Used for filtering only.
    landmarks : np.ndarray | None
        Optional ``(K, 2)`` landmarks (e.g. 5-pt for RetinaFace). May be None.
    """

    xyxy: tuple
    score: float = 1.0
    landmarks: Optional[np.ndarray] = None


class FaceDetector(Protocol):
    """Anything callable that maps a numpy RGB image to a list of ``FaceBox``."""

    def __call__(self, image_rgb: np.ndarray) -> List[FaceBox]:  # pragma: no cover - protocol
        ...


@dataclass
class CallableFaceDetector:
    """Adapt an arbitrary ``image -> iterable`` callable into a ``FaceDetector``.

    The callable may return any of:

    * a list of ``FaceBox`` instances,
    * a list of ``(x1, y1, x2, y2)`` tuples (score defaults to 1.0),
    * a list of ``(x1, y1, x2, y2, score)`` tuples,
    * a numpy array of shape ``(N, 4)`` or ``(N, 5)``.
    """

    fn: Callable[[np.ndarray], Any]
    min_score: float = 0.0

    def __call__(self, image_rgb: np.ndarray) -> List[FaceBox]:
        raw = self.fn(image_rgb)
        return _normalize_boxes(raw, self.min_score)


@dataclass
class LibreYOLOFaceDetector:
    """Adapt a LibreYOLO detector that emits face boxes into a ``FaceDetector``.

    Any LibreYOLO model that returns ``Results`` with ``.boxes.xyxy`` works.
    By default all detected boxes are treated as faces; pass ``face_class``
    to filter to a specific class id when the underlying model is multi-class.
    """

    model: Any
    conf: float = 0.4
    face_class: Optional[int] = None
    imgsz: Optional[int] = None

    def __call__(self, image_rgb: np.ndarray) -> List[FaceBox]:
        # Avoid the import cost when not used
        from PIL import Image

        pil = Image.fromarray(image_rgb)
        kwargs = {"conf": self.conf}
        if self.imgsz is not None:
            kwargs["imgsz"] = self.imgsz
        if self.face_class is not None:
            kwargs["classes"] = [self.face_class]
        result = self.model(pil, **kwargs)
        if isinstance(result, list):
            result = result[0]

        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy
        if hasattr(xyxy, "cpu"):
            xyxy = xyxy.cpu().numpy()
        conf = boxes.conf
        if hasattr(conf, "cpu"):
            conf = conf.cpu().numpy()
        return [
            FaceBox(xyxy=(float(b[0]), float(b[1]), float(b[2]), float(b[3])),
                    score=float(s))
            for b, s in zip(xyxy, conf)
        ]


@dataclass
class RetinaFaceAdapter:
    """Optional adapter around ``face_detection.RetinaFace`` for upstream parity.

    Requires the ``gaze-retinaface`` optional extra (``pip install
    libreyolo[gaze-retinaface]``). Not imported eagerly.
    """

    min_score: float = 0.5
    gpu_id: Optional[int] = None
    _impl: Any = field(default=None, init=False, repr=False)

    def _load(self) -> None:
        if self._impl is not None:
            return
        try:
            from face_detection import RetinaFace
        except ImportError as e:  # pragma: no cover - depends on env
            raise ImportError(
                "RetinaFaceAdapter requires the optional 'face_detection' package. "
                "Install with: pip install libreyolo[gaze-retinaface]"
            ) from e
        self._impl = RetinaFace() if self.gpu_id is None else RetinaFace(gpu_id=self.gpu_id)

    def __call__(self, image_rgb: np.ndarray) -> List[FaceBox]:
        self._load()
        # face_detection expects BGR
        image_bgr = image_rgb[:, :, ::-1].copy()
        faces = self._impl(image_bgr)
        if faces is None:
            return []
        out: List[FaceBox] = []
        for box, landmark, score in faces:
            if float(score) < self.min_score:
                continue
            out.append(
                FaceBox(
                    xyxy=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                    score=float(score),
                    landmarks=np.asarray(landmark, dtype=np.float32),
                )
            )
        return out


def _normalize_boxes(raw: Any, min_score: float) -> List[FaceBox]:
    """Coerce a flexible callable's output into a list of FaceBox."""
    if raw is None:
        return []
    if isinstance(raw, np.ndarray):
        return _from_array(raw, min_score)
    boxes: List[FaceBox] = []
    for item in raw:
        if isinstance(item, FaceBox):
            if item.score >= min_score:
                boxes.append(item)
            continue
        seq = list(item)
        if len(seq) == 4:
            boxes.append(FaceBox(xyxy=(float(seq[0]), float(seq[1]),
                                       float(seq[2]), float(seq[3])),
                                 score=1.0))
        elif len(seq) >= 5:
            score = float(seq[4])
            if score < min_score:
                continue
            boxes.append(FaceBox(xyxy=(float(seq[0]), float(seq[1]),
                                       float(seq[2]), float(seq[3])),
                                 score=score))
        else:
            raise ValueError(
                f"Unsupported face-detector tuple length {len(seq)}; expected 4 or 5+."
            )
    return boxes


def _from_array(arr: np.ndarray, min_score: float) -> List[FaceBox]:
    if arr.ndim != 2 or arr.shape[1] not in (4, 5):
        raise ValueError(
            f"Expected face-detector array of shape (N, 4) or (N, 5), got {arr.shape}."
        )
    boxes: List[FaceBox] = []
    for row in arr:
        if arr.shape[1] == 5:
            score = float(row[4])
            if score < min_score:
                continue
            boxes.append(FaceBox(xyxy=(float(row[0]), float(row[1]),
                                       float(row[2]), float(row[3])),
                                 score=score))
        else:
            boxes.append(FaceBox(xyxy=(float(row[0]), float(row[1]),
                                       float(row[2]), float(row[3])),
                                 score=1.0))
    return boxes


def resolve_face_detector(spec: Any) -> Optional[FaceDetector]:
    """Coerce a user-supplied detector spec into a ``FaceDetector`` or None.

    Accepts: None (no detector), a ``FaceDetector``-protocol object,
    a plain callable, or a LibreYOLO model wrapper (anything with a
    ``__call__`` that returns Results with ``.boxes``).
    """
    if spec is None:
        return None
    if isinstance(spec, (CallableFaceDetector, LibreYOLOFaceDetector, RetinaFaceAdapter)):
        return spec
    # Duck-type a LibreYOLO model: has a `_runner` attribute and a `predict` method
    if hasattr(spec, "_runner") or hasattr(spec, "predict"):
        return LibreYOLOFaceDetector(model=spec)
    if callable(spec):
        return CallableFaceDetector(fn=spec)
    raise TypeError(
        f"Unsupported face_detector spec: {type(spec).__name__}. "
        "Provide a callable, a LibreYOLO model, or a FaceDetector instance."
    )
