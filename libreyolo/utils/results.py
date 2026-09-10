"""Flat result containers for LibreYOLO."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
from .lazy import lazy_module


# torch is resolved on first use so this module stays importable in a
# torch-free ONNX deployment (discussions/711).
torch = lazy_module("torch")


if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch as _torch

    TensorLike = Union[_torch.Tensor, np.ndarray]
else:
    # Evaluated at import time, so it must not touch the lazy torch proxy.
    # Annotations elsewhere in this module are strings (see __future__ import),
    # so nothing dereferences this alias at runtime.
    TensorLike = Any


def _move(data: TensorLike | None, *args, **kwargs):
    if data is None:
        return None
    if isinstance(data, torch.Tensor):
        return data.to(*args, **kwargs)
    if isinstance(data, np.ndarray):
        return torch.as_tensor(data).to(*args, **kwargs)
    return data


def _cpu(data: TensorLike | None):
    if isinstance(data, torch.Tensor):
        return data.cpu()
    return data


def _cuda(data: TensorLike | None):
    if isinstance(data, torch.Tensor):
        return data.cuda()
    if isinstance(data, np.ndarray):
        return torch.as_tensor(data).cuda()
    return data


def _numpy(data: TensorLike | None):
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return data


def _slice_first(data: TensorLike | None, idx):
    if data is None:
        return None
    sliced = data[idx]
    if isinstance(sliced, torch.Tensor):
        if sliced.ndim == data.ndim - 1:
            sliced = sliced.unsqueeze(0)
    elif isinstance(sliced, np.ndarray):
        if sliced.ndim == data.ndim - 1:
            sliced = np.expand_dims(sliced, axis=0)
    else:
        sliced = np.asarray([sliced])
    return sliced


class Boxes:
    """Wrap detection boxes for a single image."""

    def __init__(
        self,
        boxes: TensorLike,
        conf: TensorLike,
        cls: TensorLike,
        id: TensorLike | None = None,
        orig_shape: Tuple[int, int] | None = None,
    ):
        self._boxes = boxes
        self._conf = conf
        self._cls = cls
        self._id = id
        self.orig_shape = orig_shape

    @property
    def xyxy(self) -> TensorLike:
        return self._boxes

    @property
    def conf(self) -> TensorLike:
        return self._conf

    @property
    def cls(self) -> TensorLike:
        return self._cls

    @property
    def id(self) -> TensorLike | None:
        return self._id

    @property
    def is_track(self) -> bool:
        return self._id is not None

    @property
    def xywh(self) -> TensorLike:
        b = self._boxes
        x1, y1, x2, y2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        if isinstance(b, torch.Tensor):
            return torch.stack([cx, cy, w, h], dim=1)
        return np.stack([cx, cy, w, h], axis=1)

    @property
    def xyxyn(self) -> TensorLike:
        """Normalized xyxy boxes."""
        return self._normalize_boxes(self.xyxy)

    @property
    def xywhn(self) -> TensorLike:
        """Normalized xywh boxes."""
        return self._normalize_boxes(self.xywh)

    def _normalize_boxes(self, boxes: TensorLike) -> TensorLike:
        if self.orig_shape is None:
            raise ValueError("orig_shape is required for normalized box coordinates")
        h, w = self.orig_shape
        if isinstance(boxes, torch.Tensor):
            scale = torch.tensor([w, h, w, h], dtype=boxes.dtype, device=boxes.device)
        else:
            scale = np.array([w, h, w, h], dtype=boxes.dtype)
        return boxes / scale

    def with_id(self, id: TensorLike | None) -> "Boxes":
        return Boxes(self._boxes, self._conf, self._cls, id, self.orig_shape)

    def with_orig_shape(self, orig_shape: Tuple[int, int] | None) -> "Boxes":
        return Boxes(self._boxes, self._conf, self._cls, self._id, orig_shape)

    @property
    def data(self) -> TensorLike:
        parts = [self._boxes]
        if self._id is not None:
            parts.append(self._id.reshape(-1, 1))
        parts.extend([self._conf.reshape(-1, 1), self._cls.reshape(-1, 1)])
        if isinstance(self._boxes, torch.Tensor):
            return torch.cat(parts, dim=1)
        return np.concatenate(parts, axis=1)

    def to(self, *args, **kwargs) -> "Boxes":
        return Boxes(
            _move(self._boxes, *args, **kwargs),
            _move(self._conf, *args, **kwargs),
            _move(self._cls, *args, **kwargs),
            _move(self._id, *args, **kwargs),
            self.orig_shape,
        )

    def cpu(self) -> "Boxes":
        return Boxes(
            _cpu(self._boxes),
            _cpu(self._conf),
            _cpu(self._cls),
            _cpu(self._id),
            self.orig_shape,
        )

    def cuda(self) -> "Boxes":
        return Boxes(
            _cuda(self._boxes),
            _cuda(self._conf),
            _cuda(self._cls),
            _cuda(self._id),
            self.orig_shape,
        )

    def numpy(self) -> "Boxes":
        return Boxes(
            _numpy(self._boxes),
            _numpy(self._conf),
            _numpy(self._cls),
            _numpy(self._id),
            self.orig_shape,
        )

    def __getitem__(self, idx) -> "Boxes":
        return Boxes(
            _slice_first(self._boxes, idx),
            _slice_first(self._conf, idx),
            _slice_first(self._cls, idx),
            _slice_first(self._id, idx),
            self.orig_shape,
        )

    def __len__(self) -> int:
        return int(self._boxes.shape[0])

    def __repr__(self) -> str:
        return (
            f"Boxes(n={len(self)}, "
            f"xyxy={tuple(self._boxes.shape)}, "
            f"conf={tuple(self._conf.shape)}, "
            f"cls={tuple(self._cls.shape)}, "
            f"is_track={self.is_track})"
        )


class Masks:
    """Wrap instance masks for a single image."""

    def __init__(
        self,
        masks: TensorLike,
        orig_shape: Tuple[int, int],
    ):
        self._masks = masks
        self.orig_shape = orig_shape

    @property
    def data(self) -> TensorLike:
        return self._masks

    @property
    def xy(self) -> List[np.ndarray]:
        return self._masks_to_contours(normalize=False)

    @property
    def xyn(self) -> List[np.ndarray]:
        return self._masks_to_contours(normalize=True)

    def _masks_to_contours(self, normalize: bool) -> List[np.ndarray]:
        import cv2

        masks_np = _numpy(self._masks).astype(np.uint8)
        h, w = self.orig_shape
        contours_list = []
        for mask in masks_np:
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if contours:
                contour = (
                    max(contours, key=cv2.contourArea).squeeze(1).astype(np.float64)
                )
                if normalize:
                    contour[:, 0] /= w
                    contour[:, 1] /= h
                contours_list.append(contour)
            else:
                contours_list.append(np.empty((0, 2), dtype=np.float64))
        return contours_list

    def to(self, *args, **kwargs) -> "Masks":
        moved = _move(self._masks, *args, **kwargs)
        if moved is self._masks and not isinstance(moved, torch.Tensor):
            return self
        return Masks(moved, self.orig_shape)

    def cpu(self) -> "Masks":
        if isinstance(self._masks, torch.Tensor):
            return Masks(self._masks.cpu(), self.orig_shape)
        return self

    def cuda(self) -> "Masks":
        if isinstance(self._masks, torch.Tensor):
            return Masks(self._masks.cuda(), self.orig_shape)
        return self

    def numpy(self) -> "Masks":
        if isinstance(self._masks, torch.Tensor):
            return Masks(self._masks.detach().cpu().numpy(), self.orig_shape)
        return self

    def __getitem__(self, idx) -> "Masks":
        return Masks(_slice_first(self._masks, idx), self.orig_shape)

    def __len__(self) -> int:
        return int(self._masks.shape[0])

    def __repr__(self) -> str:
        return (
            f"Masks(n={len(self)}, "
            f"shape={tuple(self._masks.shape)}, "
            f"orig_shape={self.orig_shape})"
        )


class _TensorPayload:
    """Small wrapper used for future flat result slots."""

    def __init__(self, data: TensorLike, orig_shape: Tuple[int, int] | None = None):
        self.data = data
        self.orig_shape = orig_shape

    def to(self, *args, **kwargs):
        return self.__class__(_move(self.data, *args, **kwargs), self.orig_shape)

    def cpu(self):
        return self.__class__(_cpu(self.data), self.orig_shape)

    def cuda(self):
        return self.__class__(_cuda(self.data), self.orig_shape)

    def numpy(self):
        return self.__class__(_numpy(self.data), self.orig_shape)

    def __getitem__(self, idx):
        return self.__class__(_slice_first(self.data, idx), self.orig_shape)

    def __len__(self) -> int:
        return int(self.data.shape[0])


class Boxes3D(_TensorPayload):
    """Camera-frame cuboids, metres, with scalar-first quaternions.

    ``data`` is (N, 14): center xyz, dimensions wlh, quaternion
    wxyz, combined ranking score, class id, 2D confidence, 3D confidence.
    ``intrinsics`` is the shared (3, 3) calibration on the original canvas.
    Camera axes are x right, y down, z forward. Local box axes carry length,
    height, width respectively, following vis4d's OpenCV box convention.
    No world frame is implied.
    """

    def __init__(self, data, orig_shape=None, intrinsics=None):
        if not isinstance(data, (np.ndarray, torch.Tensor)):
            data = np.asarray(data, dtype=np.float32)
        if intrinsics is not None and not isinstance(
            intrinsics, (np.ndarray, torch.Tensor)
        ):
            intrinsics = np.asarray(intrinsics, dtype=np.float32)
        super().__init__(data, orig_shape)
        values = np.asarray(_numpy(data))
        if values.ndim != 2 or values.shape[1] != 14:
            raise ValueError("Boxes3D data must have shape (N, 14).")
        if not np.isfinite(values).all():
            raise ValueError("Boxes3D data must be finite.")
        if (values[:, 3:6] <= 0).any():
            raise ValueError("Boxes3D dimensions must be positive.")
        if (np.linalg.norm(values[:, 6:10], axis=1) < 1e-8).any():
            raise ValueError("Boxes3D quaternions must be nonzero.")
        if (values[:, 10] < 0).any():
            raise ValueError("Boxes3D combined scores must be nonnegative.")
        confidences = values[:, [12, 13]]
        if ((confidences < 0) | (confidences > 1)).any():
            raise ValueError("Boxes3D 2D/3D confidence must be in [0, 1].")
        ids = values[:, 11]
        if ((ids < 0) | (ids != np.floor(ids))).any():
            raise ValueError("Boxes3D class ids must be nonnegative integers.")
        if intrinsics is not None:
            k = np.asarray(_numpy(intrinsics))
            if (
                k.shape != (3, 3)
                or not np.isfinite(k).all()
                or k[0, 0] <= 0
                or k[1, 1] <= 0
                or not np.allclose(k[2], [0, 0, 1])
            ):
                raise ValueError(
                    "intrinsics must be a finite calibrated (3, 3) matrix."
                )
        self.intrinsics = intrinsics

    def _validate_alignment(self, boxes, orig_shape):
        if boxes is None or len(boxes) != len(self):
            raise ValueError("Results.boxes3d must be row-aligned with Results.boxes.")
        if self.orig_shape is not None and tuple(self.orig_shape) != tuple(orig_shape):
            raise ValueError("Boxes3D orig_shape must match the Results image canvas.")

    def __repr__(self):
        return f"Boxes3D(n={len(self)}, orig_shape={self.orig_shape})"

    @property
    def xyz(self):
        return self.data[:, :3]

    @property
    def dimensions(self):
        return self.data[:, 3:6]

    @property
    def quaternion(self):
        return self.data[:, 6:10]

    @property
    def conf(self):
        return self.data[:, 10]

    @property
    def cls(self):
        return self.data[:, 11]

    @property
    def conf2d(self):
        return self.data[:, 12]

    @property
    def conf3d(self):
        return self.data[:, 13]

    def to(self, *args, **kwargs):
        return Boxes3D(
            _move(self.data, *args, **kwargs),
            self.orig_shape,
            _move(self.intrinsics, *args, **kwargs),
        )

    def cpu(self):
        return Boxes3D(_cpu(self.data), self.orig_shape, _cpu(self.intrinsics))

    def cuda(self):
        return self.to("cuda")

    def numpy(self):
        return Boxes3D(_numpy(self.data), self.orig_shape, _numpy(self.intrinsics))

    def __getitem__(self, idx):
        return Boxes3D(_slice_first(self.data, idx), self.orig_shape, self.intrinsics)

    @property
    def corners(self):
        """Return numpy (N, 8, 3) camera-frame corners, without mutating data."""
        values = np.asarray(_numpy(self.data), dtype=np.float64)
        q = values[:, 6:10]
        q = q / np.linalg.norm(q, axis=1, keepdims=True)
        w, x, y, z = q.T
        rotation = np.stack(
            [
                1 - 2 * (y * y + z * z),
                2 * (x * y - z * w),
                2 * (x * z + y * w),
                2 * (x * y + z * w),
                1 - 2 * (x * x + z * z),
                2 * (y * z - x * w),
                2 * (x * z - y * w),
                2 * (y * z + x * w),
                1 - 2 * (x * x + y * y),
            ],
            axis=1,
        ).reshape(-1, 3, 3)
        # OpenCV cuboid convention from SysCV/vis4d v1.0.0 (Apache-2.0),
        # op/box/box3d.py::boxes3d_to_corners. See THIRD_PARTY_NOTICES.txt.
        signs = np.array([
            [1, 1, -1], [1, 1, 1], [-1, 1, -1], [-1, 1, 1],
            [1, -1, -1], [1, -1, 1], [-1, -1, -1], [-1, -1, 1],
        ])
        local = signs[None] * values[:, None, [4, 5, 3]] / 2
        return np.einsum("nij,nkj->nki", rotation, local) + values[:, None, :3]


class Keypoints(_TensorPayload):
    @property
    def xy(self) -> TensorLike:
        return self.data[..., :2]

    @property
    def xyn(self) -> TensorLike:
        if self.orig_shape is None:
            raise ValueError("orig_shape is required for normalized keypoints")
        h, w = self.orig_shape
        xy = self.xy
        if isinstance(xy, torch.Tensor):
            scale = torch.tensor([w, h], dtype=xy.dtype, device=xy.device)
        else:
            scale = np.array([w, h], dtype=xy.dtype)
        return xy / scale

    @property
    def conf(self) -> TensorLike | None:
        if self.data.shape[-1] < 3:
            return None
        return self.data[..., 2]

    @property
    def has_visible(self) -> TensorLike:
        conf = self.conf
        if conf is None:
            if isinstance(self.data, torch.Tensor):
                return torch.ones(
                    self.data.shape[:-1], dtype=torch.bool, device=self.data.device
                )
            return np.ones(self.data.shape[:-1], dtype=bool)
        return conf > 0


class Points(_TensorPayload):
    """Wrap point-localization predictions for a single image.

    Data shape is ``(N, 4)`` with rows ``x, y, class, confidence``.
    Coordinates are absolute image pixels unless accessed through ``xyn``.
    """

    def __init__(self, data: TensorLike, orig_shape: Tuple[int, int] | None = None):
        if data.ndim == 1:
            if isinstance(data, torch.Tensor):
                data = data.unsqueeze(0)
            else:
                data = data[None, :]
        if data.ndim != 2 or data.shape[-1] != 4:
            raise ValueError(
                f"expected (N, 4) point rows but got shape {tuple(data.shape)}: "
                "x, y, class, confidence"
            )
        super().__init__(data, orig_shape)

    @property
    def xy(self) -> TensorLike:
        return self.data[:, :2]

    @property
    def xyn(self) -> TensorLike:
        if self.orig_shape is None:
            raise ValueError("orig_shape is required for normalized point coordinates")
        h, w = self.orig_shape
        xy = self.xy
        if isinstance(xy, torch.Tensor):
            scale = torch.tensor([w, h], dtype=xy.dtype, device=xy.device)
        else:
            scale = np.array([w, h], dtype=xy.dtype)
        return xy / scale

    @property
    def cls(self) -> TensorLike:
        return self.data[:, 2]

    @property
    def conf(self) -> TensorLike:
        return self.data[:, 3]

    def __repr__(self) -> str:
        return (
            f"Points(n={len(self)}, "
            f"shape={tuple(self.data.shape)}, "
            f"orig_shape={self.orig_shape})"
        )


class Probs(_TensorPayload):
    @property
    def top1(self) -> int:
        values = _numpy(self.data)
        return int(np.argmax(values))

    @property
    def top5(self) -> List[int]:
        values = _numpy(self.data)
        return np.argsort(values)[-5:][::-1].astype(int).tolist()

    @property
    def top1conf(self):
        return self.data[self.top1]

    @property
    def top5conf(self):
        indices = self.top5
        if isinstance(self.data, torch.Tensor):
            return self.data[torch.tensor(indices, device=self.data.device)]
        return self.data[indices]

    def __getitem__(self, idx):
        # A classification probability vector is whole-image, not per-detection;
        # keep it intact so shared Results slicing (e.g. ``result[0]``) cannot
        # truncate it to a single class. Mirrors SemanticMask/DepthMap.
        return self.__class__(self.data, self.orig_shape)


class SemanticMask(_TensorPayload):
    """Dense semantic segmentation map for a single image.

    Data shape is ``(H, W)`` integer class IDs on the original image canvas.
    ``255`` is the ignore/void value and never counts as a class.
    """

    IGNORE_INDEX = 255

    def __init__(self, data: TensorLike, orig_shape: Tuple[int, int] | None = None):
        if data.ndim != 2:
            raise ValueError(
                f"expected (H, W) semantic class map but got shape {tuple(data.shape)}"
            )
        if orig_shape is None:
            orig_shape = (int(data.shape[0]), int(data.shape[1]))
        super().__init__(data, orig_shape)

    @property
    def classes(self) -> List[int]:
        """Sorted class IDs present in the map, excluding the ignore value."""
        values = np.unique(_numpy(self.data))
        return [int(v) for v in values if int(v) != self.IGNORE_INDEX]

    def class_mask(self, class_id: int) -> TensorLike:
        """Boolean ``(H, W)`` mask selecting the pixels of one class."""
        return self.data == class_id

    def __getitem__(self, idx):
        # Instance indexing does not apply to a dense map; keep it intact so
        # shared Results slicing paths cannot corrupt the (H, W) layout.
        return self.__class__(self.data, self.orig_shape)

    def __repr__(self) -> str:
        return (
            f"SemanticMask(shape={tuple(self.data.shape)}, "
            f"classes={len(self.classes)}, orig_shape={self.orig_shape})"
        )


class PanopticSegmentation(_TensorPayload):
    """Panoptic segmentation result for a single image.

    Panoptic segmentation assigns every pixel exactly one non-overlapping
    segment, unifying "stuff" (amorphous background regions) and "things"
    (countable object instances). ``data`` is a ``(H, W)`` integer segment-id
    map on the original image canvas; ``segments_info`` describes each segment
    id that appears in the map.

    ``segments_info`` is a list of dicts, one per segment, each with at least::

        {"id": int, "category_id": int}

    where ``id`` matches a value in the map and ``category_id`` is the class
    index in the model's ``names``.

    thing-vs-stuff is a *per-category* property of the label set (mirroring the
    COCO-panoptic GT, where ``isthing`` lives on the ``categories`` list, not on
    per-segment ``segments_info`` entries), so the category metadata is the
    source of truth. As a convenience a prediction payload MAY denormalize it
    onto each segment (``"isthing": bool``, derived from ``category_id``); it is
    optional and, when present, must agree with the category-level map. This
    keeps the payload consistent with the GT contract in
    ``docs/dataset_schema.md`` and puts the derive-from-category responsibility
    on the producer (a model's ``_postprocess_predictions`` /
    ``PanopticValidator``), not on downstream consumers.

    ``predict`` populates this slot whenever a model family's ``_postprocess``
    returns a ``panoptic`` segment-id map plus ``segments_info``; evaluation is
    ``PanopticValidator`` (Panoptic Quality) over a ``PanopticDataset``.
    ``predict(save=True)`` renders the map via ``draw_panoptic`` and
    ``Results.summary`` reports one row per segment.
    """

    IGNORE_INDEX = 0  # COCO panoptic convention: segment id 0 is unlabeled/void.

    def __init__(
        self,
        data: TensorLike,
        segments_info: Optional[List[dict]] = None,
        orig_shape: Tuple[int, int] | None = None,
    ):
        if data.ndim != 2:
            raise ValueError(
                f"expected (H, W) panoptic segment-id map but got shape "
                f"{tuple(data.shape)}"
            )
        if orig_shape is None:
            orig_shape = (int(data.shape[0]), int(data.shape[1]))
        super().__init__(data, orig_shape)
        # Plain-Python metadata; carried verbatim across device/array moves.
        self.segments_info: List[dict] = list(segments_info or [])

    @property
    def segment_ids(self) -> List[int]:
        """Sorted segment ids present in the map, excluding the void id."""
        values = np.unique(_numpy(self.data))
        return [int(v) for v in values if int(v) != self.IGNORE_INDEX]

    def segment_mask(self, segment_id: int) -> TensorLike:
        """Boolean ``(H, W)`` mask selecting the pixels of one segment id."""
        return self.data == segment_id

    # segments_info is not tensor data, so the base _TensorPayload moves (which
    # rebuild via ``self.__class__(data, orig_shape)``) would drop it. Override
    # the move/slice methods to carry it through.
    def to(self, *args, **kwargs):
        return self.__class__(
            _move(self.data, *args, **kwargs), self.segments_info, self.orig_shape
        )

    def cpu(self):
        return self.__class__(_cpu(self.data), self.segments_info, self.orig_shape)

    def cuda(self):
        return self.__class__(_cuda(self.data), self.segments_info, self.orig_shape)

    def numpy(self):
        return self.__class__(_numpy(self.data), self.segments_info, self.orig_shape)

    def __getitem__(self, idx):
        # A dense panoptic map is whole-image, not per-instance; keep it intact
        # so shared Results slicing (e.g. ``result[0]``) cannot corrupt the
        # (H, W) layout. Mirrors SemanticMask/DepthMap.
        return self.__class__(self.data, self.segments_info, self.orig_shape)

    def __repr__(self) -> str:
        return (
            f"PanopticSegmentation(shape={tuple(self.data.shape)}, "
            f"segments={len(self.segment_ids)}, orig_shape={self.orig_shape})"
        )


class DepthMap(_TensorPayload):
    """Dense relative depth prediction with an explicit numeric encoding.

    Data is ``(H, W)`` on the original canvas. The default ``inverse_depth``
    encoding has larger values nearer the camera. ``depth`` and ``log_depth``
    have larger values farther away. Each is affine-relative in its named
    space; no metric unit or cross-image scale is implied.
    """

    ENCODINGS = ("inverse_depth", "depth", "log_depth")

    def __init__(
        self,
        data: TensorLike,
        orig_shape: Tuple[int, int] | None = None,
        *,
        encoding: str = "inverse_depth",
    ):
        if encoding not in self.ENCODINGS:
            raise ValueError(f"depth encoding must be one of {self.ENCODINGS}")
        if data.ndim != 2:
            raise ValueError(
                f"expected (H, W) depth map but got shape {tuple(data.shape)}"
            )
        if orig_shape is None:
            orig_shape = (int(data.shape[0]), int(data.shape[1]))
        super().__init__(data, orig_shape)
        self.encoding = encoding

    @property
    def near_is_high(self) -> bool:
        return self.encoding == "inverse_depth"

    def to(self, *args, **kwargs):
        return self.__class__(
            _move(self.data, *args, **kwargs), self.orig_shape, encoding=self.encoding
        )

    def cpu(self):
        return self.__class__(_cpu(self.data), self.orig_shape, encoding=self.encoding)

    def cuda(self):
        return self.__class__(_cuda(self.data), self.orig_shape, encoding=self.encoding)

    def numpy(self):
        return self.__class__(
            _numpy(self.data), self.orig_shape, encoding=self.encoding
        )

    def _finite_values(self) -> np.ndarray:
        values = np.asarray(_numpy(self.data), dtype=np.float32)
        return values[np.isfinite(values)]

    @property
    def min(self) -> float:
        values = self._finite_values()
        return float(values.min()) if values.size else 0.0

    @property
    def max(self) -> float:
        values = self._finite_values()
        return float(values.max()) if values.size else 0.0

    @property
    def mean(self) -> float:
        values = self._finite_values()
        return float(values.mean()) if values.size else 0.0

    def normalized(self) -> TensorLike:
        """Depth map rescaled to ``[0, 1]`` over finite values."""
        data = self.data
        lo, hi = self.min, self.max
        if hi - lo <= 0:
            return data * 0
        normalized = (data - lo) / (hi - lo)
        if isinstance(normalized, torch.Tensor):
            return torch.where(
                torch.isfinite(normalized), normalized, torch.zeros_like(normalized)
            )
        return np.where(np.isfinite(normalized), normalized, np.zeros_like(normalized))

    def __getitem__(self, idx):
        # Instance indexing does not apply to a dense map; keep it intact so
        # shared Results slicing paths cannot corrupt the (H, W) layout.
        return self.__class__(self.data, self.orig_shape, encoding=self.encoding)

    def __repr__(self) -> str:
        return (
            f"DepthMap(shape={tuple(self.data.shape)}, "
            f"range=({self.min:.4g}, {self.max:.4g}), "
            f"orig_shape={self.orig_shape}, encoding={self.encoding!r})"
        )


class AlbedoMap(_TensorPayload):
    """Float32 linear-RGB diffuse reflectance, ``(H, W, 3)`` in ``[0, 1]``.

    Values describe surface colour without illumination. Rendering applies
    the standard sRGB transfer function; the numeric payload remains linear.
    """

    def __init__(self, data: TensorLike, orig_shape=None):
        if not isinstance(data, (torch.Tensor, np.ndarray)):
            raise TypeError("albedo data must be a torch.Tensor or numpy.ndarray")
        if data.ndim != 3 or data.shape[-1] != 3 or min(data.shape[:2]) <= 0:
            raise ValueError("albedo must have positive (H, W, 3) shape")
        if isinstance(data, torch.Tensor):
            data = data.float()
            valid = torch.isfinite(data).all() & ((data >= 0) & (data <= 1)).all()
        else:
            data = np.asarray(data, dtype=np.float32)
            valid = np.isfinite(data).all() and ((data >= 0) & (data <= 1)).all()
        if not bool(valid):
            raise ValueError("albedo must be finite linear RGB in [0, 1]")
        shape = tuple(map(int, data.shape[:2]))
        if orig_shape is not None and tuple(orig_shape) != shape:
            raise ValueError("albedo must match the original image canvas")
        super().__init__(data, shape)

    @property
    def array(self) -> np.ndarray:
        return np.asarray(_numpy(self.data), dtype=np.float32)

    def to_rgb(self) -> np.ndarray:
        linear = self.array
        srgb = np.where(
            linear <= 0.0031308,
            12.92 * linear,
            1.055 * np.power(linear, 1 / 2.4) - 0.055,
        )
        return np.rint(srgb.clip(0, 1) * 255).astype(np.uint8)

    def save(self, path: str | Path) -> None:
        from PIL import Image

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(self.to_rgb()).save(path)

    def __getitem__(self, idx):
        return self.__class__(self.data, self.orig_shape)

    def __len__(self):
        return 1


class EdgeMap(_TensorPayload):
    """Dense edge-probability map for a single image.

    Data is float32 ``(H, W)`` on the original image canvas. ``0`` means
    non-edge and ``1`` means edge. The continuous map is retained so callers
    can choose a threshold appropriate to their dataset or application.
    """

    def __init__(self, data: TensorLike, orig_shape: Tuple[int, int] | None = None):
        if not isinstance(data, (torch.Tensor, np.ndarray)):
            raise TypeError("edge-map data must be a torch.Tensor or numpy.ndarray")
        if data.ndim != 2:
            raise ValueError(
                f"expected (H, W) edge map but got shape {tuple(data.shape)}"
            )
        if int(data.shape[0]) <= 0 or int(data.shape[1]) <= 0:
            raise ValueError("edge-map height and width must be positive")

        if isinstance(data, torch.Tensor):
            data = data.to(dtype=torch.float32)
            if not bool(torch.isfinite(data).all()):
                raise ValueError("edge map contains non-finite values")
            if bool(((data < 0.0) | (data > 1.0)).any()):
                raise ValueError("edge-map values must be in [0, 1]")
        else:
            data = np.asarray(data, dtype=np.float32)
            if not bool(np.isfinite(data).all()):
                raise ValueError("edge map contains non-finite values")
            if bool(np.any((data < 0.0) | (data > 1.0))):
                raise ValueError("edge-map values must be in [0, 1]")

        data_shape = (int(data.shape[0]), int(data.shape[1]))
        if orig_shape is None:
            orig_shape = data_shape
        else:
            orig_shape = (int(orig_shape[0]), int(orig_shape[1]))
            if orig_shape != data_shape:
                raise ValueError(
                    f"edge map shape {data_shape} does not match original "
                    f"image shape {orig_shape}"
                )
        super().__init__(data, orig_shape)

    def binary(self, threshold: float = 0.5) -> TensorLike:
        """Return a boolean edge mask at ``threshold``."""
        threshold = float(threshold)
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError(
                f"edge threshold must be finite and in [0, 1], got {threshold}"
            )
        return self.data >= threshold

    @property
    def array(self) -> np.ndarray:
        """Return the map as an ``(H, W)`` float32 NumPy array."""
        return np.asarray(_numpy(self.data), dtype=np.float32)

    def __getitem__(self, idx):
        # A dense edge map is whole-image data, not an instance collection.
        return self.__class__(self.data, self.orig_shape)

    def __len__(self) -> int:
        return 1

    def __repr__(self) -> str:
        return (
            f"EdgeMap(shape={tuple(self.data.shape)}, "
            f"range=({float(self.data.min()):.4g}, "
            f"{float(self.data.max()):.4g}), orig_shape={self.orig_shape})"
        )


class NormalMap(_TensorPayload):
    """Dense surface-normal field for a single image.

    Data is float32 ``(H, W, 3)`` on the original image canvas in the OpenCV
    camera frame: ``+x`` right, ``+y`` down, and ``+z`` into the scene.
    Normals face the camera, so a fronto-parallel surface is ``(0, 0, -1)``.
    Producers must emit a unit vector at every pixel.
    """

    def __init__(self, data: TensorLike, orig_shape: Tuple[int, int] | None = None):
        if not isinstance(data, (torch.Tensor, np.ndarray)):
            raise TypeError("normal-map data must be a torch.Tensor or numpy.ndarray")
        if data.ndim != 3 or data.shape[-1] != 3:
            raise ValueError(
                f"expected (H, W, 3) normal map but got shape {tuple(data.shape)}"
            )
        if int(data.shape[0]) <= 0 or int(data.shape[1]) <= 0:
            raise ValueError("normal-map height and width must be positive")

        if isinstance(data, torch.Tensor):
            data = data.to(dtype=torch.float32)
        else:
            data = np.asarray(data, dtype=np.float32)

        data_shape = (int(data.shape[0]), int(data.shape[1]))
        if orig_shape is None:
            orig_shape = data_shape
        else:
            orig_shape = (int(orig_shape[0]), int(orig_shape[1]))
            if orig_shape != data_shape:
                raise ValueError(
                    f"normal map shape {data_shape} does not match original "
                    f"image shape {orig_shape}"
                )
        super().__init__(data, orig_shape)

    def assert_normalized(self, atol: float = 1e-4) -> None:
        """Assert that every pixel is finite and unit length within ``atol``."""
        if atol < 0:
            raise ValueError(f"atol must be non-negative, got {atol}")

        if isinstance(self.data, torch.Tensor):
            finite = torch.isfinite(self.data).all(dim=-1)
            if not bool(finite.all()):
                invalid = int((~finite).sum().item())
                raise AssertionError(
                    f"normal map contains {invalid} non-finite pixel(s)"
                )
            norms = torch.linalg.vector_norm(self.data, dim=-1)
            max_error = float((norms - 1.0).abs().max().item())
        else:
            finite = np.isfinite(self.data).all(axis=-1)
            if not bool(finite.all()):
                invalid = int((~finite).sum())
                raise AssertionError(
                    f"normal map contains {invalid} non-finite pixel(s)"
                )
            norms = np.linalg.norm(self.data, axis=-1)
            max_error = float(np.max(np.abs(norms - 1.0)))

        if max_error > atol:
            raise AssertionError(
                f"normal map is not unit-normalized: maximum norm error "
                f"{max_error:.6g} exceeds atol={atol:.6g}"
            )

    def __getitem__(self, idx):
        # A dense normal field is whole-image data, not an instance collection.
        return self.__class__(self.data, self.orig_shape)

    def __len__(self) -> int:
        return 1

    def __repr__(self) -> str:
        return (
            f"NormalMap(shape={tuple(self.data.shape)}, orig_shape={self.orig_shape})"
        )


class RestoredImage(_TensorPayload):
    """Dense restored RGB image for a single input.

    Data shape is ``(H, W, 3)`` uint8 RGB on the original image canvas.
    """

    def __init__(self, data: TensorLike, orig_shape: Tuple[int, int] | None = None):
        if data.ndim != 3 or data.shape[-1] != 3:
            raise ValueError(
                f"expected (H, W, 3) restored RGB image but got shape {tuple(data.shape)}"
            )
        if orig_shape is None:
            orig_shape = (int(data.shape[0]), int(data.shape[1]))
        super().__init__(data, orig_shape)

    @property
    def array(self) -> np.ndarray:
        """Return the raw HWC uint8 RGB ndarray."""

        arr = np.asarray(_numpy(self.data))
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr

    def save(self, path: str | Path) -> None:
        """Write the restored RGB image to disk."""

        from PIL import Image

        path = Path(path)
        if path.parent and path.parent != Path("."):
            path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(self.array, mode="RGB").save(path)

    def __getitem__(self, idx):
        # Instance indexing does not apply to a dense restored image; keep it
        # intact so shared Results slicing paths cannot corrupt the HWC layout.
        return self.__class__(self.data, self.orig_shape)

    def __len__(self) -> int:
        return 1

    def __repr__(self) -> str:
        return (
            f"RestoredImage(shape={tuple(self.data.shape)}, "
            f"orig_shape={self.orig_shape})"
        )


class Matte(_TensorPayload):
    """Dense soft alpha matte for a single image.

    Data shape is ``(H, W)`` float32 in ``[0, 1]`` on the original image canvas.
    ``1`` is fully foreground (opaque), ``0`` is fully background (transparent).
    A soft matte subsumes a hard background-removal mask (threshold at 0.5) and
    carries the anti-aliased edges (hair, fur) that binary masks discard.
    """

    def __init__(self, data: TensorLike, orig_shape: Tuple[int, int] | None = None):
        if data.ndim != 2:
            raise ValueError(
                f"expected (H, W) alpha matte but got shape {tuple(data.shape)}"
            )
        if orig_shape is None:
            orig_shape = (int(data.shape[0]), int(data.shape[1]))
        super().__init__(data, orig_shape)

    @property
    def array(self) -> np.ndarray:
        """Return the raw ``(H, W)`` float32 alpha matte clamped to ``[0, 1]``."""
        arr = np.asarray(_numpy(self.data), dtype=np.float32)
        return np.clip(arr, 0.0, 1.0)

    def __getitem__(self, idx):
        # A dense matte is whole-image, not per-detection; keep it intact so
        # shared Results slicing paths cannot corrupt the (H, W) layout.
        return self.__class__(self.data, self.orig_shape)

    def __len__(self) -> int:
        return 1

    def __repr__(self) -> str:
        return f"Matte(shape={tuple(self.data.shape)}, orig_shape={self.orig_shape})"


class OCRRegions(_TensorPayload):
    """Located text regions with transcripts for a single image.

    ``data`` is an ``(N, 4, 2)`` float array of 4-point polygons in
    original-image pixel coordinates, ordered top-left, top-right,
    bottom-right, bottom-left per region. Regions are in reading order
    (top to bottom, then left to right). ``texts`` is the list of N
    transcripts; ``confidence`` is the per-region recognition score and
    ``det_confidence`` the detection score, both ``(N,)`` float arrays.

    Detection quads are genuine polygons (rotated text), so they do not
    populate ``Results.boxes``; use :attr:`xyxy` for axis-aligned hulls.
    """

    def __init__(
        self,
        data: TensorLike,
        texts: Optional[List[str]] = None,
        confidence: TensorLike | None = None,
        det_confidence: TensorLike | None = None,
        orig_shape: Tuple[int, int] | None = None,
    ):
        if isinstance(data, np.ndarray):
            data = torch.as_tensor(data)
        if data.numel() == 0:
            data = data.reshape(0, 4, 2)
        if data.ndim != 3 or data.shape[-2:] != (4, 2):
            raise ValueError(
                f"expected (N, 4, 2) OCR polygons but got shape {tuple(data.shape)}"
            )
        super().__init__(data, orig_shape)
        n = int(data.shape[0])
        self.texts: List[str] = list(texts) if texts is not None else [""] * n
        if len(self.texts) != n:
            raise ValueError(
                f"expected {n} transcripts to match {n} polygons, got {len(self.texts)}"
            )

        def _as_scores(values):
            if values is None:
                if isinstance(data, torch.Tensor):
                    return torch.zeros(n, dtype=torch.float32)
                return np.zeros(n, dtype=np.float32)
            if isinstance(values, torch.Tensor):
                values = values.reshape(-1).float()
            else:
                values = np.asarray(values, dtype=np.float32).reshape(-1)
            if int(values.shape[0]) != n:
                raise ValueError(
                    f"expected {n} scores to match {n} polygons, got {int(values.shape[0])}"
                )
            return values

        self._conf = _as_scores(confidence)
        self._det_conf = _as_scores(det_confidence)

    @property
    def polygons(self) -> TensorLike:
        return self.data

    @property
    def conf(self) -> TensorLike:
        return self._conf

    @property
    def det_conf(self) -> TensorLike:
        return self._det_conf

    @property
    def xyxy(self) -> TensorLike:
        """Axis-aligned bounding boxes of the polygons, ``(N, 4)``."""
        polys = self.data
        if isinstance(polys, torch.Tensor):
            if len(self) == 0:
                return torch.zeros((0, 4), dtype=torch.float32)
            x = polys[..., 0]
            y = polys[..., 1]
            return torch.stack(
                [
                    x.min(dim=1).values,
                    y.min(dim=1).values,
                    x.max(dim=1).values,
                    y.max(dim=1).values,
                ],
                dim=1,
            )
        if len(self) == 0:
            return np.zeros((0, 4), dtype=np.float32)
        x = polys[..., 0]
        y = polys[..., 1]
        return np.stack(
            [x.min(axis=1), y.min(axis=1), x.max(axis=1), y.max(axis=1)], axis=1
        )

    # texts/scores are extra payload the base _TensorPayload moves (which
    # rebuild via ``self.__class__(data, orig_shape)``) would drop. Override
    # the move/slice methods to carry them through, mirroring
    # PanopticSegmentation.segments_info.
    def to(self, *args, **kwargs):
        return self.__class__(
            _move(self.data, *args, **kwargs),
            self.texts,
            _move(self._conf, *args, **kwargs),
            _move(self._det_conf, *args, **kwargs),
            self.orig_shape,
        )

    def cpu(self):
        return self.__class__(
            _cpu(self.data),
            self.texts,
            _cpu(self._conf),
            _cpu(self._det_conf),
            self.orig_shape,
        )

    def cuda(self):
        return self.__class__(
            _cuda(self.data),
            self.texts,
            _cuda(self._conf),
            _cuda(self._det_conf),
            self.orig_shape,
        )

    def numpy(self):
        return self.__class__(
            _numpy(self.data),
            self.texts,
            _numpy(self._conf),
            _numpy(self._det_conf),
            self.orig_shape,
        )

    def __getitem__(self, idx):
        if isinstance(idx, int):
            indices = [idx]
        elif isinstance(idx, slice):
            indices = list(range(len(self)))[idx]
        else:
            indices = [int(i) for i in np.atleast_1d(np.asarray(idx)).reshape(-1)]
        return self.__class__(
            _slice_first(self.data, idx) if isinstance(idx, int) else self.data[idx],
            [self.texts[i] for i in indices],
            self._conf[indices],
            self._det_conf[indices],
            self.orig_shape,
        )

    def __repr__(self) -> str:
        return (
            f"OCRRegions(n={len(self)}, "
            f"shape={tuple(self.data.shape)}, "
            f"orig_shape={self.orig_shape})"
        )


class OBB(_TensorPayload):
    def __init__(self, data: TensorLike, orig_shape: Tuple[int, int] | None = None):
        if data.ndim == 1:
            data = data[None, :]
        n = data.shape[-1]
        if n not in {7, 8}:
            raise ValueError(
                f"expected 7 or 8 OBB values but got {n}: "
                "xywhr, optional track_id, conf, cls"
            )
        super().__init__(data, orig_shape)

    @property
    def xywhr(self) -> TensorLike:
        return self.data[:, :5]

    @property
    def is_track(self) -> bool:
        return self.data.shape[-1] == 8

    @property
    def id(self) -> TensorLike | None:
        return self.data[:, -3] if self.is_track else None

    @property
    def conf(self) -> TensorLike:
        return self.data[:, -2]

    @property
    def cls(self) -> TensorLike:
        return self.data[:, -1]

    @property
    def xyxyxyxy(self) -> TensorLike:
        box = self.xywhr
        if isinstance(box, torch.Tensor):
            xy = box[:, :2]
            w = box[:, 2] / 2
            h = box[:, 3] / 2
            angle = box[:, 4]
            cos = torch.cos(angle)
            sin = torch.sin(angle)
            corners = torch.stack(
                [
                    torch.stack([-w, -h], dim=1),
                    torch.stack([w, -h], dim=1),
                    torch.stack([w, h], dim=1),
                    torch.stack([-w, h], dim=1),
                ],
                dim=1,
            )
            rot = torch.stack(
                [
                    torch.stack([cos, -sin], dim=1),
                    torch.stack([sin, cos], dim=1),
                ],
                dim=1,
            )
            return torch.matmul(corners, rot.transpose(1, 2)) + xy[:, None, :]

        xy = box[:, :2]
        w = box[:, 2] / 2
        h = box[:, 3] / 2
        angle = box[:, 4]
        cos = np.cos(angle)
        sin = np.sin(angle)
        corners = np.stack(
            [
                np.stack([-w, -h], axis=1),
                np.stack([w, -h], axis=1),
                np.stack([w, h], axis=1),
                np.stack([-w, h], axis=1),
            ],
            axis=1,
        )
        rot = np.stack(
            [
                np.stack([cos, -sin], axis=1),
                np.stack([sin, cos], axis=1),
            ],
            axis=1,
        )
        return np.matmul(corners, np.swapaxes(rot, 1, 2)) + xy[:, None, :]

    @property
    def xyxyxyxyn(self) -> TensorLike:
        if self.orig_shape is None:
            raise ValueError("orig_shape is required for normalized OBB coordinates")
        h, w = self.orig_shape
        corners = self.xyxyxyxy
        if isinstance(corners, torch.Tensor):
            scale = torch.tensor([w, h], dtype=corners.dtype, device=corners.device)
        else:
            scale = np.array([w, h], dtype=corners.dtype)
        return corners / scale

    @property
    def xyxy(self) -> TensorLike:
        corners = self.xyxyxyxy
        x = corners[..., 0]
        y = corners[..., 1]
        if isinstance(corners, torch.Tensor):
            return torch.stack(
                [
                    x.min(dim=1).values,
                    y.min(dim=1).values,
                    x.max(dim=1).values,
                    y.max(dim=1).values,
                ],
                dim=1,
            )
        return np.stack(
            [x.min(axis=1), y.min(axis=1), x.max(axis=1), y.max(axis=1)], axis=1
        )


class Gaze(_TensorPayload):
    """Per-face gaze angles in radians.

    Data shape: (N, 2) where column 0 is pitch and column 1 is yaw.
    Aligned row-by-row with the parent Results.boxes (face boxes).
    The L2CS convention is used: positive yaw rotates the gaze toward
    the subject's left, positive pitch rotates it downward.
    """

    def __init__(self, data: TensorLike, orig_shape: Tuple[int, int] | None = None):
        if data.ndim == 1:
            if isinstance(data, torch.Tensor):
                data = data.unsqueeze(0)
            else:
                data = data[None, :]
        if data.shape[-1] != 2:
            raise ValueError(
                f"expected (N, 2) pitch/yaw, got shape {tuple(data.shape)}"
            )
        super().__init__(data, orig_shape)

    @property
    def pitch(self) -> TensorLike:
        return self.data[..., 0]

    @property
    def yaw(self) -> TensorLike:
        return self.data[..., 1]

    @property
    def pitch_deg(self) -> TensorLike:
        return self.pitch * (180.0 / math.pi)

    @property
    def yaw_deg(self) -> TensorLike:
        return self.yaw * (180.0 / math.pi)

    @property
    def direction_3d(self) -> TensorLike:
        """Unit gaze direction in the camera frame: (N, 3), columns (x, y, z).

        Matches upstream L2CS-Net ``gazeto3d``: (-cos(p)*sin(y), -sin(p), -cos(p)*cos(y)).
        """
        p, y = self.pitch, self.yaw
        if isinstance(self.data, torch.Tensor):
            cos_p, sin_p = torch.cos(p), torch.sin(p)
            cos_y, sin_y = torch.cos(y), torch.sin(y)
            return torch.stack([-cos_p * sin_y, -sin_p, -cos_p * cos_y], dim=-1)
        cos_p, sin_p = np.cos(p), np.sin(p)
        cos_y, sin_y = np.cos(y), np.sin(y)
        return np.stack([-cos_p * sin_y, -sin_p, -cos_p * cos_y], axis=-1)

    def __repr__(self) -> str:
        return (
            f"Gaze(n={len(self)}, "
            f"shape={tuple(self.data.shape)}, "
            f"orig_shape={self.orig_shape})"
        )


class Embeddings(_TensorPayload):
    """L2-normalized vectors produced by the generic ``embed`` task.

    Data always has shape ``(N, D)``. A whole-image result carries one row and
    no boxes; region embeddings are row-aligned with ``Results.boxes``. Each
    row is float32 and L2-normalized by its inference path, so cosine
    similarity is a dot product.
    """

    def __init__(self, data: TensorLike, orig_shape: Tuple[int, int] | None = None):
        if data.ndim == 1:
            if isinstance(data, torch.Tensor):
                data = data.unsqueeze(0)
            else:
                data = data[None, :]
        if data.ndim != 2:
            raise ValueError(
                f"expected (N, D) embeddings, got shape {tuple(data.shape)}"
            )
        super().__init__(data, orig_shape)

    @property
    def dim(self) -> int:
        return int(self.data.shape[-1])

    @property
    def normalized(self) -> TensorLike:
        """Defensive re-L2-normalization of each row."""
        d = self.data
        if isinstance(d, torch.Tensor):
            return d / d.norm(dim=-1, keepdim=True).clamp_min(1e-10)
        norm = np.linalg.norm(d, axis=-1, keepdims=True)
        return d / np.clip(norm, 1e-10, None)

    def similarity(self, other: "Embeddings | TensorLike") -> TensorLike:
        """Cosine similarity of these rows against ``other``.

        Returns ``(N, M)`` for an ``(M, D)`` gallery (or another Embeddings),
        or ``(N,)`` for a single ``(D,)`` vector.
        """
        a = self.normalized
        b = other.normalized if isinstance(other, Embeddings) else other
        single = getattr(b, "ndim", 2) == 1
        if isinstance(a, torch.Tensor):
            b = (
                b
                if isinstance(b, torch.Tensor)
                else torch.as_tensor(b, dtype=a.dtype, device=a.device)
            )
            b = b.reshape(1, -1) if single else b
            b = b / b.norm(dim=-1, keepdim=True).clamp_min(1e-10)
            sim = a @ b.T
        else:
            b = b if isinstance(b, np.ndarray) else _numpy(b)
            b = b.reshape(1, -1) if single else b
            b = b / np.clip(np.linalg.norm(b, axis=-1, keepdims=True), 1e-10, None)
            sim = a @ b.T
        return sim[:, 0] if single else sim

    def verify(self, i: int, j: int, threshold: float = 0.4) -> bool:
        """Whether rows ``i`` and ``j`` meet a cosine-similarity threshold."""
        sim = self.similarity(self.data[j])
        return bool(float(sim[i]) >= threshold)

    def __repr__(self) -> str:
        return (
            f"Embeddings(n={len(self)}, dim={self.dim}, shape={tuple(self.data.shape)})"
        )


class Identities:
    """Named gallery matches row-aligned with ``Results.embeddings``.

    Produced by the ``embed`` task when a ``Gallery`` is supplied. ``name`` is
    ``None`` below the match threshold (*unknown*); the nearest below-threshold
    name is never guessed.
    """

    def __init__(
        self,
        names: List[Optional[str]],
        scores: TensorLike,
    ):
        self._names = list(names)
        self._scores = np.asarray(_numpy(scores), dtype=np.float32).reshape(-1)
        if len(self._names) != self._scores.shape[0]:
            raise ValueError(
                f"names ({len(self._names)}) and scores "
                f"({self._scores.shape[0]}) must be row-aligned"
            )

    @property
    def name(self) -> List[Optional[str]]:
        """Matched name per embedding row, ``None`` for unknown."""
        return list(self._names)

    @property
    def score(self) -> np.ndarray:
        """Best gallery cosine similarity per embedding row."""
        return self._scores

    @property
    def data(self) -> List[Tuple[Optional[str], float]]:
        return [(n, float(s)) for n, s in zip(self._names, self._scores)]

    # Container protocol used by Results._apply — identity labels are
    # device-less, so tensor movement is a no-op.
    def to(self, *args, **kwargs) -> "Identities":
        return self

    def cpu(self) -> "Identities":
        return self

    def cuda(self) -> "Identities":
        return self

    def numpy(self) -> "Identities":
        return self

    def __getitem__(self, idx) -> "Identities":
        if isinstance(idx, (int, np.integer)):
            return Identities([self._names[idx]], self._scores[idx : idx + 1])
        if isinstance(idx, slice):
            return Identities(self._names[idx], self._scores[idx])
        idx = np.asarray(idx)
        if idx.dtype == bool:
            idx = np.flatnonzero(idx)
        return Identities([self._names[i] for i in idx], self._scores[idx])

    def __len__(self) -> int:
        return len(self._names)

    def __repr__(self) -> str:
        known = sum(1 for n in self._names if n is not None)
        return f"Identities(n={len(self)}, known={known})"


class Meshes:
    """Parametric human body meshes for a single image.

    Rows are aligned with the parent ``Results.boxes`` (person boxes), the same
    contract ``Keypoints`` follows for the pose task: row ``i`` of every tensor
    here describes the person in box ``i``.

    Everything is expressed in the camera frame of the original image.
    ``transl`` is metric (meters) with +z pointing away from the camera;
    ``vertices`` and ``joints3d`` are metric and already include ``transl``;
    ``joints2d`` is in pixels on the original image canvas, not on the crop the
    network actually saw. A world/gravity frame is deliberately absent in this
    version, so no field here silently means "world".

    Parameter layouts differ between body models, so nothing about the shapes
    is hard-coded: ``body_model`` names the parameterization and the counts are
    read back from the tensors. For ``"mhr"`` (Momentum Human Rig), rotations
    are Euler angles in radians rather than axis-angle, ``body_pose`` is a flat
    per-joint parameter vector rather than one triplet per joint (rig joints
    carry different degrees of freedom), and ``betas`` are identity blendshape
    coefficients. Model-specific extras such as skeleton scale, hand pose and
    facial expression live in ``extras``.
    """

    def __init__(
        self,
        global_orient: TensorLike,
        body_pose: TensorLike,
        betas: TensorLike,
        transl: TensorLike,
        *,
        body_model: str,
        vertices: TensorLike | None = None,
        faces: TensorLike | None = None,
        joints3d: TensorLike | None = None,
        joints2d: TensorLike | None = None,
        conf: TensorLike | None = None,
        focal_length: TensorLike | None = None,
        extras: Optional[Dict[str, TensorLike]] = None,
        orig_shape: Tuple[int, int] | None = None,
    ):
        self.global_orient = global_orient
        self.body_pose = body_pose
        self.betas = betas
        self.transl = transl
        self.body_model = str(body_model)
        self.vertices = vertices
        # Topology is shared by every person in the image, so it is stored once
        # and never sliced per row.
        self.faces = faces
        self.joints3d = joints3d
        self.joints2d = joints2d
        self.conf = conf
        self.focal_length = focal_length
        self.extras = dict(extras) if extras else {}
        self.orig_shape = orig_shape

    # Per-row tensors, in the order they are rebuilt by _rebuild().
    _ROW_FIELDS = (
        "global_orient",
        "body_pose",
        "betas",
        "transl",
        "vertices",
        "joints3d",
        "joints2d",
        "conf",
        "focal_length",
    )

    def _rebuild(self, fn, shared_fn=None) -> "Meshes":
        """Rebuild with ``fn`` over per-row tensors.

        ``shared_fn`` handles the shared face topology; it follows ``fn`` for
        device and dtype moves but is the identity for row slicing, where
        selecting a person must not touch the mesh connectivity.
        """
        if shared_fn is None:
            shared_fn = fn
        values = {name: fn(getattr(self, name)) for name in self._ROW_FIELDS}
        return Meshes(
            values.pop("global_orient"),
            values.pop("body_pose"),
            values.pop("betas"),
            values.pop("transl"),
            body_model=self.body_model,
            faces=shared_fn(self.faces),
            extras={k: fn(v) for k, v in self.extras.items()},
            orig_shape=self.orig_shape,
            **values,
        )

    @property
    def num_vertices(self) -> int:
        return 0 if self.vertices is None else int(self.vertices.shape[1])

    @property
    def num_joints(self) -> int:
        return 0 if self.joints3d is None else int(self.joints3d.shape[1])

    @property
    def num_betas(self) -> int:
        return int(self.betas.shape[-1])

    @property
    def has_vertices(self) -> bool:
        return self.vertices is not None

    @property
    def params(self) -> Dict[str, TensorLike]:
        """The parametric core, shaped to splat into a body-model forward."""
        core = {
            "global_orient": self.global_orient,
            "body_pose": self.body_pose,
            "betas": self.betas,
            "transl": self.transl,
        }
        core.update(self.extras)
        return core

    def save_obj(self, path: str | Path, index: int = 0) -> None:
        """Write one person's mesh to a Wavefront OBJ file."""
        if self.vertices is None or self.faces is None:
            raise ValueError(
                "This result carries no mesh geometry, only parameters, so it "
                "cannot be written as OBJ."
            )
        if not 0 <= index < len(self):
            raise IndexError(f"index {index} is out of range for {len(self)} mesh(es)")
        verts = np.asarray(_numpy(self.vertices))[index]
        faces = np.asarray(_numpy(self.faces))

        path = Path(path)
        if path.parent and path.parent != Path("."):
            path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(f"# LibreYOLO body mesh ({self.body_model})\n")
            for v in verts:
                fh.write(f"v {float(v[0]):.6f} {float(v[1]):.6f} {float(v[2]):.6f}\n")
            # OBJ vertex indices are 1-based.
            for f in faces:
                fh.write(f"f {int(f[0]) + 1} {int(f[1]) + 1} {int(f[2]) + 1}\n")

    def to(self, *args, **kwargs) -> "Meshes":
        return self._rebuild(lambda d: _move(d, *args, **kwargs))

    def cpu(self) -> "Meshes":
        return self._rebuild(_cpu)

    def cuda(self) -> "Meshes":
        return self._rebuild(_cuda)

    def numpy(self) -> "Meshes":
        return self._rebuild(_numpy)

    def __getitem__(self, idx) -> "Meshes":
        return self._rebuild(lambda d: _slice_first(d, idx), shared_fn=lambda d: d)

    def __len__(self) -> int:
        return int(self.global_orient.shape[0])

    def __repr__(self) -> str:
        return (
            f"Meshes(n={len(self)}, body_model='{self.body_model}', "
            f"betas={self.num_betas}, vertices={self.num_vertices}, "
            f"joints={self.num_joints}, orig_shape={self.orig_shape})"
        )


class Results:
    """Single-image result with flat detection/segmentation slots."""

    _keys = (
        "boxes",
        "masks",
        "probs",
        "keypoints",
        "obb",
        "gaze",
        "points",
        "semantic_mask",
        "panoptic",
        "depth_map",
        "normal_map",
        "albedo",
        "edges",
        "restored",
        "matte",
        "ocr",
        "embeddings",
        "identities",
        "meshes",
        "boxes3d",
    )

    def __init__(
        self,
        boxes: Optional[Boxes],
        orig_shape: Tuple[int, int],
        path: Optional[str] = None,
        names: Optional[Dict[int, str]] = None,
        masks: Optional[Masks] = None,
        keypoints: Optional[Keypoints] = None,
        probs: Optional[Probs] = None,
        obb: Optional[OBB] = None,
        gaze: Optional[Gaze] = None,
        points: Optional[Points] = None,
        semantic_mask: Optional[SemanticMask] = None,
        depth_map: Optional[DepthMap] = None,
        restored: Optional[RestoredImage] = None,
        speed: Optional[Dict[str, float]] = None,
        track_id: Optional[TensorLike] = None,
        frame_idx: Optional[int] = None,
        # New parameters go after the complete v1.3 signature so v1.3-era
        # positional calls keep binding to the same parameters.
        panoptic: Optional[PanopticSegmentation] = None,
        matte: Optional[Matte] = None,
        ocr: Optional[OCRRegions] = None,
        restore_scale: int = 1,
        embeddings: Optional[Embeddings] = None,
        identities: Optional[Identities] = None,
        meshes: Optional[Meshes] = None,
        normal_map: Optional[NormalMap] = None,
        edges: Optional[EdgeMap] = None,
        boxes3d: Optional[Boxes3D] = None,
        albedo: Optional[AlbedoMap] = None,
    ):
        if boxes is not None and boxes.orig_shape is None:
            boxes = boxes.with_orig_shape(orig_shape)
        if boxes is not None and track_id is not None:
            boxes = boxes.with_id(track_id)
        if points is not None and points.orig_shape is None:
            points = Points(points.data, orig_shape)
        if depth_map is not None and depth_map.orig_shape is None:
            depth_map = DepthMap(depth_map.data, orig_shape, encoding=depth_map.encoding)
        if albedo is not None and albedo.orig_shape != tuple(orig_shape):
            raise ValueError("albedo must match Results.orig_shape")
        if normal_map is not None and normal_map.orig_shape != tuple(orig_shape):
            normal_map = NormalMap(normal_map.data, orig_shape)
        if edges is not None and edges.orig_shape != tuple(orig_shape):
            edges = EdgeMap(edges.data, orig_shape)
        if restored is not None and restored.orig_shape is None:
            restored = RestoredImage(restored.data, orig_shape)
        if matte is not None and matte.orig_shape is None:
            matte = Matte(matte.data, orig_shape)
        if ocr is not None and ocr.orig_shape is None:
            ocr = OCRRegions(ocr.data, ocr.texts, ocr.conf, ocr.det_conf, orig_shape)

        if boxes3d is not None:
            boxes3d._validate_alignment(boxes, orig_shape)
        self.boxes = boxes
        self.masks = masks
        self.keypoints = keypoints
        self.probs = probs
        self.obb = obb
        self.gaze = gaze
        self.points = points
        self.semantic_mask = semantic_mask
        self.panoptic = panoptic
        self.depth_map = depth_map
        self.normal_map = normal_map
        self.albedo = albedo
        self.edges = edges
        self.restored = restored
        self.matte = matte
        self.ocr = ocr
        self.meshes = meshes
        self.boxes3d = boxes3d
        # Integer upscale factor of a restore/super-resolution result: the
        # restored canvas is ``restore_scale`` times the input. 1 for
        # deblur/denoise and every non-restore task.
        self.restore_scale = int(restore_scale) if restore_scale else 1
        self.embeddings = embeddings
        self.identities = identities
        self.orig_shape = orig_shape
        self.path = path
        self.names = names or {}
        self.speed = speed or {}
        self.track_id = (
            track_id if track_id is not None else (boxes.id if boxes else None)
        )
        self.frame_idx = frame_idx

    def _new(self, **overrides) -> "Results":
        data = {
            "boxes": self.boxes,
            "orig_shape": self.orig_shape,
            "path": self.path,
            "names": self.names,
            "masks": self.masks,
            "keypoints": self.keypoints,
            "probs": self.probs,
            "obb": self.obb,
            "gaze": self.gaze,
            "points": self.points,
            "semantic_mask": self.semantic_mask,
            "panoptic": self.panoptic,
            "depth_map": self.depth_map,
            "normal_map": self.normal_map,
            "albedo": self.albedo,
            "edges": self.edges,
            "restored": self.restored,
            "matte": self.matte,
            "ocr": self.ocr,
            "meshes": self.meshes,
            "boxes3d": self.boxes3d,
            "restore_scale": self.restore_scale,
            "embeddings": self.embeddings,
            "identities": self.identities,
            "speed": dict(self.speed),
            "track_id": self.track_id,
            "frame_idx": self.frame_idx,
        }
        data.update(overrides)
        return Results(**data)

    def to(self, *args, **kwargs) -> "Results":
        return self._apply("to", *args, **kwargs)

    def cpu(self) -> "Results":
        return self._apply("cpu")

    def cuda(self) -> "Results":
        return self.to("cuda")

    def numpy(self) -> "Results":
        return self._apply("numpy")

    def _apply(self, method: str, *args, **kwargs) -> "Results":
        overrides = {}
        for key in self._keys:
            value = getattr(self, key)
            overrides[key] = (
                getattr(value, method)(*args, **kwargs) if value is not None else None
            )

        if method == "cpu":
            overrides["track_id"] = _cpu(self.track_id)
        elif method == "numpy":
            overrides["track_id"] = _numpy(self.track_id)
        elif method == "to":
            overrides["track_id"] = _move(self.track_id, *args, **kwargs)
        elif method == "__getitem__":
            overrides["track_id"] = _slice_first(self.track_id, args[0])

        return self._new(**overrides)

    def _select(self, idx) -> "Results":
        return self._apply("__getitem__", idx)

    def __getitem__(self, idx) -> "Results":
        return self._select(idx)

    def update(
        self,
        boxes: Optional[Boxes] = None,
        masks: Optional[Masks] = None,
        probs: Optional[Probs] = None,
        keypoints: Optional[Keypoints] = None,
        obb: Optional[OBB] = None,
        gaze: Optional[Gaze] = None,
        points: Optional[Points] = None,
        semantic_mask: Optional[SemanticMask] = None,
        depth_map: Optional[DepthMap] = None,
        restored: Optional[RestoredImage] = None,
        track_id: Optional[TensorLike] = None,
        # New parameters go after the complete v1.3 signature so v1.3-era
        # positional calls keep binding to the same parameters.
        panoptic: Optional[PanopticSegmentation] = None,
        matte: Optional[Matte] = None,
        ocr: Optional[OCRRegions] = None,
        restore_scale: Optional[int] = None,
        embeddings: Optional[Embeddings] = None,
        identities: Optional[Identities] = None,
        meshes: Optional[Meshes] = None,
        normal_map: Optional[NormalMap] = None,
        edges: Optional[EdgeMap] = None,
        boxes3d: Optional[Boxes3D] = None,
        albedo: Optional[AlbedoMap] = None,
    ) -> "Results":
        aligned_cuboids = boxes3d if boxes3d is not None else self.boxes3d
        if aligned_cuboids is not None:
            aligned_cuboids._validate_alignment(
                boxes if boxes is not None else self.boxes, self.orig_shape
            )
        if boxes is not None:
            self.boxes = boxes.with_orig_shape(self.orig_shape)
        if masks is not None:
            self.masks = masks
        if probs is not None:
            self.probs = probs
        if keypoints is not None:
            self.keypoints = keypoints
        if obb is not None:
            self.obb = obb
        if gaze is not None:
            self.gaze = gaze
        if points is not None:
            self.points = (
                points
                if points.orig_shape is not None
                else Points(points.data, self.orig_shape)
            )
        if semantic_mask is not None:
            self.semantic_mask = semantic_mask
        if panoptic is not None:
            self.panoptic = panoptic
        if depth_map is not None:
            self.depth_map = depth_map
        if albedo is not None:
            if albedo.orig_shape != tuple(self.orig_shape):
                raise ValueError("albedo must match Results.orig_shape")
            self.albedo = albedo
        if normal_map is not None:
            self.normal_map = (
                normal_map
                if normal_map.orig_shape == tuple(self.orig_shape)
                else NormalMap(normal_map.data, self.orig_shape)
            )
        if edges is not None:
            self.edges = (
                edges
                if edges.orig_shape == tuple(self.orig_shape)
                else EdgeMap(edges.data, self.orig_shape)
            )
        if restored is not None:
            self.restored = restored
        if boxes3d is not None:
            self.boxes3d = boxes3d
        if meshes is not None:
            self.meshes = meshes
        if matte is not None:
            self.matte = (
                matte
                if matte.orig_shape is not None
                else Matte(matte.data, self.orig_shape)
            )
        if ocr is not None:
            self.ocr = (
                ocr
                if ocr.orig_shape is not None
                else OCRRegions(
                    ocr.data, ocr.texts, ocr.conf, ocr.det_conf, self.orig_shape
                )
            )
        if restore_scale is not None:
            self.restore_scale = int(restore_scale) if restore_scale else 1
        if embeddings is not None:
            self.embeddings = embeddings
        if identities is not None:
            self.identities = identities
        if track_id is not None:
            self.track_id = track_id
            if self.boxes is not None:
                self.boxes = self.boxes.with_id(track_id)
        return self

    @property
    def normals(self) -> Optional[NormalMap]:
        """Alias for :attr:`normal_map`, matching the plural dense-task API."""
        return self.normal_map

    @normals.setter
    def normals(self, value: Optional[NormalMap]) -> None:
        self.normal_map = value

    def plot(self, image=None):
        """Render dense outputs or calibrated 3D cuboids."""
        if self.albedo is not None:
            from PIL import Image

            return Image.fromarray(self.albedo.to_rgb())
        if self.depth_map is not None and self.boxes3d is None:
            from PIL import Image
            from .drawing import draw_depth_map

            data = _numpy(self.depth_map.data)
            if not self.depth_map.near_is_high:
                data = -data
            h, w = self.orig_shape
            return draw_depth_map(Image.new("RGB", (w, h)), data)
        if self.boxes3d is not None:
            from PIL import Image
            from .drawing import draw_boxes3d

            canvas = Image.fromarray(self._source_rgb(image, self.orig_shape))
            return draw_boxes3d(canvas, self.boxes3d)
        if self.normal_map is None and self.edges is None:
            raise NotImplementedError(
                "Results.plot() supports depth, normal, albedo, edge, and calibrated 3D results."
            )

        from PIL import Image

        h, w = self.orig_shape
        if self.edges is not None:
            from .drawing import draw_edge_map

            return draw_edge_map(Image.new("RGB", (w, h)), self.edges.array)

        from .drawing import draw_normal_map

        canvas = Image.new("RGB", (w, h))
        return draw_normal_map(canvas, _numpy(self.normal_map.data))

    def cutout(self, image: Any = None) -> np.ndarray:
        """Return an RGBA ``(H, W, 4)`` uint8 cutout: source RGB + matte alpha.

        The alpha channel is the soft matte scaled to ``[0, 255]``. The RGB is
        taken from ``image`` when given (a PIL image or ``HxWx3`` array), else
        reloaded from ``self.path``. Only valid for matte results.
        """
        if self.matte is None:
            raise ValueError(
                "cutout() is only defined for matte results (Results.matte is None)."
            )
        alpha = self.matte.array  # (H, W) float32 in [0, 1]
        h, w = alpha.shape
        rgb = self._source_rgb(image, (h, w))
        alpha_u8 = np.rint(alpha * 255.0).astype(np.uint8)
        return np.dstack([rgb, alpha_u8])

    def _source_rgb(self, image: Any, hw: Tuple[int, int]) -> np.ndarray:
        """Load the source image as an ``HxWx3`` uint8 RGB array on the matte canvas."""
        from PIL import Image

        h, w = hw
        if image is None:
            if not self.path:
                raise ValueError(
                    "cutout()/save() needs the source image but Results.path is unset; "
                    "pass image=<PIL.Image or HxWx3 array>."
                )
            rgb = np.asarray(Image.open(self.path).convert("RGB"))
        elif isinstance(image, Image.Image):
            rgb = np.asarray(image.convert("RGB"))
        else:
            rgb = np.asarray(image)
            if rgb.ndim == 2:
                rgb = np.stack([rgb] * 3, axis=-1)
            if rgb.shape[-1] == 4:
                rgb = rgb[..., :3]
        if rgb.shape[:2] != (h, w):
            rgb = np.asarray(
                Image.fromarray(rgb.astype(np.uint8)).resize((w, h), Image.BILINEAR)
            )
        return rgb.astype(np.uint8)

    def save(self, path: str, image: Any = None) -> str:
        """Save a matte result as a transparent-background RGBA PNG cutout.

        Returns the written path. Requires the source image (via ``image`` or
        ``self.path``).
        """
        from PIL import Image

        if self.matte is None:
            raise NotImplementedError(
                "Results.save() writes a transparent-PNG cutout and is defined for "
                "matte results only. Use result.plot()/CLI --save for other tasks."
            )
        rgba = self.cutout(image=image)
        out = Path(path)
        if out.parent and str(out.parent) not in (".", ""):
            out.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgba, mode="RGBA").save(out)
        return str(out)

    def summary(
        self,
        normalize: bool = False,
        decimals: int = 5,
        embeddings: bool = False,
    ) -> List[Dict[str, Any]]:
        if self.boxes is None:
            if self.embeddings is not None:
                emb = (
                    self.embeddings.numpy()
                    if isinstance(self.embeddings.data, torch.Tensor)
                    else self.embeddings
                )
                rows = []
                for i in range(len(emb)):
                    row = {"embedding_dim": int(emb.dim)}
                    if embeddings:
                        row["embedding"] = [
                            round(float(value), decimals) for value in emb.data[i]
                        ]
                    if self.identities is not None and i < len(self.identities):
                        row["identity"] = self.identities.name[i]
                        row["identity_score"] = round(
                            float(self.identities.score[i]), decimals
                        )
                    rows.append(row)
                return rows
            if self.ocr is not None:
                ocr_np = self.ocr.numpy()
                h, w = self.orig_shape
                rows = []
                for i in range(len(ocr_np)):
                    polygon = np.asarray(ocr_np.data[i], dtype=float)
                    if normalize:
                        polygon = polygon / np.array([w, h], dtype=float)
                    rows.append(
                        {
                            "name": "text",
                            "text": ocr_np.texts[i],
                            "confidence": round(float(ocr_np.conf[i]), decimals),
                            "det_confidence": round(
                                float(ocr_np.det_conf[i]), decimals
                            ),
                            "polygon": {
                                "x": [round(float(x), decimals) for x in polygon[:, 0]],
                                "y": [round(float(y), decimals) for y in polygon[:, 1]],
                            },
                        }
                    )
                return rows
            if self.points is not None:
                points_np = self.points.numpy()
                xy_values = points_np.xyn if normalize else points_np.xy
                rows = []
                for i in range(len(points_np)):
                    cls_id = int(points_np.cls[i])
                    rows.append(
                        {
                            "name": self.names.get(cls_id, str(cls_id)),
                            "class": cls_id,
                            "confidence": round(float(points_np.conf[i]), decimals),
                            "point": {
                                "x": round(float(xy_values[i, 0]), decimals),
                                "y": round(float(xy_values[i, 1]), decimals),
                            },
                        }
                    )
                return rows
            if self.panoptic is not None:
                pan_np = _numpy(self.panoptic.data)
                total = int(pan_np.size)
                rows = []
                for seg in self.panoptic.segments_info:
                    cat_id = int(seg["category_id"])
                    count = int((pan_np == int(seg["id"])).sum())
                    row = {
                        "name": self.names.get(cat_id, str(cat_id)),
                        "class": cat_id,
                        "segment_id": int(seg["id"]),
                        "isthing": bool(seg.get("isthing", True)),
                        "pixel_count": count,
                        "pixel_fraction": round(count / total, decimals),
                    }
                    if "score" in seg:
                        row["confidence"] = round(float(seg["score"]), decimals)
                    rows.append(row)
                return rows
            if self.semantic_mask is not None:
                mask_np = _numpy(self.semantic_mask.data)
                total = int(mask_np.size)
                rows = []
                for cls_id in self.semantic_mask.classes:
                    count = int((mask_np == cls_id).sum())
                    rows.append(
                        {
                            "name": self.names.get(cls_id, str(cls_id)),
                            "class": cls_id,
                            "pixel_count": count,
                            "pixel_fraction": round(count / total, decimals),
                        }
                    )
                return rows
            if self.depth_map is not None:
                return [
                    {
                        "name": "depth_map",
                        "min": round(self.depth_map.min, decimals),
                        "max": round(self.depth_map.max, decimals),
                        "mean": round(self.depth_map.mean, decimals),
                        **({"encoding": self.depth_map.encoding}
                           if self.depth_map.encoding != "inverse_depth" else {}),
                    }
                ]
            if self.albedo is not None:
                return [{"name": "albedo", "shape": list(self.albedo.data.shape),
                         "color_space": "linear_rgb"}]
            if self.normal_map is not None:
                h, w = self.normal_map.orig_shape
                return [
                    {
                        "name": "normal_map",
                        "shape": [int(h), int(w), 3],
                        "frame": "opencv",
                        "orientation": "camera-facing",
                    }
                ]
            if self.edges is not None:
                h, w = self.edges.orig_shape
                edge_values = self.edges.array
                return [
                    {
                        "name": "edges",
                        "shape": [int(h), int(w)],
                        "min": round(float(edge_values.min()), decimals),
                        "max": round(float(edge_values.max()), decimals),
                        "mean": round(float(edge_values.mean()), decimals),
                    }
                ]
            if self.restored is not None:
                h, w = self.restored.array.shape[:2]
                return [
                    {
                        "name": "restored",
                        "shape": [int(h), int(w), 3],
                        "scale": int(self.restore_scale),
                    }
                ]
            if self.matte is not None:
                matte_np = self.matte.array
                h, w = matte_np.shape[:2]
                fg = float((matte_np >= 0.5).mean())
                return [
                    {
                        "name": "matte",
                        "shape": [int(h), int(w)],
                        "coverage": round(fg, decimals),
                    }
                ]
            if self.probs is None:
                return []
            probs_np = _numpy(self.probs.data)
            rows = []
            for cls_id in self.probs.top5:
                rows.append(
                    {
                        "name": self.names.get(cls_id, str(cls_id)),
                        "class": int(cls_id),
                        "confidence": round(float(probs_np[cls_id]), decimals),
                    }
                )
            return rows

        boxes_np = self.boxes.numpy()
        obb_np = None
        if self.obb is not None:
            obb_np = (
                self.obb.numpy()
                if isinstance(self.obb.data, torch.Tensor)
                else self.obb
            )
        # Converted once rather than per row: mesh payloads carry vertex arrays
        # large enough that repeating the conversion per person is wasteful.
        meshes_np = self.meshes.numpy() if self.meshes is not None else None
        cuboids = self.boxes3d.numpy() if self.boxes3d is not None else None
        track_ids = _numpy(self.track_id)
        rows = []
        for i in range(len(boxes_np)):
            cls_id = int(boxes_np.cls[i])
            box_values = boxes_np.xyxyn[i] if normalize else boxes_np.xyxy[i]
            row = {
                "name": self.names.get(cls_id, str(cls_id)),
                "class": cls_id,
                "confidence": round(float(boxes_np.conf[i]), decimals),
                "box": {
                    "x1": round(float(box_values[0]), decimals),
                    "y1": round(float(box_values[1]), decimals),
                    "x2": round(float(box_values[2]), decimals),
                    "y2": round(float(box_values[3]), decimals),
                },
            }
            if cuboids is not None and i < len(cuboids):
                row["box3d"] = {
                    "center": np.round(cuboids.xyz[i], decimals).tolist(),
                    "dimensions": np.round(cuboids.dimensions[i], decimals).tolist(),
                    "quaternion_wxyz": np.round(cuboids.quaternion[i], decimals).tolist(),
                    "confidence_2d": round(float(cuboids.conf2d[i]), decimals),
                    "confidence_3d": round(float(cuboids.conf3d[i]), decimals),
                    "coordinate_frame": "camera",
                    "units": "metres",
                }
            if obb_np is not None and i < len(obb_np):
                xywhr = np.asarray(obb_np.xywhr[i], dtype=float).copy()
                corners = np.asarray(
                    obb_np.xyxyxyxyn[i] if normalize else obb_np.xyxyxyxy[i],
                    dtype=float,
                )
                if normalize:
                    h, w = self.orig_shape
                    xywhr[0] /= w
                    xywhr[1] /= h
                    xywhr[2] /= w
                    xywhr[3] /= h
                row["obb"] = {
                    "x_center": round(float(xywhr[0]), decimals),
                    "y_center": round(float(xywhr[1]), decimals),
                    "width": round(float(xywhr[2]), decimals),
                    "height": round(float(xywhr[3]), decimals),
                    "rotation": round(float(xywhr[4]), decimals),
                }
                row["corners"] = {
                    "x": [round(float(x), decimals) for x in corners[:, 0]],
                    "y": [round(float(y), decimals) for y in corners[:, 1]],
                }
            if self.masks is not None:
                segment = self.masks.xyn[i] if normalize else self.masks.xy[i]
                row["segments"] = {
                    "x": [round(float(x), decimals) for x in segment[:, 0]],
                    "y": [round(float(y), decimals) for y in segment[:, 1]],
                }
            if self.gaze is not None and i < len(self.gaze):
                gaze_np = (
                    self.gaze.numpy()
                    if isinstance(self.gaze.data, torch.Tensor)
                    else self.gaze
                )
                row["gaze"] = {
                    "pitch_rad": round(float(gaze_np.data[i, 0]), decimals),
                    "yaw_rad": round(float(gaze_np.data[i, 1]), decimals),
                    "pitch_deg": round(
                        float(gaze_np.data[i, 0]) * 180.0 / math.pi, decimals
                    ),
                    "yaw_deg": round(
                        float(gaze_np.data[i, 1]) * 180.0 / math.pi, decimals
                    ),
                }
            if self.embeddings is not None and i < len(self.embeddings):
                emb = (
                    self.embeddings.numpy()
                    if isinstance(self.embeddings.data, torch.Tensor)
                    else self.embeddings
                )
                # A 512-float vector is ~2 KB/face — omit it from summaries by
                # default and surface only its dimension; opt in with embeddings=True.
                row["embedding_dim"] = int(emb.dim)
                if embeddings:
                    row["embedding"] = [round(float(v), decimals) for v in emb.data[i]]
            if self.identities is not None and i < len(self.identities):
                row["identity"] = self.identities.name[i]
                row["identity_score"] = round(float(self.identities.score[i]), decimals)
            if meshes_np is not None and i < len(meshes_np):
                # Vertices are deliberately omitted: tens of thousands of
                # coordinates per person is not something to hand back as JSON.
                # Use ``result.meshes.vertices`` or ``save_obj`` for geometry.
                mesh_row = {
                    "body_model": meshes_np.body_model,
                    "global_orient": [
                        round(float(v), decimals) for v in meshes_np.global_orient[i]
                    ],
                    "transl": [round(float(v), decimals) for v in meshes_np.transl[i]],
                    "betas": [round(float(v), decimals) for v in meshes_np.betas[i]],
                    "num_vertices": meshes_np.num_vertices,
                }
                if meshes_np.conf is not None:
                    mesh_row["confidence"] = round(float(meshes_np.conf[i]), decimals)
                if meshes_np.focal_length is not None:
                    mesh_row["focal_length"] = round(
                        float(np.asarray(meshes_np.focal_length[i]).reshape(-1)[0]),
                        decimals,
                    )
                if meshes_np.joints2d is not None:
                    joints2d = meshes_np.joints2d[i]
                    if normalize:
                        h, w = self.orig_shape
                        joints2d = joints2d / np.array([w, h], dtype=float)
                    mesh_row["joints2d"] = {
                        "x": [round(float(x), decimals) for x in joints2d[:, 0]],
                        "y": [round(float(y), decimals) for y in joints2d[:, 1]],
                    }
                row["mesh"] = mesh_row
            if track_ids is not None:
                row["track_id"] = int(track_ids[i])
            rows.append(row)
        return rows

    def to_json(self, **kwargs) -> str:
        return json.dumps(self.summary(**kwargs))

    def __len__(self) -> int:
        if self.boxes is not None:
            return len(self.boxes)
        if self.points is not None:
            return len(self.points)
        if self.embeddings is not None:
            return len(self.embeddings)
        if self.probs is not None:
            return 1
        if self.semantic_mask is not None:
            return 1
        if self.panoptic is not None:
            return 1
        if self.depth_map is not None:
            return 1
        if self.normal_map is not None:
            return 1
        if self.albedo is not None:
            return 1
        if self.edges is not None:
            return 1
        if self.restored is not None:
            return 1
        if self.matte is not None:
            return 1
        if self.ocr is not None:
            return len(self.ocr)
        if self.meshes is not None:
            return len(self.meshes)
        return 0

    def __repr__(self) -> str:
        parts = [
            f"path='{self.path}'",
            f"orig_shape={self.orig_shape}",
            f"boxes={self.boxes}",
        ]
        if self.points is not None:
            parts.append(f"points={self.points}")
        if self.masks is not None:
            parts.append(f"masks={self.masks}")
        if self.semantic_mask is not None:
            parts.append(f"semantic_mask={self.semantic_mask}")
        if self.panoptic is not None:
            parts.append(f"panoptic={self.panoptic}")
        if self.depth_map is not None:
            parts.append(f"depth_map={self.depth_map}")
        if self.normal_map is not None:
            parts.append(f"normal_map={self.normal_map}")
        if self.albedo is not None:
            parts.append(f"albedo_shape={tuple(self.albedo.data.shape)}")
        if self.edges is not None:
            parts.append(f"edges={self.edges}")
        if self.restored is not None:
            parts.append(f"restored={self.restored}")
            if self.restore_scale != 1:
                parts.append(f"restore_scale={self.restore_scale}")
        if self.matte is not None:
            parts.append(f"matte={self.matte}")
        if self.ocr is not None:
            parts.append(f"ocr={self.ocr}")
        if self.meshes is not None:
            parts.append(f"meshes={self.meshes}")
        if self.boxes3d is not None:
            parts.append(f"boxes3d={self.boxes3d}")
        if self.track_id is not None:
            parts.append(f"track_ids={len(self.track_id)}")
        if self.frame_idx is not None:
            parts.append(f"frame_idx={self.frame_idx}")
        return f"Results({', '.join(parts)})"


def stack_result_embeddings(prediction: Any) -> torch.Tensor:
    """Stack every ``Results.embeddings`` row into one CPU float32 tensor."""
    payloads: List[torch.Tensor] = []

    def collect(value: Any) -> None:
        if isinstance(value, Results):
            if value.embeddings is None:
                raise RuntimeError(
                    "Prediction did not produce embeddings. Load the model with "
                    "task='embed' before calling embed()."
                )
            data = value.embeddings.data
            tensor = (
                data.detach().to(device="cpu", dtype=torch.float32)
                if isinstance(data, torch.Tensor)
                else torch.as_tensor(data, dtype=torch.float32)
            )
            if tensor.ndim == 1:
                tensor = tensor.unsqueeze(0)
            if tensor.ndim != 2:
                raise RuntimeError(
                    f"Expected (N, D) embeddings, got shape {tuple(tensor.shape)}."
                )
            payloads.append(tensor)
            return
        if isinstance(value, (str, bytes, np.ndarray)):
            raise RuntimeError(
                "Prediction did not return Results objects with embeddings."
            )
        try:
            iterator = iter(value)
        except TypeError as exc:
            raise RuntimeError(
                "Prediction did not return Results objects with embeddings."
            ) from exc
        for item in iterator:
            collect(item)

    collect(prediction)
    if not payloads:
        return torch.empty((0, 0), dtype=torch.float32)
    # Zero-row payloads (e.g. a face-less image whose embedder had not yet
    # resolved its dimension) contribute no rows and must not fail or skew the
    # dimension-consistency check.
    non_empty = [tensor for tensor in payloads if tensor.shape[0] > 0]
    if not non_empty:
        width = max(int(tensor.shape[1]) for tensor in payloads)
        return torch.empty((0, width), dtype=torch.float32)
    dimensions = {int(tensor.shape[1]) for tensor in non_empty}
    if len(dimensions) != 1:
        raise RuntimeError(
            f"Cannot stack embeddings with different dimensions: {sorted(dimensions)}."
        )
    return torch.cat(non_empty, dim=0)
