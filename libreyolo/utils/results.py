"""Ultralytics-style flat result containers for LibreYOLO."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch


TensorLike = Union[torch.Tensor, np.ndarray]


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
                contour = max(contours, key=cv2.contourArea).squeeze(1).astype(np.float64)
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
                return torch.ones(self.data.shape[:-1], dtype=torch.bool, device=self.data.device)
            return np.ones(self.data.shape[:-1], dtype=bool)
        return conf > 0


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
                [x.min(dim=1).values, y.min(dim=1).values, x.max(dim=1).values, y.max(dim=1).values],
                dim=1,
            )
        return np.stack([x.min(axis=1), y.min(axis=1), x.max(axis=1), y.max(axis=1)], axis=1)


class Results:
    """Single-image result with flat Ultralytics-compatible slots."""

    _keys = ("boxes", "masks", "probs", "keypoints", "obb")

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
        speed: Optional[Dict[str, float]] = None,
        track_id: Optional[TensorLike] = None,
        frame_idx: Optional[int] = None,
    ):
        if boxes is not None and boxes.orig_shape is None:
            boxes = boxes.with_orig_shape(orig_shape)
        if boxes is not None and track_id is not None:
            boxes = boxes.with_id(track_id)

        self.boxes = boxes
        self.masks = masks
        self.keypoints = keypoints
        self.probs = probs
        self.obb = obb
        self.orig_shape = orig_shape
        self.path = path
        self.names = names or {}
        self.speed = speed or {}
        self.track_id = track_id if track_id is not None else (boxes.id if boxes else None)
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
            overrides[key] = getattr(value, method)(*args, **kwargs) if value is not None else None

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
        track_id: Optional[TensorLike] = None,
    ) -> "Results":
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
        if track_id is not None:
            self.track_id = track_id
            if self.boxes is not None:
                self.boxes = self.boxes.with_id(track_id)
        return self

    def summary(self, normalize: bool = False, decimals: int = 5) -> List[Dict[str, Any]]:
        if self.boxes is None:
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
            if self.masks is not None:
                segment = self.masks.xyn[i] if normalize else self.masks.xy[i]
                row["segments"] = {
                    "x": [round(float(x), decimals) for x in segment[:, 0]],
                    "y": [round(float(y), decimals) for y in segment[:, 1]],
                }
            if track_ids is not None:
                row["track_id"] = int(track_ids[i])
            rows.append(row)
        return rows

    def to_json(self, **kwargs) -> str:
        return json.dumps(self.summary(**kwargs))

    def __len__(self) -> int:
        if self.boxes is not None:
            return len(self.boxes)
        if self.probs is not None:
            return 1
        return 0

    def __repr__(self) -> str:
        parts = [
            f"path='{self.path}'",
            f"orig_shape={self.orig_shape}",
            f"boxes={self.boxes}",
        ]
        if self.masks is not None:
            parts.append(f"masks={self.masks}")
        if self.track_id is not None:
            parts.append(f"track_ids={len(self.track_id)}")
        if self.frame_idx is not None:
            parts.append(f"frame_idx={self.frame_idx}")
        return f"Results({', '.join(parts)})"
