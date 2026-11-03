"""Optical flow dataset for LibreYOLO.

A flow dataset is a set of frame *sequences*. Frames live under
``images/<split>/<sequence>/`` and are ordered by file name; every
consecutive pair ``(frame_i, frame_i+1)`` inside one sequence directory is
one sample, whose ground truth is the flow from ``frame_i`` to ``frame_i+1``
stored under ``flows_dir`` with the same relative path and stem::

    dataset/
        images/val/alley_1/frame_0001.png
        images/val/alley_1/frame_0002.png
        images/val/alley_1/frame_0003.png
        flows/val/alley_1/frame_0001.flo     # frame_0001 -> frame_0002
        flows/val/alley_1/frame_0002.flo     # frame_0002 -> frame_0003

The last frame of a sequence has no flow file: there is no next frame. Frames
in different directories never pair with each other.

The dataset YAML follows the common contract (``path``/``train``/``val``/
``test``) plus three flow keys:

- ``flows_dir``: flow directory name substituted for ``images`` in each
  frame path (default ``flows``).
- ``flow_format``: ``flo`` (Middlebury ``.flo``, the default), ``kitti``
  (16-bit RGB PNG, ``(value - 2**15) / 64`` with the blue channel as the
  validity mask) or ``npy`` (a ``(H, W, 2)`` float array; non-finite values
  are invalid).
- ``flow_valid_suffix``: suffix of an optional per-pair validity mask stored
  next to the flow file as ``<stem><suffix>.png`` (default ``_valid``);
  non-zero pixels are valid. It is combined with the format's own mask.

Ground-truth vectors are in pixels of the frame canvas, ``(dx, dy)`` from
frame ``i`` to frame ``i + 1``. Samples are returned at native resolution;
flow training and augmentation are not part of this loader.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .utils import get_img_files, load_data_config

logger = logging.getLogger(__name__)

FLO_TAG = 202021.25
FLOW_FORMATS = ("flo", "kitti", "npy")
FLOW_FORMAT_EXTENSIONS = {"flo": ".flo", "kitti": ".png", "npy": ".npy"}
DEFAULT_FLOWS_DIR = "flows"
DEFAULT_FLOW_VALID_SUFFIX = "_valid"
_KITTI_OFFSET = 2.0**15
_KITTI_SCALE = 64.0


# ---------------------------------------------------------------------------
# Flow file readers and writers
# ---------------------------------------------------------------------------


def read_flo(path: str | Path) -> np.ndarray:
    """Read a Middlebury ``.flo`` file into a float32 ``(H, W, 2)`` array."""
    with open(path, "rb") as handle:
        tag = np.frombuffer(handle.read(4), dtype=np.float32)
        if tag.size != 1 or float(tag[0]) != FLO_TAG:
            raise ValueError(
                f"{path} is not a Middlebury .flo file (bad magic number)."
            )
        width = int(np.frombuffer(handle.read(4), dtype=np.int32)[0])
        height = int(np.frombuffer(handle.read(4), dtype=np.int32)[0])
        if width <= 0 or height <= 0:
            raise ValueError(f"{path} declares an invalid size {width}x{height}.")
        count = width * height * 2
        data = np.frombuffer(handle.read(count * 4), dtype=np.float32)
        if data.size != count:
            raise ValueError(f"{path} is truncated: expected {count} floats.")
    return data.reshape(height, width, 2).astype(np.float32, copy=True)


def write_flo(path: str | Path, flow: np.ndarray) -> None:
    """Write a float32 ``(H, W, 2)`` array as a Middlebury ``.flo`` file."""
    flow = np.asarray(flow, dtype=np.float32)
    if flow.ndim != 3 or flow.shape[-1] != 2:
        raise ValueError(f"flow must have shape (H, W, 2), got {flow.shape}.")
    height, width = flow.shape[:2]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(np.array([FLO_TAG], dtype=np.float32).tobytes())
        handle.write(np.array([width, height], dtype=np.int32).tobytes())
        handle.write(np.ascontiguousarray(flow).tobytes())


def read_kitti_flow(path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """Read a KITTI 16-bit PNG flow file.

    Returns ``(flow, valid)``: ``flow`` is float32 ``(H, W, 2)`` decoded as
    ``(value - 2**15) / 64`` from the red (u) and green (v) channels, and
    ``valid`` is the boolean ``(H, W)`` mask stored in the blue channel.
    """
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read KITTI flow file {path}.")
    if image.ndim != 3 or image.shape[-1] < 3 or image.dtype != np.uint16:
        raise ValueError(
            f"{path} is not a KITTI flow file: expected a 16-bit RGB PNG, "
            f"got dtype {image.dtype} with shape {image.shape}."
        )
    # OpenCV returns BGR: blue is the validity mask, green is v, red is u.
    u = (image[..., 2].astype(np.float32) - _KITTI_OFFSET) / _KITTI_SCALE
    v = (image[..., 1].astype(np.float32) - _KITTI_OFFSET) / _KITTI_SCALE
    valid = image[..., 0] > 0
    return np.stack([u, v], axis=-1).astype(np.float32), valid


def write_kitti_flow(
    path: str | Path, flow: np.ndarray, valid: Optional[np.ndarray] = None
) -> None:
    """Write a ``(H, W, 2)`` flow (and optional mask) as a KITTI 16-bit PNG."""
    import cv2

    flow = np.asarray(flow, dtype=np.float32)
    if flow.ndim != 3 or flow.shape[-1] != 2:
        raise ValueError(f"flow must have shape (H, W, 2), got {flow.shape}.")
    if valid is None:
        valid = np.isfinite(flow).all(axis=-1)
    encoded = np.clip(
        np.nan_to_num(flow) * _KITTI_SCALE + _KITTI_OFFSET, 0, 65535
    ).astype(np.uint16)
    image = np.zeros((*flow.shape[:2], 3), dtype=np.uint16)
    image[..., 2] = encoded[..., 0]
    image[..., 1] = encoded[..., 1]
    image[..., 0] = np.asarray(valid).astype(np.uint16)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not write KITTI flow file {path}.")


def read_npy_flow(path: str | Path) -> np.ndarray:
    """Read a ``(H, W, 2)`` float array saved with ``numpy.save``."""
    flow = np.load(path)
    if flow.ndim != 3 or flow.shape[-1] != 2:
        raise ValueError(
            f"Flow file {path} has shape {flow.shape}; expected (H, W, 2)."
        )
    return flow.astype(np.float32)


def read_flow_file(path: str | Path, flow_format: str) -> Tuple[np.ndarray, np.ndarray]:
    """Read a flow file in *flow_format*; return ``(flow, valid)``.

    ``flow`` is float32 ``(H, W, 2)``; ``valid`` is boolean ``(H, W)``.
    Non-finite vectors are always invalid, whatever the format says.
    """
    flow_format = normalize_flow_format(flow_format)
    if flow_format == "flo":
        flow = read_flo(path)
        valid = np.ones(flow.shape[:2], dtype=bool)
    elif flow_format == "kitti":
        flow, valid = read_kitti_flow(path)
    else:
        flow = read_npy_flow(path)
        valid = np.ones(flow.shape[:2], dtype=bool)
    finite = np.isfinite(flow).all(axis=-1)
    valid = valid & finite
    flow = np.where(finite[..., None], flow, 0.0).astype(np.float32)
    return flow, valid


def read_valid_mask(path: str | Path) -> np.ndarray:
    """Read an optional validity mask image; non-zero pixels are valid."""
    with Image.open(path) as mask_img:
        mask = np.asarray(mask_img)
    if mask.ndim == 3:
        mask = mask[..., 0]
    if mask.ndim != 2:
        raise ValueError(f"Flow mask {path} has shape {mask.shape}; expected (H, W).")
    return mask > 0


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def normalize_flow_format(flow_format: object) -> str:
    value = str(flow_format or "flo").strip().lower().lstrip(".")
    if value not in FLOW_FORMATS:
        raise ValueError(
            f"flow_format must be one of {', '.join(FLOW_FORMATS)}, got {flow_format!r}."
        )
    return value


def build_flow_pairs(img_files: Sequence[Path]) -> List[Tuple[Path, Path]]:
    """Pair consecutive frames that share a parent directory.

    Frames are sorted by path, so files are ordered by name inside each
    sequence directory. A directory with a single frame contributes no pair.
    """
    ordered = sorted(Path(p) for p in img_files)
    pairs: List[Tuple[Path, Path]] = []
    for current, following in zip(ordered, ordered[1:]):
        if current.parent == following.parent:
            pairs.append((current, following))
    return pairs


def img2flow_path(
    img_path: Path, flows_dir: str = DEFAULT_FLOWS_DIR, flow_format: str = "flo"
) -> Path:
    """Map a frame path to its flow file (flow from this frame to the next).

    Mirrors ``img2label_paths``: the ``images`` directory component becomes
    ``flows_dir`` and the extension becomes the format's extension.
    """
    extension = FLOW_FORMAT_EXTENSIONS[normalize_flow_format(flow_format)]
    path_str = str(img_path)
    for sep in (os.sep, "/", "\\"):
        path_str = path_str.replace(f"{sep}images{sep}", f"{sep}{flows_dir}{sep}")
        path_str = path_str.replace(f"{sep}images", f"{sep}{flows_dir}")
    return Path(path_str).with_suffix(extension)


def flow2valid_path(flow_path: Path, valid_suffix: str = DEFAULT_FLOW_VALID_SUFFIX) -> Optional[Path]:
    """Return the optional ``<stem><suffix>.png`` mask next to a flow file."""
    candidate = flow_path.with_name(f"{flow_path.stem}{valid_suffix}.png")
    return candidate if candidate.exists() else None


def resolve_flow_data(data: str | Path, allow_scripts: bool = False) -> Dict:
    """Load a flow dataset YAML config through ``load_data_config``."""
    return load_data_config(str(data), allow_scripts=allow_scripts)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class FlowDataset(Dataset):
    """Consecutive-frame pairs with dense flow ground truth.

    Each item is ``(img_a, img_b, flow, valid, info, index)``: two ``(3, H, W)``
    float images in ``[0, 1]``, the ``(2, H, W)`` float32 flow from ``img_a``
    to ``img_b`` in pixels, a ``(H, W)`` boolean validity mask, and an info
    dict with ``img_path``, ``img_path_next``, ``flow_path``, ``sequence``
    and ``orig_shape``.
    """

    def __init__(self, data_config: Dict, split: str):
        self.split = split
        split_value = data_config.get(split)
        if not split_value:
            raise ValueError(f"Flow dataset config has no '{split}' split.")
        img_files = data_config.get(f"{split}_img_files") or get_img_files(split_value)
        if not img_files:
            raise FileNotFoundError(
                f"No frames found for flow split '{split}' at {split_value}."
            )
        self.flows_dir = str(data_config.get("flows_dir") or DEFAULT_FLOWS_DIR)
        self.flow_format = normalize_flow_format(data_config.get("flow_format"))
        self.valid_suffix = str(
            data_config.get("flow_valid_suffix") or DEFAULT_FLOW_VALID_SUFFIX
        )

        self.pairs = build_flow_pairs(img_files)
        if not self.pairs:
            raise ValueError(
                f"Flow split '{split}' holds no consecutive frame pair: every "
                "sequence directory needs at least two frames."
            )
        self.flow_files = [
            img2flow_path(first, self.flows_dir, self.flow_format)
            for first, _ in self.pairs
        ]
        missing = [str(p) for p in self.flow_files if not p.exists()]
        if missing:
            preview = ", ".join(missing[:3])
            raise FileNotFoundError(
                f"{len(missing)} flow file(s) missing for split '{split}' "
                f"(e.g. {preview}). Expected '{self.flow_format}' flow files "
                f"under '{self.flows_dir}' mirroring the frames tree, one per "
                "consecutive pair, named after the first frame of the pair."
            )
        self.valid_files = [
            flow2valid_path(path, self.valid_suffix) for path in self.flow_files
        ]

    def __len__(self) -> int:
        return len(self.pairs)

    def _load_image(self, path: Path) -> np.ndarray:
        with Image.open(path) as img_pil:
            return np.array(img_pil.convert("RGB"))

    def load_target(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(flow (H, W, 2), valid (H, W))`` for pair *index*."""
        flow, valid = read_flow_file(self.flow_files[index], self.flow_format)
        mask_path = self.valid_files[index]
        if mask_path is not None:
            mask = read_valid_mask(mask_path)
            if mask.shape != valid.shape:
                raise ValueError(
                    f"Flow mask {mask_path} shape {mask.shape} does not match "
                    f"flow shape {valid.shape}."
                )
            valid = valid & mask
        return flow, valid

    def __getitem__(self, index: int):
        first, second = self.pairs[index]
        img_a = self._load_image(first)
        img_b = self._load_image(second)
        if img_a.shape != img_b.shape:
            raise ValueError(
                f"Frames {first} and {second} differ in size "
                f"({img_a.shape[:2]} vs {img_b.shape[:2]}); a flow pair must "
                "share one canvas."
            )
        flow, valid = self.load_target(index)
        orig_shape = img_a.shape[:2]
        if flow.shape[:2] != orig_shape:
            raise ValueError(
                f"Flow file {self.flow_files[index]} shape {flow.shape[:2]} does "
                f"not match frame shape {orig_shape}."
            )

        def to_tensor(img: np.ndarray) -> torch.Tensor:
            return (
                torch.from_numpy(np.ascontiguousarray(img))
                .permute(2, 0, 1)
                .float()
                .div_(255.0)
            )

        flow_tensor = torch.from_numpy(np.ascontiguousarray(flow)).permute(2, 0, 1).float()
        valid_tensor = torch.from_numpy(np.ascontiguousarray(valid)).bool()
        info = {
            "img_path": str(first),
            "img_path_next": str(second),
            "flow_path": str(self.flow_files[index]),
            "sequence": first.parent.name,
            "orig_shape": (int(orig_shape[0]), int(orig_shape[1])),
        }
        return to_tensor(img_a), to_tensor(img_b), flow_tensor, valid_tensor, info, index


def _stack_or_list(items: List[torch.Tensor]):
    if items and all(t.shape == items[0].shape for t in items):
        return torch.stack(items, dim=0)
    return items


def flow_collate_fn(batch):
    """Collate flow samples into ``(imgs_a, imgs_b, flows, valids, infos, ids)``.

    Tensors are stacked when every sample in the batch shares one canvas and
    kept as lists otherwise, so sequences of different resolutions can share
    a validation loader.
    """
    imgs_a = _stack_or_list([item[0] for item in batch])
    imgs_b = _stack_or_list([item[1] for item in batch])
    flows = _stack_or_list([item[2] for item in batch])
    valids = _stack_or_list([item[3] for item in batch])
    infos = [item[4] for item in batch]
    ids = [item[5] for item in batch]
    return imgs_a, imgs_b, flows, valids, infos, ids


__all__ = [
    "FLOW_FORMATS",
    "FlowDataset",
    "build_flow_pairs",
    "flow_collate_fn",
    "img2flow_path",
    "normalize_flow_format",
    "read_flo",
    "read_flow_file",
    "read_kitti_flow",
    "read_npy_flow",
    "resolve_flow_data",
    "write_flo",
    "write_kitti_flow",
]
