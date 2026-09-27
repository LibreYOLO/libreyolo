"""User-supplied video classification dataset for the V-JEPA 2 attentive probe.

The dataset is entirely user-supplied. Nothing here downloads, mirrors or
bundles Something-Something V2, Diving48, Kinetics or any other corpus.

Data YAML keeps the usual LibreYOLO shape::

    path: /path/to/dataset
    train: train.txt
    val: val.txt
    names:
      0: class_a
      1: class_b

Each manifest line is ``<relative-video-path> <integer-class-id>``. Paths may
contain spaces: the class id is parsed from the **last** whitespace-separated
field and everything before it is the path. A tab, if present, is treated as
the delimiter instead, which makes paths with trailing spaces representable.
Quoting the path also works.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .preprocess import clip_frame_indices, preprocess_frames


class ManifestError(ValueError):
    """Raised for malformed dataset YAML or manifest rows."""


def parse_manifest_line(line: str, lineno: int) -> Tuple[str, int]:
    """Parse one ``<path> <class-id>`` row.

    The class id is taken from the last field so that unquoted paths with
    spaces still parse. A malformed row is an error, never a silent skip.
    """
    raw = line.rstrip("\n").rstrip("\r")
    if not raw.strip():
        raise ManifestError(f"line {lineno}: blank line")

    if "\t" in raw:
        path_part, _, label_part = raw.rpartition("\t")
    else:
        path_part, _, label_part = raw.rpartition(" ")

    if not path_part.strip():
        raise ManifestError(
            f"line {lineno}: expected '<video-path> <class-id>', got {raw!r}"
        )

    path_part = path_part.strip()
    if len(path_part) >= 2 and path_part[0] == path_part[-1] and path_part[0] in "\"'":
        path_part = path_part[1:-1]

    try:
        label = int(label_part.strip())
    except ValueError:
        raise ManifestError(
            f"line {lineno}: class id {label_part.strip()!r} is not an integer"
        ) from None

    return path_part, label


def load_video_dataset(data_yaml: str | Path) -> Dict:
    """Validate the YAML and both manifests before any training starts.

    Every failure mode is reported up front rather than at the first bad batch:
    missing keys, missing files, duplicate class ids, out-of-range labels and a
    names/label mismatch.
    """
    import yaml

    yaml_path = Path(data_yaml)
    if not yaml_path.exists():
        raise ManifestError(f"dataset YAML not found: {yaml_path}")

    with yaml_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ManifestError(f"{yaml_path}: expected a YAML mapping")

    names = config.get("names")
    if not isinstance(names, dict) or not names:
        raise ManifestError(f"{yaml_path}: 'names' must be a non-empty mapping")

    try:
        names = {int(k): str(v) for k, v in names.items()}
    except (TypeError, ValueError):
        raise ManifestError(f"{yaml_path}: 'names' keys must be integers") from None

    expected = set(range(len(names)))
    if set(names) != expected:
        raise ManifestError(
            f"{yaml_path}: 'names' keys must be contiguous 0..{len(names) - 1}, "
            f"got {sorted(names)}"
        )

    root = Path(config.get("path", yaml_path.parent))
    if not root.is_absolute():
        root = (yaml_path.parent / root).resolve()

    splits: Dict[str, List[Tuple[Path, int]]] = {}
    for split in ("train", "val"):
        rel = config.get(split)
        if rel is None:
            if split == "train":
                raise ManifestError(f"{yaml_path}: missing required '{split}' manifest")
            continue
        manifest = Path(rel)
        if not manifest.is_absolute():
            manifest = root / manifest
        if not manifest.exists():
            raise ManifestError(f"{yaml_path}: {split} manifest not found: {manifest}")

        entries: List[Tuple[Path, int]] = []
        with manifest.open(encoding="utf-8") as handle:
            for lineno, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                rel_path, label = parse_manifest_line(line, lineno)
                if not 0 <= label < len(names):
                    raise ManifestError(
                        f"{manifest}:{lineno}: class id {label} out of range for "
                        f"{len(names)} classes"
                    )
                video = Path(rel_path)
                if not video.is_absolute():
                    video = root / video
                if not video.exists():
                    raise ManifestError(f"{manifest}:{lineno}: video not found: {video}")
                entries.append((video, label))

        if not entries:
            raise ManifestError(f"{manifest}: no usable rows")
        splits[split] = entries

    labels_seen = {label for entries in splits.values() for _, label in entries}
    unused = set(names) - labels_seen
    if unused:
        raise ManifestError(
            f"{yaml_path}: classes {sorted(unused)} appear in 'names' but in no "
            "manifest row; fix the names mapping or the manifests"
        )

    return {"root": root, "names": names, "nc": len(names), **splits}


def decode_clip(
    path: Path,
    clip_frames: int,
    frame_stride: int,
    crop_size: int,
    *,
    train: bool,
    rng: random.Random | None = None,
) -> torch.Tensor:
    """Decode one clip as ``(F, C, H, W)``.

    Validation is deterministic (centered). Training takes a random temporal
    crop, which is the only augmentation the pinned probe recipe requires.
    """
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ManifestError(f"could not open video: {path}")
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            total = 0
            while capture.grab():
                total += 1
            capture.release()
            capture = cv2.VideoCapture(str(path))
        if total <= 0:
            raise ManifestError(f"decoded no frames from {path}")

        indices = clip_frame_indices(total, clip_frames, frame_stride)
        if train and rng is not None:
            span = (clip_frames - 1) * frame_stride + 1
            if total > span:
                start = rng.randint(0, total - span)
                indices = [start + i * frame_stride for i in range(clip_frames)]

        wanted = set(indices)
        decoded: Dict[int, np.ndarray] = {}
        position = 0
        highest = max(wanted)
        while position <= highest:
            ok, frame = capture.read()
            if not ok:
                break
            if position in wanted:
                decoded[position] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            position += 1
        if not decoded:
            raise ManifestError(f"could not decode any frame from {path}")

        held = decoded[min(decoded)]
        frames = []
        for index in indices:
            held = decoded.get(index, held)
            frames.append(held)
    finally:
        capture.release()

    # preprocess_frames returns (1, F, C, H, W); the dataset item drops batch.
    return preprocess_frames(frames, crop_size)[0]


class VideoClipDataset(Dataset):
    """Returns ``((F, C, H, W), int)``; the collate keeps time out of batch."""

    def __init__(
        self,
        entries: Sequence[Tuple[Path, int]],
        clip_frames: int,
        frame_stride: int,
        crop_size: int,
        *,
        train: bool,
        seed: int = 0,
    ):
        self.entries = list(entries)
        self.clip_frames = clip_frames
        self.frame_stride = frame_stride
        self.crop_size = crop_size
        self.train = train
        self._seed = seed

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int):
        path, label = self.entries[index]
        rng = random.Random(self._seed + index) if self.train else None
        clip = decode_clip(
            path,
            self.clip_frames,
            self.frame_stride,
            self.crop_size,
            train=self.train,
            rng=rng,
        )
        return clip, label


def collate_clips(batch):
    """``(B, F, C, H, W)`` plus labels. Time must never fold into batch."""
    clips = torch.stack([item[0] for item in batch], dim=0)
    labels = torch.as_tensor([item[1] for item in batch], dtype=torch.long)
    return clips, labels
