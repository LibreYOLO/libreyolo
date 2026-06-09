"""One-pass dataset scan for LibreDoctor.

The dataset is read once into a :class:`DatasetSnapshot`; every check is a
pure function over the snapshot. Label files are always parsed; image
contents are decoded in a separate, threaded pass (``scan_images``) that the
``--fast`` mode skips entirely.
"""

import hashlib
import io
import logging
import math
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml
from PIL import Image
from tqdm import tqdm

from ..data.utils import img2label_paths, load_data_config, resolve_dataset_yaml
from .config import DoctorError

logger = logging.getLogger(__name__)

SPLITS = ("train", "val", "test")

_EXIF_ORIENTATION_TAG = 0x0112

# Channel count per PIL mode, to spot mixed-channel datasets.
_MODE_CHANNELS = {
    "1": 1,
    "L": 1,
    "LA": 2,
    "I": 1,
    "I;16": 1,
    "F": 1,
    "P": 3,  # palette images decode to RGB in practice
    "RGB": 3,
    "YCbCr": 3,
    "RGBA": 4,
    "CMYK": 4,
}


@dataclass
class LabelIssue:
    """A label line the detection parser rejects."""

    line_no: int
    reason: str


@dataclass
class ImageScan:
    """Results of decoding one image (None fields when decoding failed)."""

    ok: bool
    error: Optional[str] = None
    width: int = 0
    height: int = 0
    mode: str = ""
    exif_orientation: int = 0
    sha1: Optional[str] = None
    dhash: Optional[int] = None
    uniform: bool = False

    @property
    def channels(self) -> Optional[int]:
        return _MODE_CHANNELS.get(self.mode)


@dataclass
class ImageRecord:
    """One image with its parsed label file."""

    path: Path
    label_path: Path
    label_exists: bool
    boxes: np.ndarray  # (n, 5) float32: cls, cx, cy, w, h (normalized)
    label_issues: list[LabelIssue] = field(default_factory=list)
    label_digest: Optional[str] = None  # sha1 of label bytes, None when empty
    scan: Optional[ImageScan] = None  # filled by scan_images()

    @property
    def is_background(self) -> bool:
        return self.boxes.shape[0] == 0


@dataclass
class SplitSnapshot:
    name: str
    records: list[ImageRecord]

    @property
    def instances(self) -> int:
        return sum(r.boxes.shape[0] for r in self.records)

    def class_counts(self) -> Counter:
        """Instance count per class id (only finite, integral ids)."""
        counts: Counter = Counter()
        for r in self.records:
            if r.boxes.shape[0]:
                counts.update(int(c) for c in r.boxes[:, 0])
        return counts


@dataclass
class DatasetSnapshot:
    yaml_path: Path
    root: Path
    raw_config: dict[str, Any]  # the YAML exactly as written
    config: dict[str, Any]  # resolved by load_data_config
    nc: Optional[int]
    names: dict[int, str]
    splits: list[SplitSnapshot]
    line_field_counts: Counter  # field-count histogram across all label lines
    images_scanned: bool = False

    def split(self, name: str) -> Optional[SplitSnapshot]:
        for s in self.splits:
            if s.name == name:
                return s
        return None

    def stats(self) -> dict[str, Any]:
        per_split: dict[str, Any] = {}
        for s in self.splits:
            per_split[s.name] = {
                "images": len(s.records),
                "instances": s.instances,
                "background": sum(1 for r in s.records if r.is_background),
            }
        instance_counts = {}
        train = self.split("train")
        if train is not None:
            counts = train.class_counts()
            instance_counts = {
                self.names.get(cid, str(cid)): n for cid, n in sorted(counts.items())
            }
        return {
            "yaml": str(self.yaml_path),
            "root": str(self.root),
            "nc": self.nc,
            "splits": per_split,
            "train_instance_counts": instance_counts,
            "images_scanned": self.images_scanned,
        }


def normalize_names(raw: Any) -> dict[int, str]:
    """Normalize the YAML ``names`` field (list or dict) to dict[int, str]."""
    if isinstance(raw, dict):
        out = {}
        for k, v in raw.items():
            try:
                out[int(k)] = str(v)
            except (TypeError, ValueError):
                continue
        return out
    if isinstance(raw, (list, tuple)):
        return {i: str(v) for i, v in enumerate(raw)}
    return {}


def parse_label_file(
    path: Path, field_counts: Counter
) -> tuple[np.ndarray, list[LabelIssue], Optional[str]]:
    """Parse one detection label file.

    Mirrors what training accepts: 5 whitespace-separated fields per line,
    integer class then four finite floats. Everything else is an issue.
    """
    boxes: list[list[float]] = []
    issues: list[LabelIssue] = []
    try:
        data = path.read_bytes()
    except OSError as exc:
        return _empty_boxes(), [LabelIssue(0, f"unreadable: {exc}")], None

    digest = hashlib.sha1(data).hexdigest() if data.strip() else None
    text = data.decode("utf-8", errors="replace")
    for line_no, line in enumerate(text.splitlines(), 1):
        parts = line.split()
        if not parts:
            continue
        field_counts[len(parts)] += 1
        if len(parts) != 5:
            issues.append(LabelIssue(line_no, f"expected 5 fields, got {len(parts)}"))
            continue
        try:
            cls_id = int(parts[0])
            vals = [float(p) for p in parts[1:]]
        except ValueError:
            issues.append(LabelIssue(line_no, "non-numeric value"))
            continue
        if not all(math.isfinite(v) for v in vals):
            issues.append(LabelIssue(line_no, "non-finite value (nan/inf)"))
            continue
        boxes.append([float(cls_id), *vals])

    if boxes:
        return np.asarray(boxes, dtype=np.float32), issues, digest
    return _empty_boxes(), issues, digest


def _empty_boxes() -> np.ndarray:
    return np.zeros((0, 5), dtype=np.float32)


def build_snapshot(data: str, autodownload: bool = False) -> DatasetSnapshot:
    """Load the data YAML and parse every label file (no image decoding)."""
    try:
        yaml_path = resolve_dataset_yaml(data)
    except FileNotFoundError as exc:
        raise DoctorError(str(exc)) from exc

    try:
        with open(yaml_path, "r") as f:
            raw_config = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise DoctorError(f"Cannot parse dataset YAML {yaml_path}: {exc}") from exc
    if not isinstance(raw_config, dict):
        raise DoctorError(f"Dataset YAML {yaml_path} is not a mapping.")

    try:
        config = load_data_config(
            str(yaml_path), autodownload=autodownload, allow_scripts=False
        )
    except Exception as exc:  # load_data_config raises a mixed bag
        raise DoctorError(f"Cannot load dataset config {yaml_path}: {exc}") from exc

    field_counts: Counter = Counter()
    splits: list[SplitSnapshot] = []
    for split_name in SPLITS:
        img_files = config.get(f"{split_name}_img_files")
        if not img_files:
            continue
        label_files = config.get(f"{split_name}_label_files") or img2label_paths(
            img_files
        )
        records = []
        for img_path, label_path in zip(img_files, label_files):
            label_exists = label_path.exists()
            if label_exists:
                boxes, issues, digest = parse_label_file(label_path, field_counts)
            else:
                boxes, issues, digest = _empty_boxes(), [], None
            records.append(
                ImageRecord(
                    path=Path(img_path),
                    label_path=Path(label_path),
                    label_exists=label_exists,
                    boxes=boxes,
                    label_issues=issues,
                    label_digest=digest,
                )
            )
        splits.append(SplitSnapshot(name=split_name, records=records))

    nc = raw_config.get("nc")
    names = normalize_names(raw_config.get("names"))
    if nc is None and names:
        nc = len(names)

    return DatasetSnapshot(
        yaml_path=yaml_path,
        root=Path(config.get("path", yaml_path.parent)),
        raw_config=raw_config,
        config=config,
        nc=int(nc)
        if isinstance(nc, (int, float)) and not isinstance(nc, bool)
        else None,
        names=names,
        splits=splits,
        line_field_counts=field_counts,
    )


def detect_non_detection(snapshot: DatasetSnapshot) -> Optional[str]:
    """Format guard: return the suspected task when this is clearly not a
    detection dataset, else None.

    Fires only on *consistent* non-detection shapes so a genuinely broken
    detection dataset still gets per-line syntax errors instead.
    """
    if "kpt_shape" in snapshot.raw_config:
        return "pose (kpt_shape present in the YAML)"

    counts = snapshot.line_field_counts
    total = sum(counts.values())
    if total == 0:
        return None
    if counts.get(5, 0) / total >= 0.7:
        return None

    non5 = {n: c for n, c in counts.items() if n != 5}
    # Consistent pose shape: 5 + K*2 or 5 + K*3 extra fields on most lines.
    dominant, dominant_count = max(non5.items(), key=lambda kv: kv[1])
    if dominant_count / total > 0.5:
        extra = dominant - 5
        if extra > 0 and (extra % 3 == 0 or extra % 2 == 0):
            if dominant == 9:
                return "obb or segment (9-field label lines)"
            if dominant % 2 == 1 and dominant >= 7:
                return "segment (polygon label lines)"
            return "pose (keypoint label lines)"
    # Polygon files have varying odd field counts >= 7.
    odd_poly = sum(c for n, c in non5.items() if n >= 7 and n % 2 == 1)
    if odd_poly / total > 0.5:
        return "segment (polygon label lines)"
    return None


def scan_images(
    snapshot: DatasetSnapshot,
    workers: Optional[int] = None,
    progress: bool = True,
) -> None:
    """Decode every image once, filling ``record.scan`` in place.

    One read per file yields: corruption status, dimensions, mode, EXIF
    orientation, a content hash (exact duplicates / leakage), a dHash
    (near duplicates), and a uniformity flag.
    """
    records = [r for s in snapshot.splits for r in s.records]
    if not records:
        snapshot.images_scanned = True
        return
    if workers is None:
        workers = min(32, (os.cpu_count() or 4) + 4)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = ex.map(_scan_one, (r.path for r in records))
        iterator = zip(records, results)
        for record, scan in tqdm(
            iterator,
            total=len(records),
            desc="doctor: scanning images",
            disable=not progress,
            leave=False,
        ):
            record.scan = scan
    snapshot.images_scanned = True


def _scan_one(path: Path) -> ImageScan:
    try:
        data = path.read_bytes()
    except OSError as exc:
        return ImageScan(ok=False, error=f"unreadable: {exc}")
    if not data:
        return ImageScan(ok=False, error="zero-byte file")

    sha1 = hashlib.sha1(data).hexdigest()
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.verify()
        with Image.open(io.BytesIO(data)) as im:
            width, height = im.size
            mode = im.mode
            orientation = 0
            try:
                orientation = int(im.getexif().get(_EXIF_ORIENTATION_TAG, 0) or 0)
            except Exception:  # EXIF parsing is best-effort
                pass
            # Force a full decode (verify() misses truncated payloads) and
            # derive the dHash from a 9x8 grayscale thumbnail.
            gray = np.asarray(im.convert("L").resize((9, 8)), dtype=np.int16)
    except Exception as exc:
        return ImageScan(ok=False, error=f"{type(exc).__name__}: {exc}", sha1=sha1)

    bits = (gray[:, 1:] > gray[:, :-1]).flatten()
    dhash = int.from_bytes(np.packbits(bits).tobytes(), "big")
    uniform = int(gray.max() - gray.min()) <= 2
    return ImageScan(
        ok=True,
        width=width,
        height=height,
        mode=mode,
        exif_orientation=orientation,
        sha1=sha1,
        dhash=dhash,
        uniform=uniform,
    )


def hamming(a: int, b: int) -> int:
    """Hamming distance between two 64-bit dHashes."""
    return bin(a ^ b).count("1")
