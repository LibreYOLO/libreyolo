"""Detect-dataset reader for VLM fine-tuning.

Reads the standard LibreYOLO detect dataset (``docs/dataset_schema.md``: YAML
with ``train``/``val`` image sources and txt label rows ``class cx cy w h``
normalized to [0, 1]) and yields PIL images with rendered conversation targets.
The user never writes conversations or coordinate text; that is the library's
job, via :mod:`.targets`.

Augmentation is geometric-safe only (horizontal flip), because every geometric
transform must re-render the target text; the flip happens before rendering.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image

from ....data import get_img_files, img2label_paths
from .targets import FamilyFormat, serialize_detections

logger = logging.getLogger(__name__)

__all__ = ["VLMDetectDataset"]


def _parse_label_file(path: Path) -> List[Tuple[int, float, float, float, float]]:
    """Parse one txt label file into ``(cls, cx, cy, w, h)`` rows.

    Missing file means no objects (dataset-schema rule). Malformed rows are
    skipped with a warning rather than aborting a long training run.
    """
    if not path.exists():
        return []
    rows: List[Tuple[int, float, float, float, float]] = []
    for line_no, line in enumerate(
        path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
    ):
        parts = line.split()
        if not parts:
            continue
        try:
            cls = int(float(parts[0]))
            cx, cy, w, h = (float(v) for v in parts[1:5])
        except (ValueError, IndexError):
            logger.warning("Skipping malformed label row %s:%d: %r", path, line_no, line)
            continue
        if len(parts) != 5:
            logger.warning(
                "Skipping label row with %d fields (detect rows have 5) %s:%d",
                len(parts),
                path,
                line_no,
            )
            continue
        rows.append((cls, cx, cy, w, h))
    return rows


class VLMDetectDataset:
    """Map-style dataset: detect labels in, conversation samples out.

    Each item is a dict with ``image`` (PIL, RGB), ``prompt`` (the family's
    detection prompt for the training vocabulary), and ``target`` (the JSON
    answer text rendered by :func:`serialize_detections`).
    """

    def __init__(
        self,
        image_source,
        names: Dict[int, str],
        fmt: FamilyFormat,
        augment: bool = False,
        hflip_p: float = 0.5,
        seed: int = 0,
    ) -> None:
        self.names = dict(names)
        self.fmt = fmt
        self.augment = augment
        self.hflip_p = float(hflip_p)
        self._rng = random.Random(seed)
        self.images: List[Path] = get_img_files(image_source)
        if not self.images:
            raise FileNotFoundError(f"No images found under {image_source!r}")
        self.labels: List[Path] = img2label_paths(self.images)
        self._warned_unknown_class = False

    def __len__(self) -> int:
        return len(self.images)

    def _load_boxes(self, index: int) -> Tuple[List[List[float]], List[str]]:
        boxes: List[List[float]] = []
        labels: List[str] = []
        for cls, cx, cy, w, h in _parse_label_file(self.labels[index]):
            name = self.names.get(cls)
            if name is None:
                if not self._warned_unknown_class:
                    logger.warning(
                        "Label file %s references class id %d not present in the "
                        "dataset names; such rows are skipped.",
                        self.labels[index],
                        cls,
                    )
                    self._warned_unknown_class = True
                continue
            x1 = cx - w / 2.0
            y1 = cy - h / 2.0
            x2 = cx + w / 2.0
            y2 = cy + h / 2.0
            boxes.append(
                [
                    min(max(x1, 0.0), 1.0),
                    min(max(y1, 0.0), 1.0),
                    min(max(x2, 0.0), 1.0),
                    min(max(y2, 0.0), 1.0),
                ]
            )
            labels.append(name)
        return boxes, labels

    def __getitem__(self, index: int) -> Dict[str, object]:
        image = Image.open(self.images[index]).convert("RGB")
        boxes, labels = self._load_boxes(index)
        if self.augment and self._rng.random() < self.hflip_p:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            boxes = [[1.0 - x2, y1, 1.0 - x1, y2] for x1, y1, x2, y2 in boxes]
        target = serialize_detections(boxes, labels, self.fmt)
        return {
            "image": image,
            "prompt": self.fmt.detection_prompt,
            "target": target,
        }


def resolve_split_source(data_cfg: Dict, split: str) -> Optional[object]:
    """Return the image source for a split, or None if the split is absent.

    Native COCO JSON datasets (``annotations:`` in the YAML) are not supported
    by the VLM trainer yet; the caller gets a clear error instead of silently
    training without boxes.
    """
    annotations = data_cfg.get("annotations")
    has_coco_json = (
        isinstance(annotations, dict) and annotations.get(split)
    ) or data_cfg.get(f"{split}_annotation_file")
    if has_coco_json:
        raise NotImplementedError(
            "VLM training reads txt label files (docs/dataset_schema.md); native "
            "COCO JSON datasets are not supported yet. Convert the annotations "
            "to txt labels to train a VLM on this dataset."
        )
    return data_cfg.get(split)
