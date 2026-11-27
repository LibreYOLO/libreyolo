"""OCR dataset resolution: images plus JSONL polygon/text labels.

Layout (documented in ``docs/dataset_schema.md``)::

    <root>/
      images/<split>/  name.jpg ...
      labels/<split>.jsonl

Each JSONL line describes one image::

    {"image": "name.jpg",
     "regions": [{"polygon": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]], "text": "..."}]}

Polygons are 4-point quads in absolute pixel coordinates (tl, tr, br, bl).
Regions whose text cannot be read use ``"text": "###"`` (the ICDAR don't-care
convention): they are excluded from recognition scoring and predictions
overlapping them are not penalized in detection matching.

A dataset YAML may point at the root and rename the pieces::

    path: /abs/or/rel/root
    images: images          # dir containing <split>/ subdirs
    labels: labels          # dir containing <split>.jsonl files
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

DONT_CARE = "###"


def _load_jsonl(path: Path) -> List[dict]:
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return entries


def _parse_regions(entry: dict, source: str) -> List[Dict]:
    regions = []
    for i, region in enumerate(entry.get("regions", [])):
        polygon = np.asarray(region.get("polygon"), dtype=np.float32)
        if polygon.shape != (4, 2):
            raise ValueError(
                f"{source}: regions[{i}].polygon must be 4x2, got {polygon.shape}"
            )
        regions.append({"polygon": polygon, "text": str(region.get("text", ""))})
    return regions


def resolve_ocr_root(data) -> Tuple[Path, str, str]:
    """``(root, images_dirname, labels_dirname)`` of an OCR dataset root or YAML.

    A YAML's ``path`` is relative to the YAML's folder, and defaults to it;
    ``images`` and ``labels`` default to ``images`` and ``labels``.
    """
    data_path = Path(str(data))
    if data_path.suffix.lower() in (".yaml", ".yml") and data_path.is_file():
        import yaml

        with open(data_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        root = data_path.parent / Path(str(cfg.get("path", ".")))
        return (
            root.resolve(),
            str(cfg.get("images", "images")),
            str(cfg.get("labels", "labels")),
        )
    if data_path.is_dir():
        return data_path, "images", "labels"
    raise ValueError(
        f"OCR data not found: {data} (expected a dataset root "
        "directory or a dataset YAML; see docs/dataset_schema.md)."
    )


def resolve_ocr_samples(data, split: str = "val") -> List[Tuple[Path, List[Dict]]]:
    """Resolve ``data`` (a root directory or dataset YAML) to labelled samples.

    Returns a sorted list of ``(image_path, regions)`` where each region is
    ``{"polygon": (4, 2) float32 array, "text": str}``.
    """
    root, images_dirname, labels_dirname = resolve_ocr_root(data)
    image_dir = root / images_dirname / split
    label_file = root / labels_dirname / f"{split}.jsonl"
    if not image_dir.is_dir():
        raise ValueError(f"OCR image split directory not found: {image_dir}")
    if not label_file.is_file():
        raise ValueError(f"OCR label file not found: {label_file}")

    labels: Dict[str, List[Dict]] = {}
    for entry in _load_jsonl(label_file):
        name = entry.get("image")
        if not name:
            raise ValueError(f"{label_file}: every entry needs an 'image' key")
        labels[str(name)] = _parse_regions(entry, f"{label_file} ({name})")

    samples = []
    for image_path in sorted(image_dir.iterdir()):
        if image_path.suffix.lower() not in _IMAGE_EXTS:
            continue
        if image_path.name in labels:
            samples.append((image_path, labels[image_path.name]))
    if not samples:
        raise ValueError(
            f"No labelled images found: nothing in {image_dir} matches an "
            f"'image' entry in {label_file}."
        )
    return samples


__all__ = ["resolve_ocr_samples", "DONT_CARE"]
