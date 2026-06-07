"""COCO person-keypoints to YOLO-pose conversion helpers.

Clean-room implementation from the public COCO keypoints JSON format and
LibreYOLO's documented YOLO-pose label contract.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .utils import get_img_files, img2label_paths


def _image_id_map(coco: dict) -> dict[int, dict]:
    return {int(img["id"]): img for img in coco.get("images", [])}


def _selected_image_names(images_dir: Path) -> set[str]:
    try:
        return {path.name for path in get_img_files(images_dir)}
    except (FileNotFoundError, ValueError):
        return set()


def _format_float(value: float) -> str:
    return f"{float(value):.6f}".rstrip("0").rstrip(".") or "0"


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def convert_coco_keypoints_json_to_yolo_pose(
    json_file: str | Path,
    images_dir: str | Path,
    labels_dir: str | Path | None = None,
    *,
    category_id: int = 1,
    class_id: int = 0,
    num_keypoints: int = 17,
    write_empty: bool = True,
) -> dict[str, int]:
    """Convert one COCO keypoints JSON split into YOLO-pose label files.

    Args:
        json_file: COCO ``person_keypoints_*.json`` annotation file.
        images_dir: Directory containing the split's images. Only annotations
            whose ``file_name`` exists in this directory are converted.
        labels_dir: Destination label directory. Defaults to the standard
            ``images/... -> labels/...`` mapping for files under ``images_dir``.
        category_id: COCO category id to keep. COCO person is ``1``.
        class_id: YOLO class id to write. Person-only pose datasets use ``0``.
        num_keypoints: Expected number of keypoints per instance.
        write_empty: Create empty label files for images with no valid person
            keypoints so dataset scans are deterministic.

    Returns:
        A small summary with image, annotation, and skipped counts.
    """
    json_path = Path(json_file)
    images_path = Path(images_dir)
    if labels_dir is None:
        image_files = get_img_files(images_path)
        label_files = img2label_paths(image_files)
        label_root_by_name = {img.name: lbl for img, lbl in zip(image_files, label_files)}
    else:
        label_root = Path(labels_dir)
        label_root_by_name = {}

    with json_path.open("r", encoding="utf-8") as fh:
        coco = json.load(fh)

    images = _image_id_map(coco)
    selected_names = _selected_image_names(images_path)
    rows_by_name: dict[str, list[str]] = {name: [] for name in selected_names}
    converted = 0
    skipped = 0

    for ann in coco.get("annotations", []):
        if int(ann.get("category_id", -1)) != int(category_id):
            continue
        image = images.get(int(ann.get("image_id", -1)))
        if not image:
            skipped += 1
            continue
        file_name = Path(str(image.get("file_name", ""))).name
        if file_name not in selected_names:
            continue
        keypoints = ann.get("keypoints") or []
        if len(keypoints) != int(num_keypoints) * 3:
            skipped += 1
            continue
        if int(ann.get("num_keypoints", 0)) <= 0:
            skipped += 1
            continue
        bbox = ann.get("bbox") or []
        if len(bbox) != 4:
            skipped += 1
            continue
        width = float(image.get("width", 0))
        height = float(image.get("height", 0))
        if width <= 0 or height <= 0:
            skipped += 1
            continue

        x, y, w, h = [float(v) for v in bbox]
        if w <= 0 or h <= 0:
            skipped += 1
            continue
        values = [
            float(class_id),
            _clamp01((x + w * 0.5) / width),
            _clamp01((y + h * 0.5) / height),
            _clamp01(w / width),
            _clamp01(h / height),
        ]
        for idx in range(int(num_keypoints)):
            kx = float(keypoints[idx * 3])
            ky = float(keypoints[idx * 3 + 1])
            vis = float(keypoints[idx * 3 + 2])
            if vis <= 0 or kx < 0.0 or kx > width or ky < 0.0 or ky > height:
                values.extend([0.0, 0.0, 0.0])
            else:
                values.extend([_clamp01(kx / width), _clamp01(ky / height), vis])
        rows_by_name.setdefault(file_name, []).append(" ".join(_format_float(v) for v in values))
        converted += 1

    written = 0
    for file_name, rows in rows_by_name.items():
        if not rows and not write_empty:
            continue
        if labels_dir is None:
            label_path = label_root_by_name[file_name]
        else:
            label_path = Path(labels_dir) / Path(file_name).with_suffix(".txt").name
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text(("\n".join(rows) + ("\n" if rows else "")), encoding="utf-8")
        written += 1

    return {
        "images": len(selected_names),
        "labels": written,
        "annotations": converted,
        "skipped": skipped,
    }


def convert_coco_keypoints_splits(
    splits: Iterable[tuple[str | Path, str | Path, str | Path]],
    *,
    category_id: int = 1,
    class_id: int = 0,
    num_keypoints: int = 17,
) -> list[dict[str, int]]:
    """Convert multiple ``(json_file, images_dir, labels_dir)`` splits."""
    return [
        convert_coco_keypoints_json_to_yolo_pose(
            json_file,
            images_dir,
            labels_dir,
            category_id=category_id,
            class_id=class_id,
            num_keypoints=num_keypoints,
        )
        for json_file, images_dir, labels_dir in splits
    ]


__all__ = [
    "convert_coco_keypoints_json_to_yolo_pose",
    "convert_coco_keypoints_splits",
]
