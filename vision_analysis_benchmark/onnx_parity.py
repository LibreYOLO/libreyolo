#!/usr/bin/env python3
"""Run ONNX-vs-PyTorch validation parity checks for LibreYOLO models.

The default matrix covers model families that claim ONNX export support in the
project support table, limited to detection, instance segmentation, and pose.
Generated reports are written under ``runs/onnx_parity`` by default.

Example:
    python vision_analysis_benchmark/onnx_parity.py \
        --download-mini500 \
        --families yolo9 rfdetr ec \
        --tasks detect segment pose \
        --batch 16 \
        --device cuda:0
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
import traceback
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable
from urllib.parse import quote

import requests
import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MINI500_DATASET_ID = "LibreYOLO/coco-val2017-mini500"
MINI500_ANNOTATION = "annotations/instances_val2017_mini500.json"
MINI500_POSE_ANNOTATION = "annotations/person_keypoints_val2017_mini500.json"
MINI500_IMAGE_DIR = "images/val2017"
COCO_ANNOTATIONS_URL = "https://images.cocodataset.org/annotations/annotations_trainval2017.zip"
COCO_PERSON_KEYPOINTS_MEMBER = "annotations/person_keypoints_val2017.json"
COCO_POSE_FLIP_IDX = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]

COCO80_NAMES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "airplane",
    5: "bus",
    6: "train",
    7: "truck",
    8: "boat",
    9: "traffic light",
    10: "fire hydrant",
    11: "stop sign",
    12: "parking meter",
    13: "bench",
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe",
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    27: "tie",
    28: "suitcase",
    29: "frisbee",
    30: "skis",
    31: "snowboard",
    32: "sports ball",
    33: "kite",
    34: "baseball bat",
    35: "baseball glove",
    36: "skateboard",
    37: "surfboard",
    38: "tennis racket",
    39: "bottle",
    40: "wine glass",
    41: "cup",
    42: "fork",
    43: "knife",
    44: "spoon",
    45: "bowl",
    46: "banana",
    47: "apple",
    48: "sandwich",
    49: "orange",
    50: "broccoli",
    51: "carrot",
    52: "hot dog",
    53: "pizza",
    54: "donut",
    55: "cake",
    56: "chair",
    57: "couch",
    58: "potted plant",
    59: "bed",
    60: "dining table",
    61: "toilet",
    62: "tv",
    63: "laptop",
    64: "mouse",
    65: "remote",
    66: "keyboard",
    67: "cell phone",
    68: "microwave",
    69: "oven",
    70: "toaster",
    71: "sink",
    72: "refrigerator",
    73: "book",
    74: "clock",
    75: "vase",
    76: "scissors",
    77: "teddy bear",
    78: "hair drier",
    79: "toothbrush",
}

TASK_SUFFIX = {
    "detect": "",
    "segment": "-seg",
    "pose": "-pose",
}

DETECT_KEYS = (
    "metrics/mAP50-95",
    "metrics/mAP50",
    "metrics/precision",
    "metrics/recall",
)
SEGMENT_KEYS = (
    "metrics/mAP50-95(B)",
    "metrics/mAP50(B)",
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50-95(M)",
    "metrics/mAP50(M)",
    "metrics/precision(M)",
    "metrics/recall(M)",
)
POSE_KEYS = (
    "metrics/keypoints_mAP50-95",
    "metrics/keypoints_mAP50",
    "metrics/keypoints_mAP75",
    "metrics/keypoints_AR50-95",
    "metrics/keypoints_AR50",
    "metrics/keypoints_AR75",
)


@dataclass(frozen=True)
class ParityCase:
    id: str
    family: str
    size: str
    task: str
    weights: str
    onnx_claim: str
    notes: str = ""


@dataclass(frozen=True)
class MetricComparison:
    key: str
    pytorch: float | None
    onnx: float | None
    abs_delta: float | None
    tolerance: float
    passed: bool
    reason: str = ""


class CheckpointUnavailableError(RuntimeError):
    """Raised when a selected parity checkpoint cannot be obtained."""


def _checkpoint_unavailable_reason(exc: BaseException) -> str | None:
    """Return a reportable reason when a parity case lacks usable weights."""
    messages: list[str] = []
    current: BaseException | None = exc
    while current is not None:
        message = str(current)
        if message:
            messages.append(message)
        current = current.__cause__ or current.__context__

    text = "\n".join(messages).lower()
    if "model weights file not found" in text:
        return str(exc)
    if "pretrained weights are not available" in text:
        return str(exc)
    if "could not determine download url" in text:
        return str(exc)
    if "no weights are published at" in text:
        return str(exc)
    if "failed to download weights" in text and any(
        marker in text
        for marker in (
            "401 client error",
            "403 client error",
            "404 client error",
            "410 client error",
            "no such bucket",
            "repo not found",
            "repository not found",
        )
    ):
        return str(exc)
    return None


def _ensure_case_weights_available(
    case: ParityCase,
    *,
    resolve_weights_path=None,
    downloader=None,
) -> None:
    """Download missing checkpoint weights before model construction.

    The LibreYOLO factory logs auto-download failures and then raises a generic
    missing-file error. Preflight keeps the original HTTP/download exception
    attached to the parity record, which matters for unavailable checkpoints.
    """
    if resolve_weights_path is None:
        from libreyolo.models import _resolve_weights_path as resolve_weights_path
    if downloader is None:
        from libreyolo.utils.download import download_weights as downloader

    weights_path = Path(resolve_weights_path(case.weights))
    if weights_path.exists():
        return
    downloader(str(weights_path), case.size)


def _preflight_case_weights(case: ParityCase) -> None:
    try:
        _ensure_case_weights_available(case)
    except Exception as exc:
        reason = _checkpoint_unavailable_reason(exc)
        if reason is not None:
            raise CheckpointUnavailableError(reason) from exc
        raise


def _load_onnx_for_parity(
    onnx_path: Path,
    case: ParityCase,
    args: argparse.Namespace,
    *,
    loader=None,
):
    """Load exported ONNX through its own metadata and verify task resolution."""
    if loader is None:
        from libreyolo import LibreYOLO as loader

    model = loader(str(onnx_path), device=args.onnx_device)
    resolved_task = getattr(model, "task", None)
    if resolved_task != case.task:
        raise RuntimeError(
            f"ONNX artifact task metadata resolved to {resolved_task!r}, "
            f"expected {case.task!r} for parity case {case.id!r}."
        )
    return model


def _weights(prefix: str, size: str, task: str) -> str:
    return f"{prefix}{size}{TASK_SUFFIX[task]}.pt"


def _case(
    family: str,
    prefix: str,
    size: str,
    task: str,
    claim: str,
    notes: str = "",
) -> ParityCase:
    suffix = "" if task == "detect" else f"-{task}"
    return ParityCase(
        id=f"{family}-{size}{suffix}",
        family=family,
        size=size,
        task=task,
        weights=_weights(prefix, size, task),
        onnx_claim=claim,
        notes=notes,
    )


def build_default_cases() -> list[ParityCase]:
    cases: list[ParityCase] = []

    def add_many(
        family: str,
        prefix: str,
        sizes: Iterable[str],
        task: str,
        claim: str,
        notes: str = "",
    ) -> None:
        cases.extend(_case(family, prefix, size, task, claim, notes) for size in sizes)

    add_many("yolo9", "LibreYOLO9", ("t", "s", "m", "c"), "detect", "complete")
    add_many(
        "yolo9",
        "LibreYOLO9",
        ("s",),
        "pose",
        "complete",
        "checkpoint converted from upstream early-access release",
    )

    add_many("rfdetr", "LibreRFDETR", ("n", "s", "m", "l"), "detect", "complete")
    add_many(
        "rfdetr",
        "LibreRFDETR",
        ("n", "s", "m", "l", "x", "xx"),
        "segment",
        "complete",
    )
    add_many(
        "rfdetr",
        "LibreRFDETR",
        ("n", "s", "m", "l", "x"),
        "pose",
        "complete",
        "checkpoint converted from upstream early-access release",
    )

    add_many(
        "yolox",
        "LibreYOLOX",
        ("n", "t", "s", "m", "l", "x"),
        "detect",
        "available",
    )
    add_many(
        "yolo9_e2e",
        "LibreYOLO9E2E",
        ("t", "s", "m", "c"),
        "detect",
        "available",
    )
    add_many("yolonas", "LibreYOLONAS", ("s", "m", "l"), "detect", "available")
    add_many(
        "yolonas",
        "LibreYOLONAS",
        ("n", "s", "m", "l"),
        "pose",
        "available",
        "pose weights download from Deci CDN with pinned checksums",
    )
    add_many("dfine", "LibreDFINE", ("n", "s", "m", "l", "x"), "detect", "available")
    add_many("deim", "LibreDEIM", ("n", "s", "m", "l", "x"), "detect", "available")
    add_many(
        "deimv2",
        "LibreDEIMv2",
        ("atto", "femto", "pico", "n", "s", "m", "l", "x"),
        "detect",
        "available",
    )
    add_many(
        "rtdetr",
        "LibreRTDETR",
        ("r18", "r34", "r50", "r50m", "r101", "l", "x"),
        "detect",
        "available",
    )
    add_many(
        "rtdetrv2",
        "LibreRTDETRv2",
        ("r18", "r34", "r50", "r50m", "r101"),
        "detect",
        "available",
    )
    add_many("rtdetrv4", "LibreRTDETRv4", ("s", "m", "l", "x"), "detect", "available")
    add_many("picodet", "LibrePICODET", ("s", "m", "l"), "detect", "available")
    add_many("rtmdet", "LibreRTMDet", ("t", "s", "m", "l", "x"), "detect", "available")
    add_many("ec", "LibreEC", ("s", "m", "l", "x"), "detect", "available")
    add_many("ec", "LibreEC", ("s", "m", "l", "x"), "segment", "available")
    add_many("ec", "LibreEC", ("s", "m", "l", "x"), "pose", "available")
    return cases


DEFAULT_CASES = tuple(build_default_cases())


def metric_keys_for_task(task: str) -> tuple[str, ...]:
    if task == "segment":
        return SEGMENT_KEYS
    if task == "pose":
        return POSE_KEYS
    return DETECT_KEYS


def extract_metrics(metrics: dict[str, Any], task: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in metric_keys_for_task(task):
        value = metrics.get(key)
        if value is None:
            continue
        try:
            out[key] = float(value)
        except (TypeError, ValueError):
            continue
    return out


def compare_metrics(
    pytorch: dict[str, float],
    onnx: dict[str, float],
    task: str,
    *,
    abs_tolerance: float,
    rel_tolerance: float,
) -> list[MetricComparison]:
    comparisons: list[MetricComparison] = []
    for key in metric_keys_for_task(task):
        pt_value = pytorch.get(key)
        onnx_value = onnx.get(key)
        if pt_value is None or onnx_value is None:
            comparisons.append(
                MetricComparison(
                    key=key,
                    pytorch=pt_value,
                    onnx=onnx_value,
                    abs_delta=None,
                    tolerance=abs_tolerance,
                    passed=False,
                    reason="missing metric",
                )
            )
            continue
        if not math.isfinite(pt_value) or not math.isfinite(onnx_value):
            comparisons.append(
                MetricComparison(
                    key=key,
                    pytorch=pt_value,
                    onnx=onnx_value,
                    abs_delta=None,
                    tolerance=abs_tolerance,
                    passed=False,
                    reason="non-finite metric",
                )
            )
            continue
        delta = abs(onnx_value - pt_value)
        tolerance = max(abs_tolerance, rel_tolerance * abs(pt_value))
        comparisons.append(
            MetricComparison(
                key=key,
                pytorch=pt_value,
                onnx=onnx_value,
                abs_delta=delta,
                tolerance=tolerance,
                passed=delta <= tolerance,
            )
        )
    return comparisons


def filter_cases(
    cases: Iterable[ParityCase],
    *,
    ids: Iterable[str] | None = None,
    families: Iterable[str] | None = None,
    tasks: Iterable[str] | None = None,
    claims: Iterable[str] | None = None,
) -> list[ParityCase]:
    ids_set = {x.strip() for x in ids or [] if x.strip()}
    family_set = {x.strip() for x in families or [] if x.strip()}
    task_set = {x.strip() for x in tasks or [] if x.strip()}
    claim_set = {x.strip() for x in claims or [] if x.strip()}

    selected = []
    for case in cases:
        if ids_set and case.id not in ids_set:
            continue
        if family_set and case.family not in family_set:
            continue
        if task_set and case.task not in task_set:
            continue
        if claim_set and case.onnx_claim not in claim_set:
            continue
        selected.append(case)
    return selected


def _headers(token: str | None = None) -> dict[str, str]:
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _safe_hf_dataset_path(raw_path: str, dest: Path) -> tuple[PurePosixPath, Path]:
    raw = str(raw_path)
    if "\\" in raw or "\x00" in raw:
        raise ValueError(f"Unsafe Hugging Face dataset path: {raw_path!r}")
    rel = PurePosixPath(raw)
    if rel.is_absolute() or any(part in {"..", ""} or ":" in part for part in rel.parts):
        raise ValueError(f"Unsafe Hugging Face dataset path: {raw_path!r}")

    dest_root = dest.resolve()
    target = dest_root.joinpath(*rel.parts)
    try:
        target.resolve().relative_to(dest_root)
    except ValueError as exc:
        raise ValueError(f"Unsafe Hugging Face dataset path: {raw_path!r}") from exc
    return rel, target


def download_hf_dataset_snapshot(
    repo_id: str,
    dest: Path,
    *,
    revision: str = "main",
    token: str | None = None,
) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    api_url = f"https://huggingface.co/api/datasets/{repo_id}/tree/{revision}?recursive=true"
    response = requests.get(api_url, headers=_headers(token), timeout=60)
    response.raise_for_status()
    files = [item for item in response.json() if item.get("type") == "file"]

    for item in files:
        rel_path, target = _safe_hf_dataset_path(str(item["path"]), dest)
        expected_size = int(item.get("size") or 0)
        if target.exists() and expected_size > 0 and target.stat().st_size == expected_size:
            continue

        target.parent.mkdir(parents=True, exist_ok=True)
        url = (
            f"https://huggingface.co/datasets/{repo_id}/resolve/"
            f"{quote(revision)}/{quote(str(rel_path))}"
        )
        download_headers = {}
        if token:
            download_headers["Authorization"] = f"Bearer {token}"
        _download_atomic(url, target, headers=download_headers)
        actual_size = target.stat().st_size
        if expected_size > 0 and actual_size != expected_size:
            target.unlink(missing_ok=True)
            raise IOError(
                f"Downloaded {target} has size {actual_size}, expected {expected_size}."
            )
    return dest


def write_mini500_yaml(dataset_root: Path) -> Path:
    annotation_path = dataset_root / MINI500_ANNOTATION
    image_dir = dataset_root / MINI500_IMAGE_DIR
    if not annotation_path.exists():
        raise FileNotFoundError(f"Mini500 annotation file not found: {annotation_path}")
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Mini500 image directory not found: {image_dir}")

    yaml_path = dataset_root / "coco-val2017-mini500.yaml"
    config = {
        "path": str(dataset_root.resolve()),
        "train": MINI500_IMAGE_DIR,
        "val": MINI500_IMAGE_DIR,
        "annotations": {"val": MINI500_ANNOTATION},
        "nc": 80,
        "names": COCO80_NAMES,
    }
    yaml_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return yaml_path


def _download_atomic(
    url: str,
    target: Path,
    *,
    headers: dict[str, str] | None = None,
) -> None:
    tmp_path = target.with_name(f"{target.name}.tmp")
    tmp_path.unlink(missing_ok=True)
    try:
        with requests.get(
            url,
            headers=headers or {},
            stream=True,
            timeout=60,
        ) as response:
            response.raise_for_status()
            with open(tmp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
        tmp_path.replace(target)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _download_coco_person_keypoints(dataset_root: Path) -> Path:
    annotations_dir = dataset_root / "annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)
    target = dataset_root / COCO_PERSON_KEYPOINTS_MEMBER
    if target.exists():
        return target

    zip_path = annotations_dir / "annotations_trainval2017.zip"
    for attempt in range(2):
        if not zip_path.exists():
            _download_atomic(COCO_ANNOTATIONS_URL, zip_path)
        try:
            with zipfile.ZipFile(zip_path) as archive:
                archive.extract(COCO_PERSON_KEYPOINTS_MEMBER, path=dataset_root)
            break
        except zipfile.BadZipFile:
            zip_path.unlink(missing_ok=True)
            if attempt == 1:
                raise
    return target


def write_mini500_pose_yaml(
    dataset_root: Path,
    *,
    full_keypoints_path: Path | None = None,
) -> Path:
    instance_path = dataset_root / MINI500_ANNOTATION
    image_dir = dataset_root / MINI500_IMAGE_DIR
    if not instance_path.exists():
        raise FileNotFoundError(f"Mini500 annotation file not found: {instance_path}")
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Mini500 image directory not found: {image_dir}")

    if full_keypoints_path is None:
        full_keypoints_path = _download_coco_person_keypoints(dataset_root)
    if not full_keypoints_path.exists():
        raise FileNotFoundError(
            f"COCO person-keypoint annotation file not found: {full_keypoints_path}"
        )

    instances = json.loads(instance_path.read_text(encoding="utf-8"))
    keypoints = json.loads(full_keypoints_path.read_text(encoding="utf-8"))
    image_ids = {int(image["id"]) for image in instances.get("images", [])}
    person_categories = [
        category
        for category in keypoints.get("categories", [])
        if category.get("name") == "person"
    ]
    category = dict(person_categories[0] if person_categories else {"name": "person"})
    category["id"] = 0

    annotations = []
    for ann in keypoints.get("annotations", []):
        if int(ann.get("image_id", -1)) not in image_ids:
            continue
        item = dict(ann)
        item["category_id"] = 0
        annotations.append(item)

    filtered = {
        "info": keypoints.get("info", {}),
        "licenses": keypoints.get("licenses", []),
        "images": [
            image
            for image in keypoints.get("images", [])
            if int(image.get("id", -1)) in image_ids
        ],
        "annotations": annotations,
        "categories": [category],
    }

    pose_annotation = dataset_root / MINI500_POSE_ANNOTATION
    pose_annotation.parent.mkdir(parents=True, exist_ok=True)
    pose_annotation.write_text(
        json.dumps(filtered, separators=(",", ":")),
        encoding="utf-8",
    )

    yaml_path = dataset_root / "coco-val2017-mini500-pose.yaml"
    config = {
        "path": str(dataset_root.resolve()),
        "train": MINI500_IMAGE_DIR,
        "val": MINI500_IMAGE_DIR,
        "annotations": {"val": MINI500_POSE_ANNOTATION},
        "nc": 1,
        "names": {0: "person"},
        "kpt_shape": [17, 3],
        "flip_idx": COCO_POSE_FLIP_IDX,
    }
    yaml_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return yaml_path


def _dataset_root_from_yaml(data_yaml: Path) -> Path:
    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    root = config.get("path")
    if root is None:
        return data_yaml.parent
    root_path = Path(root)
    return root_path if root_path.is_absolute() else (data_yaml.parent / root_path)


def _resolve_yaml_path(root: Path, value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def _looks_like_pose_annotation(value: Any) -> bool:
    return "keypoints" in str(value).lower()


def _is_coco_keypoints_annotation_file(path: Path) -> bool:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    categories = data.get("categories") or []
    if any(cat.get("keypoints") for cat in categories):
        return True
    annotations = data.get("annotations") or []
    return any("keypoints" in ann and "num_keypoints" in ann for ann in annotations)


def _annotation_config_values(
    config: dict[str, Any],
    *,
    split: str | None = None,
) -> list[Any]:
    values: list[Any] = []
    if config.get("keypoints_json"):
        values.append(config["keypoints_json"])
    annotations = config.get("annotations") or {}
    if isinstance(annotations, dict):
        if split is None:
            values.extend(annotations.values())
        elif annotations.get(split):
            values.append(annotations[split])
    elif annotations:
        values.append(annotations)
    return values


def _has_pose_annotation_config(
    config: dict[str, Any],
    root: Path | None = None,
    *,
    split: str | None = None,
) -> bool:
    for value in _annotation_config_values(config, split=split):
        if not value:
            continue
        if root is not None:
            annotation_path = _resolve_yaml_path(root, value)
            if annotation_path.exists():
                if _is_coco_keypoints_annotation_file(annotation_path):
                    return True
                continue
            continue
        if _looks_like_pose_annotation(value):
            return True
    return False


def prepare_pose_data(data_yaml: Path, *, split: str = "val") -> Path:
    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    if config.get("kpt_shape"):
        if data_yaml.name == "coco-val2017-mini500-pose.yaml":
            return write_mini500_pose_yaml(_dataset_root_from_yaml(data_yaml))
        return data_yaml
    if _has_pose_annotation_config(
        config,
        _dataset_root_from_yaml(data_yaml),
        split=split,
    ):
        return data_yaml
    return write_mini500_pose_yaml(_dataset_root_from_yaml(data_yaml))


def pose_direct_paths(
    data_yaml: Path,
    *,
    split: str = "val",
) -> tuple[Path | None, Path | None]:
    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    root = _dataset_root_from_yaml(data_yaml)
    annotations = config.get("annotations") or {}
    annotation_rel = config.get("keypoints_json")
    if annotation_rel is None:
        annotation_rel = (
            annotations.get(split) if isinstance(annotations, dict) else annotations
        )
    images_rel = config.get("images_dir") or config.get(split)
    if not annotation_rel or not images_rel:
        return None, None
    annotation_path = _resolve_yaml_path(root, annotation_rel)
    if annotation_path.exists():
        if not _is_coco_keypoints_annotation_file(annotation_path):
            return None, None
    elif not _looks_like_pose_annotation(annotation_rel):
        return None, None
    return annotation_path, _resolve_yaml_path(root, images_rel)


def prepare_data(args: argparse.Namespace, output_dir: Path) -> Path:
    if args.data:
        return Path(args.data)
    if not args.download_mini500:
        raise ValueError("Pass --data PATH or --download-mini500.")

    dataset_root = Path(args.dataset_dir or output_dir / "datasets" / "coco-val2017-mini500")
    download_hf_dataset_snapshot(
        args.hf_dataset_id,
        dataset_root,
        revision=args.hf_revision,
        token=args.hf_token,
    )
    return write_mini500_yaml(dataset_root)


def _val_kwargs(
    args: argparse.Namespace,
    save_dir: Path,
    imgsz: int | None,
    *,
    data_yaml: Path | None = None,
    task: str | None = None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "data": str(data_yaml or args.data_yaml),
        "batch": args.batch,
        "conf": args.conf,
        "iou": args.iou,
        "workers": args.workers,
        "split": args.split,
        "augment": False,
        "save_json": args.save_json,
        "verbose": args.verbose,
        "allow_download_scripts": args.allow_download_scripts,
        "save_dir": str(save_dir),
        "save_plots": args.save_plots,
    }
    if args.val_device:
        kwargs["device"] = args.val_device
    if imgsz is not None:
        kwargs["imgsz"] = imgsz
    if task == "pose" and getattr(args, "pose_keypoints_json", None):
        kwargs["keypoints_json"] = str(args.pose_keypoints_json)
        kwargs["images_dir"] = str(args.pose_images_dir)
    return kwargs


def run_case(case: ParityCase, args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    from libreyolo import LibreYOLO

    started = time.perf_counter()
    case_dir = output_dir / "cases" / case.id
    case_dir.mkdir(parents=True, exist_ok=True)
    export_dir = output_dir / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = export_dir / f"{case.id}.onnx"
    imgsz = args.imgsz
    data_yaml = args.pose_data_yaml if case.task == "pose" else args.data_yaml

    record: dict[str, Any] = {
        "case": asdict(case),
        "status": "error",
        "onnx_path": str(onnx_path),
        "data": str(data_yaml),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        print(f"[{case.id}] loading PyTorch weights {case.weights}")
        _preflight_case_weights(case)
        pt_model = LibreYOLO(
            case.weights,
            size=case.size,
            task=case.task,
            device=args.device,
        )
        if imgsz is None:
            imgsz = int(pt_model._get_input_size())

        if not args.reuse_exports or not onnx_path.exists():
            print(f"[{case.id}] exporting ONNX -> {onnx_path}")
            export_kwargs = {
                "output_path": str(onnx_path),
                "simplify": args.simplify,
                "dynamic": args.dynamic,
                "opset": args.opset,
                "imgsz": imgsz,
            }
            pt_model.export("onnx", **export_kwargs)
        else:
            print(f"[{case.id}] reusing ONNX export {onnx_path}")

        print(f"[{case.id}] validating PyTorch")
        pt_metrics_raw = pt_model.val(
            **_val_kwargs(
                args,
                case_dir / "pytorch",
                imgsz,
                data_yaml=data_yaml,
                task=case.task,
            )
        )
        pt_metrics = extract_metrics(pt_metrics_raw, case.task)

        print(f"[{case.id}] validating ONNX")
        onnx_model = _load_onnx_for_parity(onnx_path, case, args)
        onnx_metrics_raw = onnx_model.val(
            **_val_kwargs(
                args,
                case_dir / "onnx",
                imgsz,
                data_yaml=data_yaml,
                task=case.task,
            )
        )
        onnx_metrics = extract_metrics(onnx_metrics_raw, case.task)

        comparisons = compare_metrics(
            pt_metrics,
            onnx_metrics,
            case.task,
            abs_tolerance=args.abs_tolerance,
            rel_tolerance=args.rel_tolerance,
        )
        passed = all(item.passed for item in comparisons)
        record.update(
            {
                "status": "passed" if passed else "failed",
                "imgsz": imgsz,
                "pytorch_metrics": pt_metrics,
                "onnx_metrics": onnx_metrics,
                "comparisons": [asdict(item) for item in comparisons],
            }
        )
    except CheckpointUnavailableError as exc:
        record.update(
            {
                "status": "unavailable",
                "error": str(exc),
                "unavailable_reason": str(exc),
            }
        )
        print(f"[{case.id}] UNAVAILABLE: {exc}", file=sys.stderr)
    except Exception as exc:
        record.update(
            {
                "status": "error",
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        print(f"[{case.id}] ERROR: {exc}", file=sys.stderr)
    finally:
        record["duration_seconds"] = round(time.perf_counter() - started, 3)
        (case_dir / "parity.json").write_text(
            json.dumps(record, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return record


def _runtime_report_config(args: argparse.Namespace) -> dict[str, Any]:
    providers: list[str] = []
    try:
        import onnxruntime as ort

        providers = list(ort.get_available_providers())
    except Exception:
        providers = []

    return {
        "data": str(args.data_yaml),
        "pose_data": str(getattr(args, "pose_data_yaml", "") or ""),
        "device": args.device,
        "onnx_device": args.onnx_device,
        "val_device": args.val_device,
        "batch": args.batch,
        "workers": args.workers,
        "split": args.split,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "opset": args.opset,
        "simplify": args.simplify,
        "dynamic": args.dynamic,
        "abs_tolerance": args.abs_tolerance,
        "rel_tolerance": args.rel_tolerance,
        "onnxruntime_providers": providers,
    }


def write_summary(
    records: list[dict[str, Any]],
    output_dir: Path,
    *,
    run_config: dict[str, Any] | None = None,
) -> None:
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_config": run_config or {},
        "counts": {
            "total": len(records),
            "passed": sum(1 for r in records if r.get("status") == "passed"),
            "failed": sum(1 for r in records if r.get("status") == "failed"),
            "error": sum(1 for r in records if r.get("status") == "error"),
            "unavailable": sum(
                1 for r in records if r.get("status") == "unavailable"
            ),
        },
        "records": records,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    csv_path = output_dir / "summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case",
                "family",
                "size",
                "task",
                "claim",
                "status",
                "metric",
                "pytorch",
                "onnx",
                "abs_delta",
                "tolerance",
                "reason",
                "error",
            ],
        )
        writer.writeheader()
        for record in records:
            case = record["case"]
            comparisons = record.get("comparisons") or []
            if not comparisons:
                writer.writerow(
                    {
                        "case": case["id"],
                        "family": case["family"],
                        "size": case["size"],
                        "task": case["task"],
                        "claim": case["onnx_claim"],
                        "status": record.get("status"),
                        "error": record.get("error", ""),
                    }
                )
                continue
            for comparison in comparisons:
                writer.writerow(
                    {
                        "case": case["id"],
                        "family": case["family"],
                        "size": case["size"],
                        "task": case["task"],
                        "claim": case["onnx_claim"],
                        "status": record.get("status"),
                        "metric": comparison["key"],
                        "pytorch": comparison["pytorch"],
                        "onnx": comparison["onnx"],
                        "abs_delta": comparison["abs_delta"],
                        "tolerance": comparison["tolerance"],
                        "reason": comparison["reason"],
                        "error": record.get("error", ""),
                    }
                )

    md_lines = [
        "# LibreYOLO ONNX Export Parity",
        "",
        f"Generated: {summary['generated_at']}",
        "",
        f"Dataset: `{(run_config or {}).get('data', '')}`",
        f"Devices: PyTorch `{(run_config or {}).get('device', '')}`, "
        f"ONNX `{(run_config or {}).get('onnx_device', '')}`, "
        f"validation `{(run_config or {}).get('val_device', '')}`",
        f"Tolerances: abs `{(run_config or {}).get('abs_tolerance', '')}`, "
        f"rel `{(run_config or {}).get('rel_tolerance', '')}`",
        "Counts: passed `{passed}`, failed `{failed}`, error `{error}`, "
        "unavailable `{unavailable}`".format(**summary["counts"]),
        "",
        "| Case | Task | ONNX claim | Status | Worst metric delta | Notes |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for record in records:
        case = record["case"]
        comparisons = record.get("comparisons") or []
        worst = ""
        if comparisons:
            numeric = [c for c in comparisons if c.get("abs_delta") is not None]
            if numeric:
                item = max(numeric, key=lambda c: float(c["abs_delta"]))
                worst = f"{item['key']} = {float(item['abs_delta']):.6g}"
        if record.get("status") == "unavailable":
            notes = (
                record.get("unavailable_reason")
                or record.get("error", "")
                or case.get("notes", "")
            )
        else:
            notes = case.get("notes") or record.get("error", "")
        md_lines.append(
            "| {case} | {task} | {claim} | {status} | {worst} | {notes} |".format(
                case=case["id"],
                task=case["task"],
                claim=case["onnx_claim"],
                status=record.get("status", ""),
                worst=worst,
                notes=str(notes).replace("|", "\\|"),
            )
        )
    (output_dir / "summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=None, help="Dataset YAML path.")
    parser.add_argument(
        "--download-mini500",
        action="store_true",
        help=f"Download {MINI500_DATASET_ID} and generate a local YAML.",
    )
    parser.add_argument("--dataset-dir", default=None, help="Local mini500 directory.")
    parser.add_argument("--hf-dataset-id", default=MINI500_DATASET_ID)
    parser.add_argument("--hf-revision", default="main")
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--output-dir", default="runs/onnx_parity")
    parser.add_argument("--case", dest="cases", nargs="*", default=None)
    parser.add_argument("--families", nargs="*", default=None)
    parser.add_argument("--tasks", nargs="*", choices=["detect", "segment", "pose"], default=None)
    parser.add_argument("--claims", nargs="*", choices=["complete", "available"], default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--device", default="auto", help="Device for PyTorch model load.")
    parser.add_argument("--onnx-device", default="cpu", help="Device for ONNX Runtime backend.")
    parser.add_argument("--val-device", default=None, help="Validation dataloader device override.")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.6)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--simplify", action="store_true")
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--reuse-exports", action="store_true")
    parser.add_argument("--save-json", action="store_true")
    parser.add_argument("--save-plots", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--allow-download-scripts", action="store_true")
    parser.add_argument(
        "--fail-on-unavailable",
        action="store_true",
        help="Return a nonzero exit code when selected checkpoint weights are unavailable.",
    )
    parser.add_argument("--abs-tolerance", type=float, default=1e-4)
    parser.add_argument("--rel-tolerance", type=float, default=1e-3)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cases = filter_cases(
        DEFAULT_CASES,
        ids=args.cases,
        families=args.families,
        tasks=args.tasks,
        claims=args.claims,
    )
    if args.limit is not None:
        cases = cases[: args.limit]

    if args.list_cases:
        for case in cases:
            print(
                f"{case.id:24s} {case.task:8s} {case.onnx_claim:12s} "
                f"{case.weights} {case.notes}"
            )
        return 0

    if not cases:
        print("No parity cases selected.", file=sys.stderr)
        return 2

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    args.data_yaml = prepare_data(args, output_dir)
    args.pose_data_yaml = (
        prepare_pose_data(args.data_yaml, split=args.split)
        if any(case.task == "pose" for case in cases)
        else args.data_yaml
    )
    if any(case.task == "pose" for case in cases):
        args.pose_keypoints_json, args.pose_images_dir = pose_direct_paths(
            args.pose_data_yaml,
            split=args.split,
        )
    else:
        args.pose_keypoints_json = None
        args.pose_images_dir = None
    print(f"Using dataset YAML: {args.data_yaml}")
    if any(case.task == "pose" for case in cases):
        print(f"Using pose dataset YAML: {args.pose_data_yaml}")
    print(f"Selected {len(cases)} parity cases")

    records = [run_case(case, args, output_dir) for case in cases]
    write_summary(records, output_dir, run_config=_runtime_report_config(args))
    failing = [
        r
        for r in records
        if r.get("status") in {"failed", "error"}
        or (args.fail_on_unavailable and r.get("status") == "unavailable")
    ]
    print(f"Wrote parity report to {output_dir}")
    return 1 if failing else 0


if __name__ == "__main__":
    raise SystemExit(main())
