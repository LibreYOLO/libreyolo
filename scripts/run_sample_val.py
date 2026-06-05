#!/usr/bin/env python3
"""Evaluate a LibreYOLO model on a dataset.

Runs a full COCO-metric validation pass and prints mAP50 / mAP50-95.
Validation plots (metrics charts, per-class AP/recall, confusion matrix,
PR curves, sample images) are saved to runs/val/<model>_<timestamp>/plots/.

Supported models
----------------
  rfdetr   RF-DETR object detector / instance segmenter
  yolo9    YOLOv9 object detector / instance segmenter

Pretrained weights
------------------
  RF-DETR   fetched automatically from HuggingFace (all sizes, det + seg)
  YOLOv9    auto-downloaded for detection; seg requires --weights

Quick start
-----------
YOLOv9-m detection, pretrained COCO weights:
    python scripts/run_sample_val.py

RF-DETR-n detection, pretrained weights:
    python scripts/run_sample_val.py --model rfdetr

RF-DETR-s segmentation, pretrained weights:
    python scripts/run_sample_val.py --model rfdetr --task segment --size s

YOLOv9-s detection, custom dataset:
    python scripts/run_sample_val.py --size s --data /path/to/data.yaml

YOLOv9 segmentation, custom trained weights:
    python scripts/run_sample_val.py --task segment --weights runs/train/exp/best.pt

Save COCO-format JSON predictions alongside the plots:
    python scripts/run_sample_val.py --save-json

Evaluate on a test split with a larger batch on a specific GPU:
    python scripts/run_sample_val.py --split test --batch 32 --device 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--model",
        default="yolo9",
        choices=["rfdetr", "yolo9"],
        help="Model family (default: yolo9)",
    )
    p.add_argument(
        "--task",
        default="detect",
        choices=["detect", "segment"],
        help="Task type (default: detect)",
    )
    p.add_argument(
        "--size",
        default=None,
        help=(
            "Model size variant. "
            "rfdetr-detect: n/s/m/l — rfdetr-segment: n/s/m/l/x/xx — "
            "yolo9: t/s/m/c. "
            "Defaults: rfdetr-detect=n, rfdetr-segment=s, yolo9=m. "
            "Custom checkpoints auto-detect size when possible."
        ),
    )
    p.add_argument(
        "--weights",
        default=None,
        help=(
            "Path to a .pt checkpoint. "
            "When omitted, RF-DETR fetches pretrained weights from HuggingFace "
            "and YOLOv9 auto-downloads LibreYOLO9<size>.pt. "
            "Required for YOLOv9 segmentation."
        ),
    )
    p.add_argument(
        "--data",
        default="coco.yaml",
        help="Path to a YOLO-format dataset YAML (default: coco.yaml)",
    )
    p.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size (default: 16)",
    )
    p.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="Input image size in pixels. Defaults to the model's native resolution.",
    )
    p.add_argument(
        "--conf",
        type=float,
        default=0.001,
        help="Confidence threshold for detections (default: 0.001)",
    )
    p.add_argument(
        "--iou",
        type=float,
        default=0.6,
        help="IoU threshold for NMS (default: 0.6)",
    )
    p.add_argument(
        "--device",
        default="auto",
        help="Device: auto, cpu, mps, 0, cuda:0 (default: auto)",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of DataLoader worker processes (default: 4)",
    )
    p.add_argument(
        "--split",
        default="val",
        choices=["val", "test"],
        help="Dataset split to evaluate (default: val)",
    )
    p.add_argument(
        "--save-dir",
        default=None,
        dest="save_dir",
        help="Output directory for results. Default: runs/val/<model>_<timestamp>/",
    )
    p.add_argument(
        "--save-json",
        action="store_true",
        default=False,
        dest="save_json",
        help="Save predictions in COCO JSON format alongside the plots",
    )
    p.add_argument(
        "--allow-download-scripts",
        action="store_true",
        default=False,
        dest="allow_download_scripts",
        help=(
            "Allow the dataset YAML's embedded Python download block to execute. "
            "Only use this with dataset files you trust."
        ),
    )
    return p.parse_args()


def _normalize_device(device: str) -> str:
    text = str(device)
    return f"cuda:{text}" if text.isdigit() else text


def _val_kwargs(args: argparse.Namespace) -> dict:
    kw = dict(
        data=args.data,
        batch=args.batch,
        device=_normalize_device(args.device),
        workers=args.workers,
        split=args.split,
        conf=args.conf,
        iou=args.iou,
        save_json=args.save_json,
        save_plots=True,
        allow_download_scripts=args.allow_download_scripts,
    )
    if args.imgsz:
        kw["imgsz"] = args.imgsz
    if args.save_dir:
        kw["save_dir"] = args.save_dir
    return kw


def _print_results(label: str, results: dict, task: str) -> None:
    print(f"\n{label} — validation results")
    print(f"  {'mAP50':<18}: {results.get('metrics/mAP50',    0.0):.4f}")
    print(f"  {'mAP50-95':<18}: {results.get('metrics/mAP50-95', 0.0):.4f}")
    if task == "segment":
        print(f"  {'Box  mAP50-95':<18}: {results.get('metrics/mAP50-95(B)', 0.0):.4f}")
        print(f"  {'Mask mAP50-95':<18}: {results.get('metrics/mAP50-95(M)', 0.0):.4f}")
        print(f"  {'Precision (B)':<18}: {results.get('metrics/precision(B)', 0.0):.4f}")
        print(f"  {'Recall    (B)':<18}: {results.get('metrics/recall(B)',    0.0):.4f}")
        print(f"  {'Precision (M)':<18}: {results.get('metrics/precision(M)', 0.0):.4f}")
        print(f"  {'Recall    (M)':<18}: {results.get('metrics/recall(M)',    0.0):.4f}")
    else:
        print(f"  {'Precision':<18}: {results.get('metrics/precision', 0.0):.4f}")
        print(f"  {'Recall':<18}: {results.get('metrics/recall',    0.0):.4f}")


def val_rfdetr(args: argparse.Namespace) -> None:
    from libreyolo.models.rfdetr.model import LibreRFDETR

    is_seg = args.task == "segment"
    device = _normalize_device(args.device)
    size = (
        args.size
        if args.size is not None
        else (None if args.weights else ("s" if is_seg else "n"))
    )
    model = LibreRFDETR(args.weights, size=size, task=args.task, device=device)
    results = model.val(**_val_kwargs(args))
    label_size = size or model.size
    _print_results(f"RF-DETR-{label_size} ({args.task})", results, args.task)


def val_yolo9(args: argparse.Namespace) -> None:
    from libreyolo.models.yolo9.model import LibreYOLO9

    is_seg = args.task == "segment"
    device = _normalize_device(args.device)

    if args.weights and args.size is None:
        from libreyolo.models import LibreYOLO

        model = LibreYOLO(args.weights, task=args.task, device=device)
        size = getattr(model, "size", "auto")
    else:
        size = args.size or "m"

        if args.weights:
            weights = args.weights
        elif is_seg:
            sys.exit(
                "Error: --weights is required for YOLOv9 segmentation "
                "(no pretrained seg weights are available for auto-download).\n"
                "Example: --weights runs/train/exp/best.pt"
            )
        else:
            weights = f"LibreYOLO9{size}.pt"

        model = LibreYOLO9(weights, size=size, task=args.task, device=device)

    results = model.val(**_val_kwargs(args))
    _print_results(f"YOLOv9-{size} ({args.task})", results, args.task)


def main() -> None:
    args = parse_args()
    dispatch = {"rfdetr": val_rfdetr, "yolo9": val_yolo9}
    dispatch[args.model](args)


if __name__ == "__main__":
    main()
