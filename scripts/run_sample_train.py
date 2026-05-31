#!/usr/bin/env python3
"""Sample training script — RF-DETR and YOLOv9 on 1 or more GPUs.

This script is intended as a reference for how to call the LibreYOLO
training API. Read the inline comments to understand each option.

Usage
-----
RF-DETR, 1 GPU, AutoBatch (default):
    python scripts/run_sample_train.py

RF-DETR, 2 GPUs:
    python scripts/run_sample_train.py --devices 0,1

YOLOv9, 1 GPU:
    python scripts/run_sample_train.py --model yolo9

YOLOv9, 2 GPUs:
    python scripts/run_sample_train.py --model yolo9 --devices 0,1

Manual batch size (skip AutoBatch):
    python scripts/run_sample_train.py --batch 16

Custom dataset and longer run:
    python scripts/run_sample_train.py --model rfdetr --data /path/to/data.yaml --epochs 100

Resume from a checkpoint:
    python scripts/run_sample_train.py --resume runs/train/rfdetr_1gpu/best.pt

Note: --allow-download-scripts is required when the data YAML contains an
embedded Python download block.  coco128.yaml uses a plain URL and does not
need it.  Only pass the flag for datasets you trust.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running directly from the repo root without installing the package.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--model",
        default="rfdetr",
        help="Model to train: rfdetr or yolo9  (default: rfdetr)",
    )
    p.add_argument(
        "--size",
        default=None,
        help="Model size variant — rfdetr: n/s/m/l, yolo9: s/m/c  (default: n for rfdetr, s for yolo9)",
    )
    p.add_argument(
        "--data",
        default="coco128.yaml",
        help="YOLO-format data.yaml path  (default: coco128.yaml, auto-downloaded)",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs  (default: 10)",
    )
    p.add_argument(
        "--batch",
        type=int,
        default=-1,
        help=(
            "Global batch size. Use -1 (default) to let AutoBatch pick the "
            "largest batch that fits in VRAM. Set a positive integer to fix it manually."
        ),
    )
    p.add_argument(
        "--nbs",
        type=int,
        default=None,
        help=(
            "Nominal batch size for gradient accumulation. "
            "When --batch < nbs the trainer accumulates gradients over nbs/batch steps. "
            "Defaults to each model's built-in value (rfdetr: 16, yolo9: 64)."
        ),
    )
    p.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="Input image size (square). Defaults to the model's built-in default.",
    )
    p.add_argument(
        "--devices",
        default="0",
        help="GPU index or comma-separated indices for multi-GPU DDP  (default: 0)",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=4,
        help="DataLoader worker processes per rank  (default: 4)",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Output directory  (default: runs/train/<model>_<ngpu>gpu)",
    )
    p.add_argument(
        "--resume",
        default=None,
        help="Path to a checkpoint to resume training from",
    )
    p.add_argument(
        "--allow-download-scripts",
        action="store_true",
        default=False,
        dest="allow_download_scripts",
        help=(
            "Allow the data YAML's embedded Python download block to run. "
            "Not needed for coco128.yaml (URL-only). Only use for datasets you trust."
        ),
    )
    return p.parse_args()


def train_rfdetr(args: argparse.Namespace) -> None:
    from libreyolo.models.rfdetr.model import LibreRFDETR

    size = args.size or "n"
    ngpu = len(args.devices.split(","))
    output = args.output or f"runs/train/rfdetr_{ngpu}gpu"

    # Pass None as model_path to start from pretrained weights.
    # Pass a .pt path (or --resume) to resume a run.
    model = LibreRFDETR(args.resume, size=size)

    train_kwargs = dict(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,       # -1 → AutoBatch; positive int → fixed batch
        device=args.devices,
        workers=args.workers,
        output_dir=output,
        amp=True,               # mixed-precision — keeps VRAM usage lower
        seed=42,
        allow_download_scripts=args.allow_download_scripts,
        exist_ok=True,
        resume=args.resume,
        # imgsz is intentionally omitted: RF-DETR derives it from the size
        # variant (n→384, s→512, m→576, l→704) and ignores any override.
    )
    if args.nbs is not None:
        train_kwargs["nbs"] = args.nbs  # otherwise RFDETRConfig default (nbs=16) applies

    result = model.train(**train_kwargs)

    best_map = result.get("best_mAP50_95", 0.0)
    save_dir = result.get("output_dir", output)
    print(f"\nTraining complete.")
    print(f"  Best mAP50-95 : {best_map:.4f}")
    print(f"  Weights saved : {save_dir}")


def train_yolo9(args: argparse.Namespace) -> None:
    from libreyolo.models.yolo9.model import LibreYOLO9

    size = args.size or "s"
    ngpu = len(args.devices.split(","))
    output = args.output or f"runs/train/yolo9_{ngpu}gpu"

    # Pass the checkpoint path when resuming so that model_path is set before
    # train(resume=True) is called; pass None for a fresh pretrained model.
    model = LibreYOLO9(args.resume, size=size)

    train_kwargs = dict(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,       # -1 → AutoBatch; positive int → fixed batch
        device=args.devices,
        workers=args.workers,
        project=str(Path(output).parent),
        name=Path(output).name,
        amp=True,
        seed=42,
        allow_download_scripts=args.allow_download_scripts,
        exist_ok=True,
        resume=bool(args.resume),
    )
    if args.nbs is not None:
        train_kwargs["nbs"] = args.nbs  # otherwise TrainConfig default (nbs=None, no cap) applies
    if args.imgsz is not None:
        train_kwargs["imgsz"] = args.imgsz

    result = model.train(**train_kwargs)

    best_map = result.get("best_mAP50_95", 0.0)
    print(f"\nTraining complete.")
    print(f"  Best mAP50-95 : {best_map:.4f}")
    print(f"  Weights saved : {output}")


def main() -> None:
    args = parse_args()

    model = args.model.lower().replace("-", "")
    if model == "rfdetr":
        train_rfdetr(args)
    elif model in ("yolo9", "yolov9"):
        train_yolo9(args)
    else:
        print(f"Unknown model '{args.model}'. Choose rfdetr or yolo9.")
        sys.exit(1)


if __name__ == "__main__":
    main()
