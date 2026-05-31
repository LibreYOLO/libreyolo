#!/usr/bin/env python3
"""Download a sample image and run all available models on it, saving annotated results."""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

SAMPLE_IMAGES = {
    "parkour": Path(__file__).resolve().parents[1] / "libreyolo" / "assets" / "parkour.jpg",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run LibreYOLO models on sample images",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--weights-dir", default="weights", help="Folder containing .pt model weights")
    p.add_argument("--images-dir", default="sample_images", help="Folder for input images")
    p.add_argument("--output-dir", default="sample_images_output", help="Folder for annotated outputs")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--augment", action="store_true", help="Enable TTA (test-time augmentation)")
    p.add_argument("--device", default="auto")
    return p.parse_args()


def discover_models(weights_dir: Path) -> tuple[list[str], list[str]]:
    all_weights = sorted(weights_dir.glob("Libre*.pt"))
    detect = [str(p) for p in all_weights if "-seg" not in p.stem]
    seg = [str(p) for p in all_weights if "-seg" in p.stem]
    return detect, seg


def prepare_samples(images_dir: Path) -> list[Path]:
    images_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, source in SAMPLE_IMAGES.items():
        dest = images_dir / f"{name}.jpg"
        if not dest.exists():
            print(f"  Copying {name}.jpg ...")
            shutil.copy2(source, dest)
        paths.append(dest)
    return paths


def run_model(model_path: str, images: list[Path], output_dir: Path, args: argparse.Namespace) -> None:
    from libreyolo import LibreYOLO

    name = Path(model_path).stem
    model_out = output_dir / name
    model_out.mkdir(parents=True, exist_ok=True)

    try:
        model = LibreYOLO(model_path=model_path, device=args.device)
    except Exception as exc:
        print(f"  ✗ Failed to load {model_path}: {exc}", file=sys.stderr)
        return

    aug_label = " [TTA]" if args.augment else ""
    model_start = time.perf_counter()
    print(f"  {name}{aug_label}  →  {model_out}/")

    for img_path in images:
        out_file = model_out / img_path.name
        try:
            t0 = time.perf_counter()
            model(
                str(img_path),
                conf=args.conf,
                iou=args.iou,
                augment=args.augment,
                save=True,
                output_path=str(out_file),
            )
            ms = (time.perf_counter() - t0) * 1000
            print(f"    {img_path.name}  {ms:.0f} ms")
        except Exception as exc:
            print(f"    ✗ {img_path.name}: {exc}", file=sys.stderr)

    total_ms = (time.perf_counter() - model_start) * 1000
    print(f"    total  {total_ms:.0f} ms")


def main() -> None:
    args = parse_args()
    weights_dir = Path(args.weights_dir)
    images_dir = Path(args.images_dir)
    output_dir = Path(args.output_dir)

    detect_models, seg_models = discover_models(weights_dir)
    all_models = detect_models + seg_models
    if not all_models:
        print(f"No Libre*.pt weights found in {weights_dir}/", file=sys.stderr)
        sys.exit(1)

    print("── Preparing sample images ────────────────────────")
    images = prepare_samples(images_dir)
    print(f"  {len(images)} image(s) in {images_dir}/\n")

    print(f"── Running {len(detect_models)} detection + {len(seg_models)} segmentation model(s) ─")
    for model_path in all_models:
        run_model(model_path, images, output_dir, args)

    print(f"\nResults saved to: {output_dir}/")


if __name__ == "__main__":
    main()
