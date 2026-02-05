"""
RF5 End-to-End Training Test.

Config-driven training validation on 5 representative Roboflow100 datasets.

Usage:
    python -m tests.e2e.test_rf5_training --config yolox.yaml --size nano
    python -m tests.e2e.test_rf5_training --config yolox.yaml --size nano --dataset aquarium-qlnqy
    python -m tests.e2e.test_rf5_training --list-configs
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml

from .rf5_datasets import (
    DEFAULT_CACHE_DIR,
    download_dataset,
    get_dataset_path,
    get_rf5_dataset_names,
)

CONFIGS_DIR = Path(__file__).parent / "configs"
RESULTS_DIR = Path(__file__).parent / "rf5_results"


def load_config(config_path: Path, size: str) -> Dict[str, Any]:
    """Load YAML config and merge size-specific settings."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Merge size-specific settings
    sizes = config.pop("sizes", {})
    if size not in sizes:
        available = list(sizes.keys())
        raise ValueError(f"Size '{size}' not in config. Available: {available}")

    size_config = sizes[size]
    config["model_size"] = size
    config["model_weights"] = size_config.get("weights")
    config["imgsz"] = size_config.get("imgsz", 640)
    config["lr0"] = size_config.get("lr0", config.get("lr0", 0.01))

    return config


def create_model(config: Dict[str, Any]):
    """Create model from config with auto-download support."""
    from libreyolo import LIBREYOLO

    weights = config.get("model_weights")
    size = config["model_size"]

    # LIBREYOLO factory handles auto-download from HuggingFace
    # Don't pass nb_classes - load COCO weights, train() will rebuild head
    return LIBREYOLO(weights, size=size)


def train_on_dataset(
    config: Dict[str, Any],
    dataset_name: str,
    cache_dir: Optional[Path] = None,
    device: str = "auto",
) -> Dict[str, Any]:
    """Train model on a single dataset."""
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR

    model_name = f"{config['model_type']}_{config['model_size']}"
    print(f"\n{'='*60}")
    print(f"Training {model_name} on {dataset_name}")
    print(f"{'='*60}")

    # Get dataset
    dataset_path = get_dataset_path(dataset_name, cache_dir)
    if dataset_path is None:
        print("Downloading dataset...")
        dataset_path = download_dataset(dataset_name, cache_dir)

    data_yaml = dataset_path / "data.yaml"
    if not data_yaml.exists():
        return {"model": model_name, "dataset": dataset_name, "success": False,
                "error": f"data.yaml not found"}

    try:
        model = create_model(config)
    except Exception as e:
        return {"model": model_name, "dataset": dataset_name, "success": False,
                "error": f"Failed to create model: {e}"}

    # Build train kwargs
    train_kwargs = {
        "data": str(data_yaml),
        "epochs": config.get("epochs", 100),
        "batch": config.get("batch_size", 16),
        "imgsz": config.get("imgsz", 640),
        "device": device,
        "project": "runs/rf5_test",
        "name": f"{model_name}_{dataset_name}",
        "exist_ok": True,
    }

    optional = [
        "lr0", "momentum", "weight_decay", "nesterov", "optimizer",
        "scheduler", "warmup_epochs", "warmup_lr_start", "no_aug_epochs", "min_lr_ratio",
        "mosaic_prob", "mixup_prob", "hsv_prob", "flip_prob",
        "degrees", "translate", "shear", "mosaic_scale", "mixup_scale",
        "ema", "ema_decay", "amp", "patience", "eval_interval", "save_period", "workers",
    ]
    for p in optional:
        if p in config:
            train_kwargs[p] = config[p]

    start_time = time.time()
    try:
        train_results = model.train(**train_kwargs)
        training_time = time.time() - start_time

        # Eval on test split
        test_mAP50, test_mAP50_95 = 0.0, 0.0
        print("Evaluating on test split...")
        try:
            test_results = model.val(data=str(data_yaml), split="test")
            test_mAP50 = test_results.get("metrics/mAP50", 0.0)
            test_mAP50_95 = test_results.get("metrics/mAP50-95", 0.0)
        except Exception as e:
            print(f"Test eval failed: {e}")

        result = {
            "model": model_name,
            "dataset": dataset_name,
            "success": True,
            "training_time_seconds": training_time,
            "train_best_mAP50": train_results.get("best_mAP50", 0.0),
            "test_mAP50": test_mAP50,
            "test_mAP50_95": test_mAP50_95,
        }

        print(f"\nDone in {training_time:.1f}s")
        print(f"  Train mAP@50: {result['train_best_mAP50']:.4f}")
        print(f"  Test mAP@50: {result['test_mAP50']:.4f}")

    except Exception as e:
        import traceback
        traceback.print_exc()
        result = {"model": model_name, "dataset": dataset_name, "success": False,
                  "error": str(e)}

    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return result


def run_rf5(config_path: Path, size: str, datasets: Optional[List[str]] = None,
            device: str = "auto") -> Dict[str, Any]:
    """Run RF5 benchmark."""
    config = load_config(config_path, size)
    datasets = datasets or get_rf5_dataset_names()
    model_name = f"{config['model_type']}_{size}"

    print("=" * 70)
    print(f"RF5 | {model_name} | LR={config.get('lr0')} | {len(datasets)} datasets")
    print("=" * 70)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"rf5_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    results = []
    for ds in datasets:
        results.append(train_on_dataset(config, ds, device=device))

        successful = [r for r in results if r.get("success")]
        test_maps = [r["test_mAP50"] for r in successful if "test_mAP50" in r]
        rf5_score = sum(test_maps) / len(test_maps) if test_maps else 0.0

        summary = {
            "rf5_score": rf5_score,
            "model": model_name,
            "successful": len(successful),
            "failed": len(results) - len(successful),
            "results": results,
            "timestamp": datetime.now().isoformat(),
        }

        with open(out, "w") as f:
            json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print(f"RF5 SCORE: {rf5_score:.4f}")
    print(f"Completed: {len(successful)}/{len(results)}")
    print("=" * 70)
    print(f"Saved: {out}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="RF5 Benchmark")
    parser.add_argument("--config", type=str, help="Config YAML (yolox.yaml, yolo9.yaml, rfdetr.yaml)")
    parser.add_argument("--size", type=str, help="Model size (nano, s, m, etc.)")
    parser.add_argument("--dataset", type=str, help="Single dataset to test")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--list-configs", action="store_true")

    args = parser.parse_args()

    if args.list_configs:
        for f in sorted(CONFIGS_DIR.glob("*.yaml")):
            cfg = yaml.safe_load(open(f))
            sizes = list(cfg.get("sizes", {}).keys())
            print(f"{f.name}: {sizes}")
        return

    if not args.config or not args.size:
        parser.error("--config and --size required")

    config_path = CONFIGS_DIR / args.config if not Path(args.config).exists() else Path(args.config)
    if not config_path.exists():
        print(f"Config not found: {args.config}")
        sys.exit(1)

    datasets = [args.dataset] if args.dataset else None
    run_rf5(config_path, args.size, datasets, args.device)


if __name__ == "__main__":
    main()
