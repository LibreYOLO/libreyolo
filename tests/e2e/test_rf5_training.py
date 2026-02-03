"""
RF5 End-to-End Training Test.

This test validates that LibreYOLO training pipelines work correctly by
training on 5 representative datasets from Roboflow100.

Usage:
    # Run with pytest (all models, all datasets)
    pytest tests/e2e/test_rf5_training.py -v -s

    # Run specific model family
    pytest tests/e2e/test_rf5_training.py -v -s -k "yolox"

    # Run specific dataset
    pytest tests/e2e/test_rf5_training.py -v -s -k "aquarium"

    # Run as standalone script
    python tests/e2e/test_rf5_training.py --model yolox-nano --dataset aquarium-qlnqy

Requirements:
    - GPU with CUDA support
    - ~10GB GPU memory for larger models
    - RF5 datasets (auto-downloaded if not present)
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import pytest
import torch

from .rf5_datasets import (
    RF5_DATASETS,
    get_rf5_dataset_names,
    get_dataset_path,
    download_dataset,
    DEFAULT_CACHE_DIR,
)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class RF5TestConfig:
    """Configuration for RF5 E2E test."""
    # Training settings (minimal for fast validation)
    epochs: int = 5  # Just enough to verify training works
    batch_size: int = 8
    imgsz: int = 640
    device: str = "auto"

    # Early stopping
    patience: int = 50  # Effectively disabled for short runs

    # Output
    project: str = "runs/rf5_test"
    save_results: bool = True
    results_dir: Path = Path("tests/e2e/rf5_results")

    # Evaluation
    eval_on_test: bool = True  # Evaluate on test split after training


# Model configurations (smallest model per family for speed)
MODELS_MINIMAL = {
    "yolox-nano": {"family": "yolox", "size": "nano", "imgsz": 416},
    "yolo9-t": {"family": "v9", "size": "t", "imgsz": 640},
    "rfdetr-base": {"family": "rfdetr", "size": "base", "imgsz": 560},
}

# Full model matrix for comprehensive testing
MODELS_FULL = {
    # YOLOX family
    "yolox-nano": {"family": "yolox", "size": "nano", "imgsz": 416},
    "yolox-tiny": {"family": "yolox", "size": "tiny", "imgsz": 416},
    "yolox-s": {"family": "yolox", "size": "s", "imgsz": 640},
    "yolox-m": {"family": "yolox", "size": "m", "imgsz": 640},
    "yolox-l": {"family": "yolox", "size": "l", "imgsz": 640},
    "yolox-x": {"family": "yolox", "size": "x", "imgsz": 640},
    # YOLOv9 family
    "yolo9-t": {"family": "v9", "size": "t", "imgsz": 640},
    "yolo9-s": {"family": "v9", "size": "s", "imgsz": 640},
    "yolo9-m": {"family": "v9", "size": "m", "imgsz": 640},
    "yolo9-c": {"family": "v9", "size": "c", "imgsz": 640},
    # RF-DETR family
    "rfdetr-base": {"family": "rfdetr", "size": "base", "imgsz": 560},
    "rfdetr-large": {"family": "rfdetr", "size": "large", "imgsz": 560},
}


# ============================================================================
# Model Factory
# ============================================================================

def create_model(model_name: str, num_classes: int):
    """
    Create a fresh model instance for training.

    Args:
        model_name: Model identifier (e.g., "yolox-nano", "yolo9-t")
        num_classes: Number of classes in the dataset

    Returns:
        Model instance ready for training
    """
    if model_name not in MODELS_FULL:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODELS_FULL.keys())}")

    config = MODELS_FULL[model_name]
    family = config["family"]
    size = config["size"]

    if family == "yolox":
        from libreyolo import LIBREYOLOX
        return LIBREYOLOX(size=size, nb_classes=num_classes)

    elif family == "v9":
        from libreyolo import LIBREYOLO9
        return LIBREYOLO9(size=size, nb_classes=num_classes)

    elif family == "rfdetr":
        from libreyolo import LIBREYOLORFDETR
        return LIBREYOLORFDETR(size=size, num_classes=num_classes)

    else:
        raise ValueError(f"Unknown model family: {family}")


# ============================================================================
# Training Functions
# ============================================================================

def train_on_dataset(
    model_name: str,
    dataset_name: str,
    config: RF5TestConfig,
    cache_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Train a model on a single RF5 dataset.

    Args:
        model_name: Model to train
        dataset_name: Dataset to train on
        config: Test configuration
        cache_dir: Dataset cache directory

    Returns:
        Dict with training results
    """
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR

    print(f"\n{'='*60}")
    print(f"Training {model_name} on {dataset_name}")
    print(f"{'='*60}")

    # Get dataset path (download if needed)
    dataset_path = get_dataset_path(dataset_name, cache_dir)
    if dataset_path is None:
        print(f"Dataset not found, downloading...")
        dataset_path = download_dataset(dataset_name, cache_dir)

    data_yaml = dataset_path / "data.yaml"
    if not data_yaml.exists():
        return {
            "model": model_name,
            "dataset": dataset_name,
            "success": False,
            "error": f"data.yaml not found at {data_yaml}",
        }

    # Read number of classes from data.yaml
    import yaml
    with open(data_yaml) as f:
        data_config = yaml.safe_load(f)
    num_classes = data_config.get("nc", 80)

    # Create fresh model
    try:
        model = create_model(model_name, num_classes)
    except Exception as e:
        return {
            "model": model_name,
            "dataset": dataset_name,
            "success": False,
            "error": f"Failed to create model: {e}",
        }

    # Get model-specific image size
    model_config = MODELS_FULL[model_name]
    imgsz = model_config.get("imgsz", config.imgsz)

    # Train
    start_time = time.time()
    try:
        train_results = model.train(
            data=str(data_yaml),
            epochs=config.epochs,
            batch=config.batch_size,
            imgsz=imgsz,
            device=config.device,
            project=config.project,
            name=f"{model_name}_{dataset_name}",
            exist_ok=True,
            patience=config.patience,
        )
        training_time = time.time() - start_time

        result = {
            "model": model_name,
            "dataset": dataset_name,
            "success": True,
            "training_time_seconds": training_time,
            "epochs_completed": train_results.get("epochs_completed", config.epochs),
            "best_mAP50": train_results.get("best_mAP50", 0.0),
            "best_mAP50_95": train_results.get("best_mAP50_95", 0.0),
            "final_loss": train_results.get("final_loss", None),
            "save_dir": str(train_results.get("save_dir", "")),
        }

        print(f"\nTraining completed in {training_time:.1f}s")
        print(f"  Best mAP@50: {result['best_mAP50']:.4f}")
        print(f"  Best mAP@50-95: {result['best_mAP50_95']:.4f}")

    except Exception as e:
        import traceback
        traceback.print_exc()
        result = {
            "model": model_name,
            "dataset": dataset_name,
            "success": False,
            "error": str(e),
            "training_time_seconds": time.time() - start_time,
        }
        print(f"\nTraining FAILED: {e}")

    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def run_rf5_test(
    models: Optional[List[str]] = None,
    datasets: Optional[List[str]] = None,
    config: Optional[RF5TestConfig] = None,
) -> Dict[str, Any]:
    """
    Run the full RF5 E2E test.

    Args:
        models: List of models to test (default: minimal set)
        datasets: List of datasets to test (default: all RF5)
        config: Test configuration

    Returns:
        Dict with all results and summary
    """
    if config is None:
        config = RF5TestConfig()

    if models is None:
        models = list(MODELS_MINIMAL.keys())

    if datasets is None:
        datasets = get_rf5_dataset_names()

    print("=" * 70)
    print("RF5 End-to-End Training Test")
    print("=" * 70)
    print(f"Models: {models}")
    print(f"Datasets: {datasets}")
    print(f"Epochs per dataset: {config.epochs}")
    print(f"Device: {config.device}")
    print("=" * 70)

    all_results = []
    start_time = time.time()

    for model_name in models:
        for dataset_name in datasets:
            result = train_on_dataset(
                model_name=model_name,
                dataset_name=dataset_name,
                config=config,
            )
            all_results.append(result)

    total_time = time.time() - start_time

    # Summary
    successful = [r for r in all_results if r.get("success", False)]
    failed = [r for r in all_results if not r.get("success", False)]

    summary = {
        "total_runs": len(all_results),
        "successful": len(successful),
        "failed": len(failed),
        "total_time_seconds": total_time,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "device": config.device,
        },
        "results": all_results,
    }

    # Print summary
    print("\n" + "=" * 70)
    print("RF5 TEST SUMMARY")
    print("=" * 70)
    print(f"Total runs: {len(all_results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print(f"Total time: {total_time/60:.1f} minutes")

    if failed:
        print("\nFailed runs:")
        for r in failed:
            print(f"  - {r['model']} on {r['dataset']}: {r.get('error', 'Unknown error')}")

    # Save results if configured
    if config.save_results:
        config.results_dir.mkdir(parents=True, exist_ok=True)
        results_file = config.results_dir / f"rf5_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nResults saved to: {results_file}")

    print("=" * 70)

    return summary


# ============================================================================
# Pytest Tests
# ============================================================================

@pytest.fixture(scope="module")
def rf5_config():
    """Test configuration for RF5."""
    return RF5TestConfig(
        epochs=3,  # Minimal epochs for CI
        batch_size=8,
        device="auto",
        save_results=False,
    )


@pytest.mark.e2e
@pytest.mark.parametrize("dataset_name", get_rf5_dataset_names())
def test_yolox_nano_training(dataset_name: str, rf5_config: RF5TestConfig):
    """Test YOLOX-nano training on RF5 datasets."""
    result = train_on_dataset(
        model_name="yolox-nano",
        dataset_name=dataset_name,
        config=rf5_config,
    )
    assert result["success"], f"Training failed: {result.get('error', 'Unknown error')}"
    assert result.get("best_mAP50", 0) >= 0, "Invalid mAP value"


@pytest.mark.e2e
@pytest.mark.parametrize("dataset_name", get_rf5_dataset_names())
def test_yolo9_t_training(dataset_name: str, rf5_config: RF5TestConfig):
    """Test YOLOv9-t training on RF5 datasets."""
    result = train_on_dataset(
        model_name="yolo9-t",
        dataset_name=dataset_name,
        config=rf5_config,
    )
    assert result["success"], f"Training failed: {result.get('error', 'Unknown error')}"


@pytest.mark.e2e
@pytest.mark.parametrize("dataset_name", get_rf5_dataset_names())
def test_rfdetr_base_training(dataset_name: str, rf5_config: RF5TestConfig):
    """Test RF-DETR-base training on RF5 datasets."""
    result = train_on_dataset(
        model_name="rfdetr-base",
        dataset_name=dataset_name,
        config=rf5_config,
    )
    assert result["success"], f"Training failed: {result.get('error', 'Unknown error')}"


@pytest.mark.e2e
def test_rf5_quick(rf5_config: RF5TestConfig):
    """Quick RF5 test: one model, one dataset."""
    result = train_on_dataset(
        model_name="yolox-nano",
        dataset_name="aquarium-qlnqy",
        config=rf5_config,
    )
    assert result["success"], f"Training failed: {result.get('error', 'Unknown error')}"


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    """CLI entry point for RF5 test."""
    parser = argparse.ArgumentParser(
        description="RF5 End-to-End Training Test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run minimal test (1 model per family, all RF5 datasets)
  python -m tests.e2e.test_rf5_training

  # Run specific model on all datasets
  python -m tests.e2e.test_rf5_training --model yolox-nano

  # Run specific model on specific dataset
  python -m tests.e2e.test_rf5_training --model yolox-nano --dataset aquarium-qlnqy

  # Run all models (full matrix)
  python -m tests.e2e.test_rf5_training --all-models

  # Custom epochs
  python -m tests.e2e.test_rf5_training --epochs 10
        """,
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=list(MODELS_FULL.keys()),
        help="Specific model to test",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=get_rf5_dataset_names(),
        help="Specific dataset to test",
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Test all models (not just minimal set)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of epochs per dataset (default: 5)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size (default: 8)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device (auto, cuda, cpu, 0, 1, etc.)",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit",
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="List RF5 datasets and exit",
    )

    args = parser.parse_args()

    if args.list_models:
        print("Available models:")
        print("\nMinimal set (fastest):")
        for name in MODELS_MINIMAL:
            config = MODELS_FULL[name]
            print(f"  {name}: {config['family']}-{config['size']}")
        print("\nFull set:")
        for name, config in MODELS_FULL.items():
            print(f"  {name}: {config['family']}-{config['size']}")
        return

    if args.list_datasets:
        from .rf5_datasets import print_rf5_info
        print_rf5_info()
        return

    # Build configuration
    config = RF5TestConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
    )

    # Determine models and datasets
    if args.model:
        models = [args.model]
    elif args.all_models:
        models = list(MODELS_FULL.keys())
    else:
        models = list(MODELS_MINIMAL.keys())

    if args.dataset:
        datasets = [args.dataset]
    else:
        datasets = get_rf5_dataset_names()

    # Run test
    summary = run_rf5_test(
        models=models,
        datasets=datasets,
        config=config,
    )

    # Exit with error if any failed
    if summary["failed"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
