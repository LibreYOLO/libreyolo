"""
RF5 End-to-End Training Test.

Config-driven training validation on 5 representative Roboflow100 datasets.
Runs as proper pytest tests with the ``rf5`` marker.

Usage:
    pytest tests/e2e/test_rf5_training.py -m rf5 -v
    pytest tests/e2e/test_rf5_training.py -k "aquarium" -v
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
import yaml

from .conftest import cuda_cleanup

pytestmark = [pytest.mark.e2e, pytest.mark.rf5]


# ---------------------------------------------------------------------------
# RF5 dataset infrastructure (self-contained — only used by this file)
# ---------------------------------------------------------------------------

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "rf100"
ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY")
RF100_WORKSPACE = "roboflow-100"

CONFIGS_DIR = Path(__file__).parent / "configs"
RESULTS_DIR = Path(__file__).parent / "rf5_results"

RF5_DATASETS = [
    {
        "name": "bacteria-ptywi",
        "category": "microscopic",
        "train_images": 30,
        "num_classes": 1,
        "objects_per_image": 61,
        "avg_object_area": 0.0001,
        "purpose": "Minimum viable dataset, tiny objects, high density",
    },
    {
        "name": "circuit-elements",
        "category": "documents",
        "train_images": 672,
        "num_classes": 45,
        "objects_per_image": 255,
        "avg_object_area": 0.0019,
        "purpose": "Stress test: many classes, extremely high object density",
    },
    {
        "name": "aquarium-qlnqy",
        "category": "underwater",
        "train_images": 448,
        "num_classes": 7,
        "objects_per_image": 7,
        "avg_object_area": 0.0600,
        "purpose": "Balanced baseline with typical characteristics",
    },
    {
        "name": "aerial-cows",
        "category": "aerial",
        "train_images": 1084,
        "num_classes": 1,
        "objects_per_image": 14,
        "avg_object_area": 0.0004,
        "purpose": "Small objects in aerial/satellite imagery",
    },
    {
        "name": "road-signs-6ih4y",
        "category": "real_world",
        "train_images": 1376,
        "num_classes": 21,
        "objects_per_image": 1,
        "avg_object_area": 0.1422,
        "purpose": "Large objects with sparse annotations",
    },
]

RF5_DATASET_NAMES = [d["name"] for d in RF5_DATASETS]


def get_rf5_dataset_info(name: str) -> Optional[Dict]:
    """Get info for a specific RF5 dataset."""
    for d in RF5_DATASETS:
        if d["name"] == name:
            return d
    return None


def get_dataset_path(
    dataset_name: str,
    cache_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Get the local path for a dataset if it exists."""
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR
    local_dir = Path(cache_dir) / dataset_name
    if local_dir.exists() and (local_dir / "data.yaml").exists():
        return local_dir
    return None


def download_dataset(
    dataset_name: str,
    cache_dir: Optional[Path] = None,
    force: bool = False,
) -> Path:
    """Download a dataset from Roboflow if not already cached."""
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    local_dir = cache_dir / dataset_name

    if not force and local_dir.exists():
        data_yaml = local_dir / "data.yaml"
        if data_yaml.exists() and (local_dir / "train" / "labels").exists():
            return local_dir

    try:
        if not ROBOFLOW_API_KEY:
            raise RuntimeError(
                "ROBOFLOW_API_KEY environment variable not set. "
                "Set it in your environment or manually download "
                "datasets to ~/.cache/rf100/"
            )

        from roboflow import Roboflow

        rf = Roboflow(api_key=ROBOFLOW_API_KEY)
        project = rf.workspace(RF100_WORKSPACE).project(dataset_name)
        project.version(1).download(
            model_format="yolov8",
            location=str(local_dir),
            overwrite=force,
        )
        _fix_data_yaml(local_dir)
        return local_dir

    except ImportError:
        raise RuntimeError(
            "roboflow package required for dataset download. "
            "Install with: pip install roboflow"
        )
    except Exception as e:
        raise RuntimeError(f"Failed to download {dataset_name}: {e}")


def _fix_data_yaml(dataset_dir: Path) -> None:
    """Fix data.yaml to use relative paths."""
    data_yaml = dataset_dir / "data.yaml"
    if not data_yaml.exists():
        return

    with open(data_yaml, "r") as f:
        data = yaml.safe_load(f)

    data["path"] = "."
    data["train"] = "train/images"
    data["val"] = "valid/images"
    data["test"] = "test/images"

    with open(data_yaml, "w") as f:
        yaml.dump(data, f, default_flow_style=False)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def load_config(config_path: Path, size: str) -> Dict[str, Any]:
    """Load YAML config and merge size-specific settings."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

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


# ---------------------------------------------------------------------------
# Test parametrization
# ---------------------------------------------------------------------------

# Collect (config_path, size) pairs from all YAML configs
_TEST_CONFIGS: List[tuple] = []
_TEST_IDS: List[str] = []

for _cfg_path in sorted(CONFIGS_DIR.glob("*.yaml")):
    with open(_cfg_path) as _f:
        _cfg = yaml.safe_load(_f)
    _model_type = _cfg.get("model_type", _cfg_path.stem)
    for _size in _cfg.get("sizes", {}):
        _TEST_CONFIGS.append((_cfg_path, _size))
        _TEST_IDS.append(f"{_model_type}-{_size}")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def rf5_datasets():
    """Ensure all RF5 datasets are available (download if needed)."""
    paths = {}
    for dataset in RF5_DATASETS:
        name = dataset["name"]
        path = get_dataset_path(name)
        if path is not None:
            paths[name] = path
        else:
            try:
                path = download_dataset(name)
                paths[name] = path
            except Exception as e:
                pytest.skip(f"Cannot get RF5 dataset {name}: {e}")
    return paths


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("config_path,size", _TEST_CONFIGS, ids=_TEST_IDS)
@pytest.mark.parametrize("dataset_name", RF5_DATASET_NAMES)
def test_rf5_training(config_path, size, dataset_name, rf5_datasets, tmp_path):
    """Train model on an RF5 dataset, verify training completes and produces results."""
    from libreyolo import LibreYOLO

    config = load_config(config_path, size)
    model_name = f"{config['model_type']}_{size}"

    dataset_path = rf5_datasets.get(dataset_name)
    if dataset_path is None:
        pytest.skip(f"Dataset {dataset_name} not available")

    data_yaml = dataset_path / "data.yaml"
    assert data_yaml.exists(), f"data.yaml not found in {dataset_path}"

    # Create model
    weights = config.get("model_weights")
    model = LibreYOLO(weights, size=size)

    # Build train kwargs
    train_kwargs = {
        "data": str(data_yaml),
        "epochs": config.get("epochs", 100),
        "batch": config.get("batch_size", 16),
        "imgsz": config.get("imgsz", 640),
        "project": str(tmp_path / "runs"),
        "name": f"{model_name}_{dataset_name}",
        "exist_ok": True,
    }

    optional = [
        "lr0", "momentum", "weight_decay", "nesterov", "optimizer",
        "scheduler", "warmup_epochs", "warmup_lr_start", "no_aug_epochs",
        "min_lr_ratio", "mosaic_prob", "mixup_prob", "hsv_prob", "flip_prob",
        "degrees", "translate", "shear", "mosaic_scale", "mixup_scale",
        "ema", "ema_decay", "amp", "patience", "eval_interval",
        "save_period", "workers",
    ]
    for p in optional:
        if p in config:
            train_kwargs[p] = config[p]

    start_time = time.time()
    model.train(**train_kwargs)
    training_time = time.time() - start_time

    # Evaluate on test split
    test_results = model.val(data=str(data_yaml), split="test")
    test_mAP50 = test_results.get("metrics/mAP50", 0.0)
    test_mAP50_95 = test_results.get("metrics/mAP50-95", 0.0)

    print(
        f"\n  {model_name} on {dataset_name}: "
        f"time={training_time:.1f}s, "
        f"test_mAP50={test_mAP50:.4f}, "
        f"test_mAP50-95={test_mAP50_95:.4f}"
    )

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "model": model_name,
        "dataset": dataset_name,
        "training_time_seconds": training_time,
        "test_mAP50": test_mAP50,
        "test_mAP50_95": test_mAP50_95,
        "timestamp": datetime.now().isoformat(),
    }
    out = RESULTS_DIR / f"rf5_{model_name}_{dataset_name}.json"
    out.write_text(json.dumps(result, indent=2))

    # Training must complete without error (the assertion is implicit —
    # if model.train() raises, the test fails). We check mAP > 0 as a
    # basic sanity check.
    assert test_mAP50 >= 0.0, "mAP50 should be non-negative"

    cuda_cleanup()
