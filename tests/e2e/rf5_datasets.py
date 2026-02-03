"""
RF5 Dataset Definitions.

RF5 is a minimal validation subset of Roboflow100, designed to quickly verify
that training code works correctly. It contains 5 carefully selected datasets
that cover key edge cases and common scenarios.

Selection criteria:
- Dataset size: from 30 to 1,376 training images
- Number of classes: from 1 to 45 classes
- Object size: from tiny (0.01% of image) to large (14% of image)
- Object density: from 1 to 255 objects per image
- Domains: scientific, industrial, nature, aerial, traffic
"""

import os
from pathlib import Path
from typing import Dict, List, Optional
import yaml

# Default cache directory (same as RF100 benchmark)
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "rf100"

# Roboflow API key for downloads (from environment variable)
# Set ROBOFLOW_API_KEY in your environment or .env file
ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY")
RF100_WORKSPACE = "roboflow-100"


# RF5 Dataset Definitions
# Each dataset covers a specific dimension of the test space
RF5_DATASETS = [
    {
        "name": "bacteria-ptywi",
        "category": "microscopic",
        "train_images": 30,
        "num_classes": 1,
        "objects_per_image": 61,
        "avg_object_area": 0.0001,  # 0.01% of image
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
        "avg_object_area": 0.1422,  # 14% of image
        "purpose": "Large objects with sparse annotations",
    },
]


def get_rf5_dataset_names() -> List[str]:
    """Get list of RF5 dataset names."""
    return [d["name"] for d in RF5_DATASETS]


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
    """
    Get the local path for a dataset if it exists.

    Args:
        dataset_name: Name of the dataset
        cache_dir: Cache directory (default: ~/.cache/rf100)

    Returns:
        Path to dataset if exists, None otherwise
    """
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
    """
    Download a dataset from Roboflow if not already cached.

    Args:
        dataset_name: Name of the dataset (e.g., "aquarium-qlnqy")
        cache_dir: Directory to cache datasets (default: ~/.cache/rf100)
        force: If True, re-download even if exists

    Returns:
        Path to the dataset directory

    Raises:
        RuntimeError: If download fails
    """
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    local_dir = cache_dir / dataset_name

    # Check if already downloaded
    if not force and local_dir.exists():
        data_yaml = local_dir / "data.yaml"
        if data_yaml.exists() and (local_dir / "train" / "labels").exists():
            return local_dir

    # Download from Roboflow
    try:
        if not ROBOFLOW_API_KEY:
            raise RuntimeError(
                "ROBOFLOW_API_KEY environment variable not set. "
                "Either set it in your environment or create a .env file. "
                "Alternatively, manually download datasets to ~/.cache/rf100/"
            )

        from roboflow import Roboflow

        rf = Roboflow(api_key=ROBOFLOW_API_KEY)
        project = rf.workspace(RF100_WORKSPACE).project(dataset_name)

        # Download version 1 in YOLOv8 format
        project.version(1).download(
            model_format="yolov8",
            location=str(local_dir),
            overwrite=force,
        )

        # Fix data.yaml paths
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

    with open(data_yaml, 'r') as f:
        data = yaml.safe_load(f)

    data['path'] = '.'
    data['train'] = 'train/images'
    data['val'] = 'valid/images'
    data['test'] = 'test/images'

    with open(data_yaml, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)


def ensure_rf5_datasets(cache_dir: Optional[Path] = None) -> Dict[str, Path]:
    """
    Ensure all RF5 datasets are available (download if needed).

    Args:
        cache_dir: Cache directory

    Returns:
        Dict mapping dataset names to their local paths
    """
    paths = {}

    for dataset in RF5_DATASETS:
        name = dataset["name"]
        print(f"Checking {name}...")

        path = get_dataset_path(name, cache_dir)
        if path is not None:
            print(f"  Found at {path}")
            paths[name] = path
        else:
            print(f"  Downloading...")
            try:
                path = download_dataset(name, cache_dir)
                paths[name] = path
                print(f"  Downloaded to {path}")
            except Exception as e:
                print(f"  Failed: {e}")

    return paths


def print_rf5_info():
    """Print information about RF5 datasets."""
    print("=" * 70)
    print("RF5 - Roboflow 5 Validation Subset")
    print("=" * 70)
    print()
    print("RF5 is a minimal validation subset of Roboflow100, designed to quickly")
    print("verify that training code works correctly.")
    print()
    print("Datasets:")
    print("-" * 70)
    print(f"{'Name':<25} {'Classes':>8} {'Train':>8} {'Obj/img':>8} {'Purpose'}")
    print("-" * 70)

    for d in RF5_DATASETS:
        print(
            f"{d['name']:<25} {d['num_classes']:>8} {d['train_images']:>8} "
            f"{d['objects_per_image']:>8} {d['purpose'][:30]}"
        )

    print("-" * 70)
    total_images = sum(d["train_images"] for d in RF5_DATASETS)
    print(f"{'Total':<25} {'':<8} {total_images:>8}")
    print()


if __name__ == "__main__":
    print_rf5_info()
    print()
    print("Checking dataset availability...")
    ensure_rf5_datasets()
