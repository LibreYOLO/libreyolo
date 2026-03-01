"""E2E test configuration and fixtures."""

import os
from pathlib import Path
from typing import Dict, List, Optional

import pytest
import torch
import yaml


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "e2e: end-to-end tests requiring full model loading")
    config.addinivalue_line("markers", "tensorrt: tests requiring TensorRT")
    config.addinivalue_line("markers", "openvino: tests requiring OpenVINO")
    config.addinivalue_line("markers", "ncnn: tests requiring ncnn")
    config.addinivalue_line("markers", "rfdetr: tests requiring RF-DETR dependencies")
    config.addinivalue_line("markers", "slow: slow tests that may take several minutes")


# ---------------------------------------------------------------------------
# Hardware detection
# ---------------------------------------------------------------------------

def has_cuda():
    """Check if CUDA is available."""
    return torch.cuda.is_available()


def has_tensorrt():
    """Check if TensorRT is installed and usable."""
    if not has_cuda():
        return False
    try:
        import tensorrt as trt
        return True
    except ImportError:
        return False


def has_openvino():
    """Check if OpenVINO is installed and usable."""
    try:
        import openvino as ov
        _ = ov.__version__
        return True
    except ImportError:
        return False


def has_ncnn():
    """Check if ncnn is installed and usable."""
    try:
        import ncnn
        return True
    except ImportError:
        return False


def has_rfdetr_deps():
    """Check if RF-DETR dependencies are installed."""
    try:
        from libreyolo.models.rfdetr.model import LibreYOLORFDETR
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Skip decorators
# ---------------------------------------------------------------------------

requires_cuda = pytest.mark.skipif(
    not has_cuda(),
    reason="CUDA not available"
)

requires_tensorrt = pytest.mark.skipif(
    not has_tensorrt(),
    reason="TensorRT not installed or CUDA not available"
)

requires_openvino = pytest.mark.skipif(
    not has_openvino(),
    reason="OpenVINO not installed (pip install openvino)"
)

requires_ncnn = pytest.mark.skipif(
    not has_ncnn(),
    reason="ncnn not installed (pip install libreyolo[ncnn])"
)

requires_rfdetr = pytest.mark.skipif(
    not has_rfdetr_deps(),
    reason="RF-DETR dependencies not installed (pip install libreyolo[rfdetr])"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def cuda_device():
    """Return CUDA device string if available."""
    if has_cuda():
        return "cuda"
    pytest.skip("CUDA not available")


@pytest.fixture(scope="session")
def gpu_info():
    """Return GPU information for logging."""
    if not has_cuda():
        return {"available": False}

    return {
        "available": True,
        "name": torch.cuda.get_device_name(0),
        "cuda_version": torch.version.cuda,
        "device_count": torch.cuda.device_count(),
    }


@pytest.fixture(scope="session")
def tensorrt_version():
    """Return TensorRT version if available."""
    if not has_tensorrt():
        pytest.skip("TensorRT not available")

    import tensorrt as trt
    return trt.__version__


@pytest.fixture(scope="session")
def sample_image():
    """Get a sample image for inference tests."""
    from libreyolo import SAMPLE_IMAGE
    return SAMPLE_IMAGE


@pytest.fixture(scope="function")
def temp_export_dir(tmp_path):
    """Create a temporary directory for export artifacts."""
    return tmp_path / "exports"


# ---------------------------------------------------------------------------
# Model parameter sets
# ---------------------------------------------------------------------------

# YOLOX model sizes
YOLOX_SIZES = ["n", "t", "s", "m", "l", "x"]

# YOLOv9 model sizes
YOLOV9_SIZES = ["t", "s", "m", "c"]

# RF-DETR model sizes
RFDETR_SIZES = ["n", "s", "m", "l"]

# Model weight file patterns
YOLOX_WEIGHTS = {
    "n": "LibreYOLOXn.pt",
    "t": "LibreYOLOXt.pt",
    "s": "LibreYOLOXs.pt",
    "m": "LibreYOLOXm.pt",
    "l": "LibreYOLOXl.pt",
    "x": "LibreYOLOXx.pt",
}

YOLOV9_WEIGHTS = {
    "t": "LibreYOLO9t.pt",
    "s": "LibreYOLO9s.pt",
    "m": "LibreYOLO9m.pt",
    "c": "LibreYOLO9c.pt",
}

RFDETR_WEIGHTS = {
    "n": "LibreRFDETRn.pth",
    "s": "LibreRFDETRs.pth",
    "m": "LibreRFDETRm.pth",
    "l": "LibreRFDETRl.pth",
}

# Quick test set (for CI - smallest models only)
QUICK_TEST_MODELS = [
    ("yolox", "n"),
    ("yolov9", "t"),
]

# Full test set (all models)
FULL_TEST_MODELS = (
    [("yolox", size) for size in YOLOX_SIZES] +
    [("yolov9", size) for size in YOLOV9_SIZES]
)

# RF-DETR test set (separate due to dependency)
RFDETR_TEST_MODELS = [("rfdetr", size) for size in RFDETR_SIZES]


def get_model_weights(model_type: str, size: str) -> str:
    """Get the weight file name for a model type and size."""
    if model_type == "yolox":
        return YOLOX_WEIGHTS[size]
    elif model_type == "yolov9":
        return YOLOV9_WEIGHTS[size]
    elif model_type == "rfdetr":
        return RFDETR_WEIGHTS[size]
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# ---------------------------------------------------------------------------
# Common utility functions for E2E tests
# ---------------------------------------------------------------------------

def compute_iou(box1, box2):
    """Compute IoU between two boxes in xyxy format.

    Simple Python implementation for test utilities (no torch dependency).
    For GPU-accelerated IoU, see libreyolo.utils.box_ops.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0


def match_detections(results1, results2, iou_threshold=0.5):
    """
    Match detections between two result sets.

    Returns:
        Tuple of (match_rate, matched_count, total_count)
    """
    if len(results1) == 0 and len(results2) == 0:
        return 1.0, 0, 0  # Perfect match (both empty)

    if len(results1) == 0 or len(results2) == 0:
        return 0.0, 0, max(len(results1), len(results2))

    boxes1 = results1.boxes.xyxy.cpu().numpy()
    classes1 = results1.boxes.cls.cpu().numpy()

    boxes2 = results2.boxes.xyxy.cpu().numpy()
    classes2 = results2.boxes.cls.cpu().numpy()

    matched = 0
    for box1, cls1 in zip(boxes1, classes1):
        for box2, cls2 in zip(boxes2, classes2):
            if cls1 == cls2:
                iou = compute_iou(box1, box2)
                if iou >= iou_threshold:
                    matched += 1
                    break

    total = max(len(boxes1), len(boxes2))
    match_rate = matched / total if total > 0 else 1.0

    return match_rate, matched, total


def load_model(model_type: str, size: str, device: str = "cuda"):
    """Load a model by type and size."""
    from libreyolo import LibreYOLO

    weights = get_model_weights(model_type, size)
    return LibreYOLO(weights, device=device)


def results_are_acceptable(
    match_rate: float, count1: int, count2: int, threshold: float = 0.7
) -> bool:
    """
    Check if results are acceptable.

    Args:
        match_rate: Detection match rate
        count1: Number of detections in first result
        count2: Number of detections in second result
        threshold: Minimum acceptable match rate (default 0.7)
    """
    if match_rate >= threshold:
        return True

    det_diff = abs(count1 - count2)
    if det_diff <= 2 and count1 > 0:
        return True

    if count1 == 0 and count2 == 0:
        return True

    return False


# ---------------------------------------------------------------------------
# RF5 dataset infrastructure
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Shared export test helpers
# ---------------------------------------------------------------------------


def run_export_compare_test(
    model_type,
    size,
    sample_image,
    tmp_path,
    format,
    export_kwargs=None,
    match_threshold=0.7,
    device="cpu",
):
    """
    Common export -> inference -> compare flow.

    Returns (exported_path, pt_results, export_results).
    """
    from libreyolo import LibreYOLO

    pt_model = load_model(model_type, size, device=device)
    pt_results = pt_model(sample_image, conf=0.25)

    export_path = str(tmp_path / f"{model_type}_{size}.{format}")
    exported_path = pt_model.export(
        format=format,
        output_path=export_path,
        **(export_kwargs or {}),
    )

    exported_model = LibreYOLO(exported_path, device=device)
    export_results = exported_model(sample_image, conf=0.25)

    match_rate, matched, total = match_detections(pt_results, export_results)
    assert results_are_acceptable(
        match_rate, len(pt_results), len(export_results),
        threshold=match_threshold,
    ), (
        f"Results mismatch: PT={len(pt_results)}, {format}={len(export_results)}, "
        f"matched={matched}/{total}, rate={match_rate:.2%}"
    )

    return exported_path, pt_results, export_results


def run_consistency_test(
    model_type,
    size,
    sample_image,
    tmp_path,
    format,
    export_kwargs=None,
    device="cpu",
    n_runs=5,
):
    """Export model and verify consistent inference results across N runs."""
    from libreyolo import LibreYOLO

    pt_model = load_model(model_type, size, device=device)
    export_path = str(tmp_path / f"{model_type}_{size}.{format}")
    exported_path = pt_model.export(
        format=format, output_path=export_path, **(export_kwargs or {})
    )

    model = LibreYOLO(exported_path, device=device)
    results = [len(model(sample_image, conf=0.25)) for _ in range(n_runs)]
    assert len(set(results)) == 1, f"Inconsistent results across runs: {results}"


def run_metadata_round_trip_test(
    model_type,
    size,
    tmp_path,
    format,
    export_kwargs=None,
    device="cpu",
):
    """Export model and verify metadata is preserved when loading."""
    from libreyolo import LibreYOLO

    pt_model = load_model(model_type, size, device=device)
    export_path = str(tmp_path / f"{model_type}_{size}.{format}")
    exported_path = pt_model.export(
        format=format, output_path=export_path, **(export_kwargs or {})
    )

    loaded_model = LibreYOLO(exported_path, device=device)
    assert loaded_model.nb_classes == pt_model.nb_classes
    assert loaded_model.names == pt_model.names
