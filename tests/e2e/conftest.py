"""E2E test configuration and fixtures."""

import gc

import pytest
import torch


def pytest_configure(config):
    """Register custom markers (e2e marker registered in root conftest)."""
    config.addinivalue_line("markers", "tensorrt: tests requiring TensorRT")
    config.addinivalue_line("markers", "openvino: tests requiring OpenVINO")
    config.addinivalue_line("markers", "ncnn: tests requiring ncnn")
    config.addinivalue_line("markers", "rfdetr: tests requiring RF-DETR dependencies")
    config.addinivalue_line("markers", "slow: slow tests that may take several minutes")
    config.addinivalue_line("markers", "rf5: RF5 training benchmark tests")


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

requires_cuda = pytest.mark.skipif(not has_cuda(), reason="CUDA not available")

requires_tensorrt = pytest.mark.skipif(
    not has_tensorrt(), reason="TensorRT not installed or CUDA not available"
)

requires_openvino = pytest.mark.skipif(
    not has_openvino(), reason="OpenVINO not installed (pip install openvino)"
)

requires_ncnn = pytest.mark.skipif(
    not has_ncnn(), reason="ncnn not installed (pip install libreyolo[ncnn])"
)

requires_rfdetr = pytest.mark.skipif(
    not has_rfdetr_deps(),
    reason="RF-DETR dependencies not installed (pip install libreyolo[rfdetr])",
)


# ---------------------------------------------------------------------------
# GPU helpers
# ---------------------------------------------------------------------------


def cuda_cleanup():
    """Free GPU memory. Call after heavy tests."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


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


@pytest.fixture(autouse=True, scope="function")
def cleanup_gpu_memory():
    """Clear GPU memory before and after each test to prevent state corruption."""
    yield
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()


@pytest.fixture(scope="class")
def reset_gpu_state():
    """Force GPU state reset between test classes (useful for RF-DETR training)."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()
    yield
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()


# ---------------------------------------------------------------------------
# Model catalog — single source of truth
# ---------------------------------------------------------------------------

# (family, size, weights)
MODEL_CATALOG = [
    ("yolox", "n", "LibreYOLOXn.pt"),
    ("yolox", "t", "LibreYOLOXt.pt"),
    ("yolox", "s", "LibreYOLOXs.pt"),
    ("yolox", "m", "LibreYOLOXm.pt"),
    ("yolox", "l", "LibreYOLOXl.pt"),
    ("yolox", "x", "LibreYOLOXx.pt"),
    ("yolo9", "t", "LibreYOLO9t.pt"),
    ("yolo9", "s", "LibreYOLO9s.pt"),
    ("yolo9", "m", "LibreYOLO9m.pt"),
    ("yolo9", "c", "LibreYOLO9c.pt"),
    ("rfdetr", "n", "LibreRFDETRn.pth"),
    ("rfdetr", "s", "LibreRFDETRs.pth"),
    ("rfdetr", "m", "LibreRFDETRm.pth"),
    ("rfdetr", "l", "LibreRFDETRl.pth"),
]

# Derived lists (no manual maintenance)
YOLOX_SIZES = [s for f, s, _ in MODEL_CATALOG if f == "yolox"]
YOLO9_SIZES = [s for f, s, _ in MODEL_CATALOG if f == "yolo9"]
RFDETR_SIZES = [s for f, s, _ in MODEL_CATALOG if f == "rfdetr"]

ALL_MODELS = [(f, s) for f, s, _ in MODEL_CATALOG]
ALL_MODELS_WITH_WEIGHTS = MODEL_CATALOG
YOLOX_YOLO9_MODELS = [(f, s) for f, s, _ in MODEL_CATALOG if f != "rfdetr"]

# Quick test set (for CI — smallest models only)
QUICK_TEST_MODELS = [("yolox", "n"), ("yolo9", "t")]

# Full test set (YOLOX + YOLO9, no RF-DETR)
FULL_TEST_MODELS = YOLOX_YOLO9_MODELS

# RF-DETR test set (separate due to dependency)
RFDETR_TEST_MODELS = [(f, s) for f, s, _ in MODEL_CATALOG if f == "rfdetr"]


def get_model_weights(family: str, size: str) -> str:
    """Get the weight file name for a model family and size."""
    for f, s, w in MODEL_CATALOG:
        if f == family and s == size:
            return w
    raise ValueError(f"Unknown model: {family}-{size}")


def make_ids(models):
    """Generate test IDs like 'yolox-n' from (family, size, ...) tuples."""
    return [f"{m[0]}-{m[1]}" for m in models]


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
        match_rate,
        len(pt_results),
        len(export_results),
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
