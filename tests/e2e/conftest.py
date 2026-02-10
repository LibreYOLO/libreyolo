"""E2E test configuration and fixtures."""

import pytest
import torch


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
        from libreyolo.rfdetr.model import LIBREYOLORFDETR
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
YOLOX_SIZES = ["nano", "tiny", "s", "m", "l", "x"]

# YOLOv9 model sizes
YOLOV9_SIZES = ["t", "s", "m", "c"]

# RF-DETR model sizes
RFDETR_SIZES = ["n", "s", "m", "l"]

# Model weight file patterns
YOLOX_WEIGHTS = {
    "nano": "libreyoloXnano.pt",
    "tiny": "libreyoloXtiny.pt",
    "s": "libreyoloXs.pt",
    "m": "libreyoloXm.pt",
    "l": "libreyoloXl.pt",
    "x": "libreyoloXx.pt",
}

YOLOV9_WEIGHTS = {
    "t": "libreyolo9t.pt",
    "s": "libreyolo9s.pt",
    "m": "libreyolo9m.pt",
    "c": "libreyolo9c.pt",
}

RFDETR_WEIGHTS = {
    "n": "librerfdetrnano.pth",
    "s": "librerfdetrsmall.pth",
    "m": "librerfdetrmedium.pth",
    "l": "librerfdetrlarge.pth",
}

# Quick test set (for CI - smallest models only)
QUICK_TEST_MODELS = [
    ("yolox", "nano"),
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
    """Compute IoU between two boxes in xyxy format."""
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
    from libreyolo import LIBREYOLO

    weights = get_model_weights(model_type, size)
    return LIBREYOLO(weights, device=device)


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
