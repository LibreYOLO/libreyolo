"""
LibreYOLO model registry and unified factory.

All model families register here. Adding a new model means:
1. Create models/<family>/ with model.py defining a class that inherits LibreYOLOBase
2. Add classmethods: can_load, detect_size, detect_nb_classes, detect_size_from_filename
3. Import and append to MODEL_REGISTRY below
"""

import re
import requests
from pathlib import Path
from typing import List, Optional, Type

from .base import LibreYOLOBase

# ---------------------------------------------------------------------------
# Model registry — order matters: first match wins in can_load()
# ---------------------------------------------------------------------------
MODEL_REGISTRY: List[Type[LibreYOLOBase]] = []

# Always-available models
from .yolox.model import LIBREYOLOX   # noqa: E402
from .yolo9.model import LIBREYOLO9      # noqa: E402

MODEL_REGISTRY.extend([LIBREYOLOX, LIBREYOLO9])


def _ensure_rfdetr():
    """Lazily register RF-DETR if its dependencies are installed."""
    if any(c.__name__ == "LIBREYOLORFDETR" for c in MODEL_REGISTRY):
        return
    import importlib.util
    if importlib.util.find_spec("rfdetr") is None:
        raise ModuleNotFoundError(
            "RF-DETR support requires extra dependencies.\n"
            "Install with: pip install libreyolo[rfdetr]"
        )
    from .rfdetr.model import LIBREYOLORFDETR  # noqa: F811
    MODEL_REGISTRY.append(LIBREYOLORFDETR)


# ---------------------------------------------------------------------------
# Internal helpers (ported from old factory.py)
# ---------------------------------------------------------------------------

def _resolve_weights_path(model_path: str) -> str:
    """Resolve bare filenames to weights/ directory."""
    path = Path(model_path)
    if path.parent == Path(".") and not model_path.startswith(("./", "../")):
        weights_path = Path("weights") / path.name
        if weights_path.exists():
            return str(weights_path)
        if path.exists():
            return str(path)
        return str(weights_path)
    return model_path


def _unwrap_state_dict(state_dict: dict) -> dict:
    """Extract weights from nested checkpoint formats (EMA, model wrappers)."""
    if "ema" in state_dict and isinstance(state_dict.get("ema"), dict):
        ema_data = state_dict["ema"]
        return ema_data.get("module", ema_data)
    if "model" in state_dict and isinstance(state_dict.get("model"), dict):
        return state_dict["model"]
    return state_dict


def _detect_family_from_filename(filename: str) -> Optional[str]:
    """Return model family hint from filename (for download routing only)."""
    fl = filename.lower()
    if re.search(r"librerfdetr", fl):
        return "rfdetr"
    if re.search(r"libreyolox", fl):
        return "yolox"
    if re.search(r"libreyolo9|yolov?9", fl):
        return "v9"
    return None


def download_weights(model_path: str, size: str):
    """Download weights from Hugging Face if not found locally."""
    path = Path(model_path)
    if path.exists():
        return

    filename = path.name

    # RF-DETR
    if re.search(r"librerfdetr(nano|small|medium|large)", filename.lower()):
        m = re.search(r"librerfdetr(nano|small|medium|large)", filename.lower())
        rfdetr_size = m.group(1)
        repo = f"Libre-YOLO/librerfdetr{rfdetr_size}"
        if rfdetr_size == "large":
            actual_filename = "rf-detr-large-2026.pth"
        else:
            actual_filename = f"rf-detr-{rfdetr_size}.pth"
        url = f"https://huggingface.co/{repo}/resolve/main/{actual_filename}"
    # YOLOX
    elif re.search(r"libreyolox(nano|tiny|s|m|l|x)", filename.lower()):
        yolox_match = re.search(r"libreyolox(nano|tiny|s|m|l|x)", filename.lower())
        yolox_size = yolox_match.group(1)
        repo = f"Libre-YOLO/libreyoloX{yolox_size}"
        url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
    # YOLOv9
    elif re.search(r"libreyolo9|yolov?9", filename.lower()):
        repo = f"Libre-YOLO/libreyolo9{size}"
        url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
    else:
        raise ValueError(
            f"Could not determine model version from filename '{filename}' for auto-download."
        )

    print(f"Model weights not found at {model_path}. Attempting download from {url}...")
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))

        with open(path, "wb") as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = int(100 * downloaded / total_size)
                        print(
                            f"\rDownloading: {percent}% ({downloaded/1024/1024:.1f}/{total_size/1024/1024:.1f} MB)",
                            end="", flush=True,
                        )
            print("\nDownload complete.")
    except Exception as e:
        if path.exists():
            path.unlink()
        raise RuntimeError(f"Failed to download weights from {url}: {e}") from e


# ---------------------------------------------------------------------------
# LIBREYOLO — unified factory function
# ---------------------------------------------------------------------------

def LIBREYOLO(
    model_path: str,
    size: str = None,
    reg_max: int = 16,
    nb_classes: int = None,
    device: str = "auto",
):
    """
    Unified factory that detects model family from weights and returns
    the appropriate model instance.

    Args:
        model_path: Path to weights (.pt/.pth), ONNX (.onnx), TensorRT (.engine),
                    or OpenVINO/ncnn directory.
        size: Model size variant (auto-detected from weights if omitted).
        reg_max: Regression max for DFL (YOLOv9 only, default: 16).
        nb_classes: Number of classes (auto-detected if omitted).
        device: Device for inference ("auto", "cuda", "cpu", "mps").

    Returns:
        Model instance (LIBREYOLOX, LIBREYOLO9, LIBREYOLORFDETR, or inference backend).
    """
    import torch

    model_path = _resolve_weights_path(model_path)

    # --- Non-PyTorch formats: delegate to inference backends ---
    if model_path.endswith(".onnx"):
        from ..inference.onnx import LIBREYOLOOnnx
        return LIBREYOLOOnnx(model_path, nb_classes=nb_classes or 80, device=device)

    if model_path.endswith(".engine"):
        from ..inference.tensorrt import LIBREYOLOTensorRT
        return LIBREYOLOTensorRT(model_path, nb_classes=nb_classes, device=device)

    if Path(model_path).is_dir() and (Path(model_path) / "model.xml").exists():
        from ..inference.openvino import LIBREYOLOOpenVINO
        return LIBREYOLOOpenVINO(model_path, nb_classes=nb_classes, device=device)

    if Path(model_path).is_dir():
        ncnn_param = Path(model_path) / "model.ncnn.param"
        ncnn_bin = Path(model_path) / "model.ncnn.bin"
        if ncnn_param.exists() and ncnn_bin.exists():
            from ..inference.ncnn import LIBREYOLONCNN
            return LIBREYOLONCNN(model_path, nb_classes=nb_classes, device=device)

    # --- Download if missing ---
    if not Path(model_path).exists():
        if size is None:
            # Ask each registered model class for a size hint from the filename
            for cls in MODEL_REGISTRY:
                detected = cls.detect_size_from_filename(Path(model_path).name)
                if detected is not None:
                    size = detected
                    print(f"Detected size '{size}' from filename")
                    break
            # Try RF-DETR (may not be registered yet)
            if size is None:
                family_hint = _detect_family_from_filename(Path(model_path).name)
                if family_hint == "rfdetr":
                    try:
                        _ensure_rfdetr()
                        for cls in MODEL_REGISTRY:
                            detected = cls.detect_size_from_filename(Path(model_path).name)
                            if detected is not None:
                                size = detected
                                print(f"Detected size '{size}' from filename")
                                break
                    except ModuleNotFoundError:
                        pass
            if size is None:
                raise ValueError(
                    f"Model weights file not found: {model_path}\n"
                    f"Cannot auto-download: unable to determine size from filename.\n"
                    f"Please specify size explicitly or provide a valid weights file path."
                )

        try:
            download_weights(model_path, size)
        except Exception as e:
            print(f"Auto-download failed: {e}")

    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model weights file not found: {model_path}")

    # --- Load weights once ---
    try:
        state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load model weights from {model_path}: {e}") from e

    weights_dict = _unwrap_state_dict(state_dict)

    # --- Ensure RF-DETR is registered if its keys are present ---
    keys_lower = [k.lower() for k in weights_dict]
    if any(
        "detr" in k or "dinov2" in k or "transformer" in k
        or ("encoder" in k and "decoder" in k) or "query_embed" in k
        or "class_embed" in k or "bbox_embed" in k
        for k in keys_lower
    ):
        try:
            _ensure_rfdetr()
        except ModuleNotFoundError:
            raise

    # --- Find the right model class ---
    matched_cls = None
    for cls in MODEL_REGISTRY:
        if cls.can_load(weights_dict):
            matched_cls = cls
            break

    if matched_cls is None:
        raise ValueError(
            "Could not detect model architecture from state dict keys.\n"
            "Supported architectures: YOLOX, YOLOv9, RF-DETR."
        )

    # --- Auto-detect size ---
    if size is None:
        if matched_cls.__name__ == "LIBREYOLORFDETR":
            # RF-DETR needs the full checkpoint for args-based detection
            size = matched_cls.detect_size(weights_dict, state_dict=state_dict)
        else:
            size = matched_cls.detect_size(weights_dict)

        if size is None:
            # Fallback: try filename
            size = matched_cls.detect_size_from_filename(Path(model_path).name)

        if size is None:
            raise ValueError(
                f"Could not automatically detect {matched_cls.__name__} model size.\n"
                f"Please specify size explicitly: LIBREYOLO('{model_path}', size='s')"
            )
        print(f"Auto-detected size: {size}")

    # --- Auto-detect nb_classes ---
    if nb_classes is None:
        nb_classes = matched_cls.detect_nb_classes(weights_dict)
        if nb_classes is None:
            nb_classes = 80

    # --- Determine how to pass weights ---
    # Checkpoints from our trainers have metadata (nc, names, model_family).
    # For those, pass the file path so _load_weights() handles nc rebuild + names.
    # For old/pretrained checkpoints, pass the extracted state_dict directly.
    has_metadata = isinstance(state_dict, dict) and "nc" in state_dict

    if matched_cls.__name__ == "LIBREYOLORFDETR":
        # RF-DETR always needs the path (handles its own loading internally)
        model = matched_cls(
            model_path=model_path, size=size, nb_classes=nb_classes, device=device
        )
    elif has_metadata:
        # Our trainer checkpoint — pass path for metadata handling
        model = matched_cls(
            model_path=model_path, size=size, nb_classes=80,
            device=device, **({"reg_max": reg_max} if matched_cls.__name__ == "LIBREYOLO9" else {})
        )
    else:
        # Pretrained checkpoint — pass extracted state dict
        model = matched_cls(
            model_path=weights_dict, size=size, nb_classes=nb_classes,
            device=device, **({"reg_max": reg_max} if matched_cls.__name__ == "LIBREYOLO9" else {})
        )

    model.model_path = model_path
    return model


__all__ = [
    "MODEL_REGISTRY",
    "LIBREYOLO",
    "LIBREYOLOX",
    "LIBREYOLO9",
    "LibreYOLOBase",
]
