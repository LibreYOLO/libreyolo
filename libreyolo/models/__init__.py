"""
LibreYOLO model registry and unified factory.

All model families register here via ``__init_subclass__``. Adding a new model means:
1. Create models/<family>/ with model.py defining a class that inherits BaseModel
2. Add classmethods: can_load, detect_size, detect_nb_classes, detect_size_from_filename
3. Import the class so that ``__init_subclass__`` adds it to ``BaseModel._registry``
"""

from __future__ import annotations

from pathlib import Path

from .base import BaseModel
from ..utils.download import download_weights

# =============================================================================
# Model registry — auto-populated by BaseModel.__init_subclass__
# Order depends on import order: first match wins in can_load()
# =============================================================================

# Always-available models (importing triggers __init_subclass__ registration)
from .yolox.model import LibreYOLOX  # noqa: E402
from .yolo9.model import LibreYOLO9  # noqa: E402
from .yolonas.model import LibreYOLONAS  # noqa: E402
from .dfine.model import LibreDFINE  # noqa: E402


def _ensure_rfdetr():
    """Lazily register RF-DETR if its dependencies are installed."""
    if any(c.__name__ == "LibreYOLORFDETR" for c in BaseModel._registry):
        return
    import importlib.util

    if importlib.util.find_spec("rfdetr") is None:
        raise ModuleNotFoundError(
            "RF-DETR support requires extra dependencies.\n"
            "Install with: pip install libreyolo[rfdetr]"
        )
    from .rfdetr.model import LibreYOLORFDETR  # noqa: F401  (import triggers registration)


# =============================================================================
# Internal helpers
# =============================================================================


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
    """Extract weights from nested checkpoint formats.

    Supports:
    - LibreYOLO trainer checkpoints (``model``)
    - legacy EMA wrappers (``ema``)
    - SuperGradients checkpoints (``ema_net`` / ``net``)
    - generic wrappers (``state_dict``)
    """
    if "ema" in state_dict and isinstance(state_dict.get("ema"), dict):
        ema_data = state_dict["ema"]
        return ema_data.get("module", ema_data)
    if "ema_net" in state_dict and isinstance(state_dict.get("ema_net"), dict):
        return state_dict["ema_net"]
    if "net" in state_dict and isinstance(state_dict.get("net"), dict):
        return state_dict["net"]
    if "model" in state_dict and isinstance(state_dict.get("model"), dict):
        return state_dict["model"]
    if "state_dict" in state_dict and isinstance(state_dict.get("state_dict"), dict):
        return state_dict["state_dict"]
    return state_dict


# =============================================================================
# LibreYOLO — unified factory function
# =============================================================================


def LibreYOLO(
    model_path: str,
    size: str | None = None,
    reg_max: int = 16,
    nb_classes: int | None = None,
    device: str = "auto",
):
    """
    Unified factory that detects model family from weights and returns
    the appropriate model instance.

    Args:
        model_path: Path to weights (.pt), ONNX (.onnx), TensorRT (.engine),
                    or OpenVINO/ncnn directory.
        size: Model size variant (auto-detected from weights if omitted).
        reg_max: Regression max for DFL (YOLOv9 only, default: 16).
        nb_classes: Number of classes (auto-detected if omitted).
        device: Device for inference ("auto", "cuda", "cpu", "mps").

    Returns:
        Model instance (LibreYOLOX, LibreYOLO9, LibreYOLORFDETR, or inference backend).
    """
    import torch

    model_path = _resolve_weights_path(model_path)

    # Non-PyTorch formats: delegate to inference backends
    if model_path.endswith(".onnx"):
        from ..backends.onnx import OnnxBackend

        return OnnxBackend(model_path, nb_classes=nb_classes or 80, device=device)

    if model_path.endswith(".torchscript"):
        from ..backends.torchscript import TorchScriptBackend

        return TorchScriptBackend(model_path, nb_classes=nb_classes, device=device)

    if model_path.endswith((".engine", ".tensorrt")):
        from ..backends.tensorrt import TensorRTBackend

        return TensorRTBackend(model_path, nb_classes=nb_classes, device=device)

    if Path(model_path).is_dir() and (Path(model_path) / "model.xml").exists():
        from ..backends.openvino import OpenVINOBackend

        return OpenVINOBackend(model_path, nb_classes=nb_classes, device=device)

    if Path(model_path).is_dir():
        ncnn_param = Path(model_path) / "model.ncnn.param"
        ncnn_bin = Path(model_path) / "model.ncnn.bin"
        if ncnn_param.exists() and ncnn_bin.exists():
            from ..backends.ncnn import NcnnBackend

            return NcnnBackend(model_path, nb_classes=nb_classes, device=device)

    # Download if missing
    if not Path(model_path).exists():
        if size is None:
            for cls in BaseModel._registry:
                detected = cls.detect_size_from_filename(Path(model_path).name)
                if detected is not None:
                    size = detected
                    print(f"Detected size '{size}' from filename")
                    break
            # Try RF-DETR (may not be registered yet — cheap check)
            if size is None:
                try:
                    _ensure_rfdetr()
                    for cls in BaseModel._registry:
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

    # Load weights once
    try:
        state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load model weights from {model_path}: {e}"
        ) from e

    weights_dict = _unwrap_state_dict(state_dict)

    # Ensure RF-DETR is registered if its keys are present.
    # D-FINE also has ``encoder``/``decoder``/``class_embed``-ish keys — we
    # gate on ``dinov2`` (RF-DETR uses DINOv2 backbone, D-FINE uses HGNetV2)
    # or the RF-DETR-specific ``query_embed`` to avoid a spurious import
    # whenever D-FINE weights are loaded.
    keys_lower = [k.lower() for k in weights_dict]
    if any("dinov2" in k or "query_embed" in k for k in keys_lower):
        try:
            _ensure_rfdetr()
        except ModuleNotFoundError:
            raise

    # Find the right model class
    matched_cls = None
    for cls in BaseModel._registry:
        if cls.can_load(weights_dict):
            matched_cls = cls
            break

    if matched_cls is None:
        raise ValueError(
            "Could not detect model architecture from state dict keys.\n"
            "Supported architectures: YOLOX, YOLOv9, YOLO-NAS, RF-DETR."
        )

    # Auto-detect size
    if size is None:
        if matched_cls.FAMILY == "rfdetr":
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
                f"Please specify size explicitly: LibreYOLO('{model_path}', size='s')"
            )
        print(f"Auto-detected size: {size}")

    # Auto-detect nb_classes
    if nb_classes is None:
        nb_classes = matched_cls.detect_nb_classes(weights_dict)
        if nb_classes is None:
            nb_classes = 80

    # Determine how to pass weights
    # Checkpoints from our trainers have metadata (nc, names, model_family).
    # For those, pass the file path so _load_weights() handles nc rebuild + names.
    # For old/pretrained checkpoints, pass the extracted state_dict directly.
    has_metadata = isinstance(state_dict, dict) and "nc" in state_dict

    # Check for -seg suffix on models that don't support segmentation
    task = matched_cls.detect_task_from_filename(Path(model_path).name)
    if task == "seg" and matched_cls.FAMILY != "rfdetr":
        raise ValueError(
            f"{matched_cls.__name__} does not support segmentation. "
            f"Only RF-DETR models support instance segmentation (-seg suffix)."
        )

    if matched_cls.FAMILY == "rfdetr":
        # RF-DETR always needs the path (handles its own loading internally)
        model = matched_cls(
            model_path=model_path,
            size=size,
            nb_classes=nb_classes,
            device=device,
            segmentation=(task == "seg"),
        )
    elif has_metadata:
        # Our trainer checkpoint — pass path for metadata handling
        model = matched_cls(
            model_path=model_path,
            size=size,
            nb_classes=80,
            device=device,
            **({"reg_max": reg_max} if matched_cls.FAMILY == "yolo9" else {}),
        )
    else:
        # Pretrained checkpoint — pass extracted state dict
        model = matched_cls(
            model_path=weights_dict,
            size=size,
            nb_classes=nb_classes,
            device=device,
            **({"reg_max": reg_max} if matched_cls.FAMILY == "yolo9" else {}),
        )

    model.model_path = model_path
    return model


__all__ = [
    "LibreYOLO",
    "LibreYOLOX",
    "LibreYOLO9",
]
