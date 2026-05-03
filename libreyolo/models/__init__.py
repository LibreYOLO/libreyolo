"""
LibreYOLO model registry and unified factory.

All model families register here via ``__init_subclass__``. Adding a new model means:
1. Create models/<family>/ with model.py defining a class that inherits BaseModel
2. Add classmethods: can_load, detect_size, detect_nb_classes, detect_size_from_filename
3. Import the class so that ``__init_subclass__`` adds it to ``BaseModel._registry``
"""

from __future__ import annotations

import logging
from pathlib import Path

from .base import BaseModel
from ..tasks import resolve_task
from ..utils.download import download_weights
from ..utils.logging import ensure_default_logging
from ..utils.serialization import load_untrusted_torch_file

logger = logging.getLogger(__name__)

# =============================================================================
# Model registry — auto-populated by BaseModel.__init_subclass__
# Order depends on import order: first match wins in can_load()
# =============================================================================

# Always-available models (importing triggers __init_subclass__ registration)
# Order matters: more-specific can_load() checks must run first. ECDET's ViT
# backbone keys ("backbone.backbone.register_token") are uniquely identifying,
# so register it before YOLOX which matches the broader "backbone.backbone"
# prefix (skill landmine §9.3).
# NOTE: LibreYOLO9E2E *must* be imported before LibreYOLO9.  E2E checkpoints
# contain all the same backbone/neck key patterns that LibreYOLO9.can_load
# matches, so the E2E discriminator (one2one_cv2 / one2one_cv3) must win first.
from .ecdet.model import LibreECDET  # noqa: E402
from .yolox.model import LibreYOLOX  # noqa: E402
from .yolo9_e2e.model import LibreYOLO9E2E  # noqa: E402
from .yolo9.model import LibreYOLO9  # noqa: E402
from .yolonas.model import LibreYOLONAS  # noqa: E402
from .deimv2.model import LibreDEIMv2  # noqa: E402
from .dfine.model import LibreDFINE  # noqa: E402
from .deim.model import LibreDEIM  # noqa: E402
from .picodet.model import LibrePicoDet  # noqa: E402
from .rtdetr.model import LibreYOLORTDETR  # noqa: E402


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


def try_ensure_rfdetr():
    """Try to register RF-DETR. Returns the model class or ``None`` if unavailable."""
    try:
        _ensure_rfdetr()
    except (ImportError, ModuleNotFoundError):
        return None
    for cls in BaseModel._registry:
        if cls.__name__ == "LibreYOLORFDETR":
            return cls
    return None


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


def _needs_rfdetr_registration(weights_dict: dict) -> bool:
    """Return True when checkpoint keys require lazy RF-DETR registration."""
    if LibreYOLORTDETR.can_load(weights_dict):
        return False

    keys_lower = [k.lower() for k in weights_dict]
    return any(
        "dinov2" in k
        or "query_embed" in k
        or "enc_out_class_embed" in k
        or "enc_out_bbox_embed" in k
        for k in keys_lower
    )


def _find_registered_family(family: str):
    for cls in BaseModel._registry:
        if cls.FAMILY == family:
            return cls
    return None


def _matching_model_classes(weights_dict: dict):
    return [cls for cls in BaseModel._registry if cls.can_load(weights_dict)]


# =============================================================================
# LibreYOLO — unified factory function
# =============================================================================


def LibreYOLO(
    model_path: str,
    size: str | None = None,
    reg_max: int = 16,
    nb_classes: int | None = None,
    device: str = "auto",
    task: str | None = None,
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
        task: Optional explicit task ("detect", "segment", "pose", "classify").

    Returns:
        Model instance (LibreYOLOX, LibreYOLO9, LibreYOLORFDETR, or inference backend).
    """
    ensure_default_logging()
    model_path = _resolve_weights_path(model_path)

    if task is not None:
        filename = Path(model_path).name
        for cls in BaseModel._registry:
            if cls.detect_size_from_filename(filename) is not None:
                resolve_task(
                    explicit_task=task,
                    default_task=cls.DEFAULT_TASK,
                    supported_tasks=cls.SUPPORTED_TASKS,
                )
                break

    # Non-PyTorch formats: delegate to inference backends
    if model_path.endswith(".onnx"):
        from ..backends.onnx import OnnxBackend

        return OnnxBackend(model_path, nb_classes=nb_classes or 80, device=device, task=task)

    if model_path.endswith(".torchscript"):
        from ..backends.torchscript import TorchScriptBackend

        return TorchScriptBackend(model_path, nb_classes=nb_classes, device=device, task=task)

    if model_path.endswith((".engine", ".tensorrt")):
        from ..backends.tensorrt import TensorRTBackend

        return TensorRTBackend(model_path, nb_classes=nb_classes, device=device, task=task)

    if Path(model_path).is_dir() and (Path(model_path) / "model.xml").exists():
        from ..backends.openvino import OpenVINOBackend

        return OpenVINOBackend(model_path, nb_classes=nb_classes, device=device, task=task)

    if Path(model_path).is_dir():
        ncnn_param = Path(model_path) / "model.ncnn.param"
        ncnn_bin = Path(model_path) / "model.ncnn.bin"
        if ncnn_param.exists() and ncnn_bin.exists():
            from ..backends.ncnn import NcnnBackend

            return NcnnBackend(model_path, nb_classes=nb_classes, device=device, task=task)

    # Download if missing
    if not Path(model_path).exists():
        if size is None:
            for cls in BaseModel._registry:
                detected = cls.detect_size_from_filename(Path(model_path).name)
                if detected is not None:
                    size = detected
                    logger.debug("Detected size '%s' from filename", size)
                    break
            # Try RF-DETR (may not be registered yet — cheap check)
            if size is None:
                try:
                    _ensure_rfdetr()
                    for cls in BaseModel._registry:
                        detected = cls.detect_size_from_filename(Path(model_path).name)
                        if detected is not None:
                            size = detected
                            logger.debug("Detected size '%s' from filename", size)
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
            logger.warning("Auto-download failed: %s", e)

    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model weights file not found: {model_path}")

    # Load weights once
    try:
        if Path(model_path).suffix == ".safetensors":
            try:
                from safetensors.torch import load_file as load_safetensors_file
            except ImportError as e:
                raise ImportError(
                    "Loading safetensors weights requires safetensors. "
                    "Install with: pip install safetensors"
                ) from e

            state_dict = load_safetensors_file(model_path, device="cpu")
        else:
            state_dict = load_untrusted_torch_file(
                model_path,
                map_location="cpu",
                context="model inspection",
            )
    except Exception as e:
        raise RuntimeError(
            f"Failed to load model weights from {model_path}: {e}"
        ) from e

    weights_dict = _unwrap_state_dict(state_dict)

    # Ensure RF-DETR is registered if its keys are present, but avoid
    # treating RT-DETR checkpoints as RF-DETR. D-FINE also has
    # ``encoder``/``decoder``-ish keys, so only RF-DETR-specific markers
    # should trigger the lazy import.
    if _needs_rfdetr_registration(weights_dict):
        try:
            _ensure_rfdetr()
        except ModuleNotFoundError:
            raise

    # Find the right model class. Metadata and filename hints come first so
    # DEIM-D-FINE and D-FINE, which intentionally share architecture keys, can
    # coexist without one stealing the other's LibreYOLO-format checkpoints.
    matched_cls = None
    metadata_family = (
        state_dict.get("model_family")
        if isinstance(state_dict, dict)
        and isinstance(state_dict.get("model_family"), str)
        else None
    )
    if metadata_family:
        cls = _find_registered_family(metadata_family)
        if cls is not None and cls.can_load(weights_dict):
            matched_cls = cls

    if matched_cls is None:
        filename = Path(model_path).name
        for cls in BaseModel._registry:
            if cls.detect_size_from_filename(filename) and cls.can_load(weights_dict):
                matched_cls = cls
                break

    if matched_cls is None:
        matching_classes = _matching_model_classes(weights_dict)
        matching_families = {cls.FAMILY for cls in matching_classes}
        # Only raise on a true D-FINE/DEIM tie. Some optional families can add
        # broader false-positive matches after lazy registration, while ECDET
        # and DEIMv2 legitimately match D-FINE/DEIM-ish decoder keys and should
        # be allowed to win via their more-specific detectors.
        if {"dfine", "deim"}.issubset(matching_families) and not (
            matching_families & {"ecdet", "deimv2"}
        ):
            raise ValueError(
                "Ambiguous D-FINE/DEIM checkpoint: both families share the same "
                "DEIM-D-FINE architecture keys.\n"
                "Use a LibreYOLO checkpoint with model_family metadata, an "
                "upstream-style filename such as dfine_hgnetv2_n_coco.pth or "
                "deim_hgnetv2_n_coco.pth, or instantiate LibreDFINE/LibreDEIM "
                "directly."
            )
        if matching_classes:
            matched_cls = matching_classes[0]

    if matched_cls is None:
        raise ValueError(
            "Could not detect model architecture from state dict keys.\n"
            "Supported architectures: YOLOX, YOLOv9, YOLOv9-E2E, YOLO-NAS, RT-DETR, RF-DETR, D-FINE, DEIM, DEIMv2."
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
        logger.debug("Auto-detected size: %s", size)

    # Determine how to pass weights
    # Checkpoints from our trainers have metadata (nc, names, model_family).
    # For those, pass the file path so _load_weights() handles nc rebuild + names.
    # For old/pretrained checkpoints, pass the extracted state_dict directly.
    has_metadata = isinstance(state_dict, dict) and "nc" in state_dict

    # Auto-detect nb_classes.
    #
    # Metadata checkpoints are reloaded via ``_load_weights()``, which reads the
    # saved ``nc`` and performs any family-specific rebuild logic. Starting from
    # the constructor default (80) avoids baking the fine-tuned class count into
    # the fresh model init too early. This matters for YOLO9-t where the class
    # branch width depends on COCO-vs-custom ``nc`` during construction.
    if nb_classes is None:
        if has_metadata:
            nb_classes = 80
        else:
            nb_classes = matched_cls.detect_nb_classes(weights_dict)
            if nb_classes is None:
                nb_classes = 80

    checkpoint_task = (
        state_dict.get("task")
        if isinstance(state_dict, dict) and isinstance(state_dict.get("task"), str)
        else None
    )
    if checkpoint_task is None and matched_cls.FAMILY == "rfdetr":
        if any(k.startswith("segmentation_head") for k in weights_dict):
            checkpoint_task = "segment"

    filename_task = matched_cls.detect_task_from_filename(Path(model_path).name)
    resolved_task = resolve_task(
        explicit_task=task,
        checkpoint_task=checkpoint_task,
        filename_task=filename_task,
        default_task=matched_cls.DEFAULT_TASK,
        supported_tasks=matched_cls.SUPPORTED_TASKS,
    )

    if matched_cls.FAMILY == "rfdetr":
        # RF-DETR always needs the path (handles its own loading internally)
        model = matched_cls(
            model_path=model_path,
            size=size,
            nb_classes=nb_classes,
            device=device,
            task=resolved_task,
        )
    elif has_metadata:
        # Our trainer checkpoint — pass path for metadata handling
        model = matched_cls(
            model_path=model_path,
            size=size,
            nb_classes=nb_classes,
            device=device,
            task=resolved_task,
            **(
                {"reg_max": reg_max}
                if matched_cls.FAMILY in ("yolo9", "yolo9_e2e")
                else {}
            ),
        )
    else:
        # Pretrained checkpoint — pass extracted state dict
        model = matched_cls(
            model_path=weights_dict,
            size=size,
            nb_classes=nb_classes,
            device=device,
            task=resolved_task,
            **(
                {"reg_max": reg_max}
                if matched_cls.FAMILY in ("yolo9", "yolo9_e2e")
                else {}
            ),
        )

    model.model_path = model_path
    return model


__all__ = [
    "LibreYOLO",
    "LibreYOLOX",
    "LibreYOLO9",
    "LibreYOLO9E2E",
    "LibreYOLONAS",
    "LibreDFINE",
    "LibreDEIM",
    "LibreDEIMv2",
    "LibreECDET",
    "LibrePicoDet",
    "LibreYOLORTDETR",
    "try_ensure_rfdetr",
]
