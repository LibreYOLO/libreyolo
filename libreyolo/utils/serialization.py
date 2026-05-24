"""Helpers for safe torch loading and LibreYOLO checkpoint metadata."""

from __future__ import annotations

import inspect
import warnings
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import torch

from ..tasks import normalize_task


SCHEMA_VERSION = "1.0"

REQUIRED_CHECKPOINT_METADATA_KEYS = (
    "model",
    "schema_version",
    "libreyolo_version",
    "model_family",
    "size",
    "task",
    "nc",
    "names",
    "imgsz",
)


class CheckpointMetadataError(ValueError):
    """Raised when a checkpoint does not satisfy the LibreYOLO metadata schema."""


def get_libreyolo_version() -> str:
    """Return the installed LibreYOLO version, with an editable-install fallback."""
    try:
        return version("libreyolo")
    except PackageNotFoundError:
        return "0.0.0.dev0"


def _supports_weights_only() -> bool:
    """Return whether the installed torch.load supports ``weights_only``."""
    try:
        signature = inspect.signature(torch.load)
    except (TypeError, ValueError):
        return False
    return "weights_only" in signature.parameters


def _torch_load(
    path: str | Path,
    *,
    map_location: Any,
    weights_only: bool,
    context: str,
):
    load_kwargs = {"map_location": map_location}

    if _supports_weights_only():
        load_kwargs["weights_only"] = weights_only
        return torch.load(path, **load_kwargs)

    if weights_only:
        raise RuntimeError(
            f"Safe loading for {context} requires a PyTorch build that supports "
            "torch.load(..., weights_only=...). Upgrade PyTorch or use a trusted "
            "checkpoint workflow."
        )

    return torch.load(path, **load_kwargs)


def load_untrusted_torch_file(
    path: str | Path,
    *,
    map_location: Any = "cpu",
    context: str = "model weights",
):
    """Safely load a user-supplied torch file."""
    return _torch_load(
        path,
        map_location=map_location,
        weights_only=True,
        context=context,
    )


def load_trusted_torch_file(
    path: str | Path,
    *,
    map_location: Any = "cpu",
    context: str = "trusted checkpoint",
):
    """Load a trusted internal torch checkpoint with full metadata."""
    return _torch_load(
        path,
        map_location=map_location,
        weights_only=False,
        context=context,
    )


def build_class_names(nc: int) -> dict[int, str]:
    """Return COCO names for 80 classes, else a generic indexed mapping."""
    if nc == 80:
        from .general import COCO_CLASSES

        return {index: name for index, name in enumerate(COCO_CLASSES)}

    return {index: f"class_{index}" for index in range(nc)}


def normalize_checkpoint_names(names: Any, nc: int) -> dict[int, str]:
    """Normalize checkpoint class names to LibreYOLO's canonical ``dict[int, str]``."""
    if isinstance(names, list):
        names = dict(enumerate(names))
    if not isinstance(names, dict):
        raise CheckpointMetadataError("names must be a dict[int, str] or list[str].")

    normalized: dict[int, str] = {}
    for key, value in names.items():
        try:
            index = int(key)
        except (TypeError, ValueError) as exc:
            raise CheckpointMetadataError(
                f"names contains a non-integer class index: {key!r}."
            ) from exc
        normalized[index] = str(value)

    expected = set(range(nc))
    extra = sorted(index for index in normalized if index not in expected)
    if extra:
        raise CheckpointMetadataError(
            "names keys must be within class indices 0..nc-1 "
            f"(nc={nc}, got out-of-range keys {extra})."
        )

    missing = sorted(expected - set(normalized))
    if missing:
        warnings.warn(
            "names is missing class indices "
            f"{missing}; padding with generic class_i labels.",
            RuntimeWarning,
            stacklevel=2,
        )

    return {index: normalized.get(index, f"class_{index}") for index in range(nc)}


def _infer_checkpoint_imgsz(
    *,
    model_family: str,
    size: str,
    task: str,
) -> int | None:
    """Infer square input size from registered family metadata when possible."""
    try:
        from ..models.base import BaseModel
    except Exception:
        return None

    for cls in BaseModel._registry:
        if cls.FAMILY != model_family:
            continue
        task_sizes = getattr(cls, "TASK_INPUT_SIZES", {}) or {}
        input_sizes = (
            task_sizes[task] if task in task_sizes else getattr(cls, "INPUT_SIZES", {})
        )
        value = input_sizes.get(size)
        if value is not None:
            return int(value)
    return None


def wrap_libreyolo_checkpoint(
    state_dict: dict[str, torch.Tensor],
    *,
    model_family: str,
    size: str,
    task: str,
    nc: int,
    names: dict[int, str] | list[str] | None = None,
    imgsz: int | None = None,
    libreyolo_version: str | None = None,
    schema_version: str = SCHEMA_VERSION,
    **extra_metadata: Any,
) -> dict[str, Any]:
    """Build a strict LibreYOLO v1.0 metadata-wrapped checkpoint."""
    normalized_task = normalize_task(task)
    if normalized_task is None:
        raise CheckpointMetadataError("task is required.")

    if names is None:
        names = build_class_names(nc)
    normalized_names = normalize_checkpoint_names(names, nc)

    if imgsz is None:
        imgsz = _infer_checkpoint_imgsz(
            model_family=model_family,
            size=size,
            task=normalized_task,
        )
    if imgsz is None:
        raise CheckpointMetadataError(
            "imgsz is required and could not be inferred from model_family/size/task."
        )

    checkpoint: dict[str, Any] = {
        "model": state_dict,
        "schema_version": schema_version,
        "libreyolo_version": libreyolo_version or get_libreyolo_version(),
        "model_family": model_family,
        "size": size,
        "task": normalized_task,
        "nc": nc,
        "names": normalized_names,
        "imgsz": int(imgsz),
    }
    # Optional fields with None are intentionally omitted from checkpoint files.
    checkpoint.update({k: v for k, v in extra_metadata.items() if v is not None})
    validate_checkpoint_metadata(checkpoint, strict=True)
    return checkpoint


def validate_checkpoint_metadata(
    checkpoint: Any,
    *,
    strict: bool = False,
) -> list[str]:
    """Validate a LibreYOLO checkpoint wrapper against metadata schema v1.0.

    This function is intentionally non-mutating. Callers that need normalized
    values should do so explicitly through the schema construction helpers.
    """
    errors: list[str] = []
    if not isinstance(checkpoint, dict):
        errors.append("checkpoint must be a dict.")
    else:
        for key in REQUIRED_CHECKPOINT_METADATA_KEYS:
            if key not in checkpoint:
                errors.append(f"missing required key: {key}")

        model = checkpoint.get("model")
        if "model" in checkpoint and not isinstance(model, dict):
            errors.append("model must be a state_dict dictionary.")

        if checkpoint.get("schema_version") != SCHEMA_VERSION:
            errors.append(
                f"schema_version must be {SCHEMA_VERSION!r}, got "
                f"{checkpoint.get('schema_version')!r}."
            )

        libreyolo_version = checkpoint.get("libreyolo_version")
        if "libreyolo_version" in checkpoint and not (
            isinstance(libreyolo_version, str) and libreyolo_version
        ):
            errors.append("libreyolo_version must be a non-empty string.")

        model_family = checkpoint.get("model_family")
        if "model_family" in checkpoint and not (
            isinstance(model_family, str) and model_family
        ):
            errors.append("model_family must be a non-empty string.")

        size = checkpoint.get("size")
        if "size" in checkpoint and not (isinstance(size, str) and size):
            errors.append("size must be a non-empty string.")

        try:
            task = normalize_task(checkpoint.get("task"))
            if task is None:
                errors.append("task is required.")
        except ValueError as exc:
            errors.append(str(exc))

        nc = checkpoint.get("nc")
        if not isinstance(nc, int) or isinstance(nc, bool) or nc <= 0:
            errors.append("nc must be a positive int.")
            nc_for_names = None
        else:
            nc_for_names = nc

        if "names" in checkpoint and nc_for_names is not None:
            try:
                normalize_checkpoint_names(
                    checkpoint["names"],
                    nc_for_names,
                )
            except CheckpointMetadataError as exc:
                errors.append(str(exc))

        imgsz = checkpoint.get("imgsz")
        if not isinstance(imgsz, int) or isinstance(imgsz, bool) or imgsz <= 0:
            errors.append("imgsz must be a positive int.")

    if strict and errors:
        raise CheckpointMetadataError("; ".join(errors))
    return errors


def warn_on_metadata_schema_version(
    metadata: Any,
    *,
    artifact: str,
    logger: Any,
) -> None:
    """Warn when exported runtime metadata is legacy or from another schema."""
    if not isinstance(metadata, dict) or not metadata:
        return

    schema_version = metadata.get("schema_version")
    if schema_version is None:
        logger.warning(
            "%s metadata has no schema_version; treating it as legacy metadata.",
            artifact,
        )
        return

    if str(schema_version) != SCHEMA_VERSION:
        logger.warning(
            "%s metadata schema_version %r differs from supported %r.",
            artifact,
            schema_version,
            SCHEMA_VERSION,
        )


def is_libreyolo_checkpoint(checkpoint: Any) -> bool:
    """Return whether a loaded object carries complete LibreYOLO v1.0 metadata."""
    return not validate_checkpoint_metadata(checkpoint, strict=False)


def unwrap_libreyolo_checkpoint(
    loaded: Any,
    *,
    strict: bool = False,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Return ``(state_dict, metadata)`` from a LibreYOLO checkpoint wrapper."""
    if strict:
        validate_checkpoint_metadata(loaded, strict=True)

    if isinstance(loaded, dict) and isinstance(loaded.get("model"), dict):
        metadata = {k: v for k, v in loaded.items() if k != "model"}
        return loaded["model"], metadata

    if strict:
        raise CheckpointMetadataError("checkpoint does not contain a model state_dict.")
    if isinstance(loaded, dict):
        return loaded, {}
    raise CheckpointMetadataError("checkpoint must be a dict.")
