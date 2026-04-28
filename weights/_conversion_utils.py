"""Shared helpers for weight-conversion scripts.

These helpers keep the family scripts small without forcing them into one
conversion format. They intentionally cover only the repeated plumbing:
repo-root imports, checkpoint loading/unwrapping, metadata packaging, and
filesystem writes.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch


_REPO_ROOT = Path(__file__).resolve().parent.parent


def repo_root() -> Path:
    """Return the repository root for local script execution."""
    return _REPO_ROOT


def add_repo_root_to_path() -> Path:
    """Ensure local ``libreyolo`` imports resolve when running from scripts."""
    root = repo_root()
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


def load_checkpoint(path: str | Path) -> Any:
    """Load a trusted checkpoint in full metadata mode."""
    return torch.load(path, map_location="cpu", weights_only=False)


def _materialize_state_dict(candidate: Any) -> Any:
    if isinstance(candidate, dict):
        return candidate

    state_dict = getattr(candidate, "state_dict", None)
    if callable(state_dict):
        return state_dict()

    return candidate


def extract_state_dict(
    checkpoint: Any,
    *,
    state_dict_keys: tuple[str, ...] = ("model", "state_dict"),
    prefer_ema: bool = True,
) -> Any:
    """Extract the model state dict from a few common checkpoint layouts."""
    if not isinstance(checkpoint, dict):
        return _materialize_state_dict(checkpoint)

    if prefer_ema and "ema" in checkpoint:
        ema = checkpoint["ema"]
        module = (
            ema.get("module")
            if isinstance(ema, dict)
            else getattr(ema, "module", None)
        )
        if module is not None:
            return _materialize_state_dict(module)

    for key in state_dict_keys:
        if key in checkpoint:
            return _materialize_state_dict(checkpoint[key])

    return checkpoint


def strip_state_dict_prefix(
    state_dict: dict[str, torch.Tensor],
    prefix: str,
) -> dict[str, torch.Tensor]:
    """Strip one leading prefix from matching state_dict keys."""
    if not prefix:
        return dict(state_dict)

    return {
        key[len(prefix):] if key.startswith(prefix) else key: value
        for key, value in state_dict.items()
    }


def build_class_names(nc: int) -> dict[int, str]:
    """Return COCO names for 80 classes, else a generic indexed mapping."""
    if nc == 80:
        add_repo_root_to_path()
        from libreyolo.utils.general import COCO_CLASSES

        return {index: name for index, name in enumerate(COCO_CLASSES)}

    return {index: f"class_{index}" for index in range(nc)}


def wrap_libreyolo_checkpoint(
    state_dict: dict[str, torch.Tensor],
    *,
    model_family: str,
    size: str,
    nc: int,
    names: dict[int, str] | None = None,
) -> dict[str, Any]:
    """Build the standard metadata-wrapped LibreYOLO checkpoint format."""
    if names is None:
        names = build_class_names(nc)

    return {
        "model": state_dict,
        "model_family": model_family,
        "size": size,
        "nc": nc,
        "names": names,
    }


def save_checkpoint(checkpoint: Any, output_path: str | Path) -> Path:
    """Create parents and save a checkpoint to disk."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)
    return output_path
