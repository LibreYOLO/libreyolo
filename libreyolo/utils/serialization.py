"""Helpers for safe vs trusted torch checkpoint loading."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import torch


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
