"""
Model-agnostic feature extraction via forward hooks.

Registers hooks on arbitrary nn.Module instances by dotted path string,
captures their output tensors during forward passes, and provides a clean
interface for retrieval and cleanup.

Inspired by torchdistill's ForwardHookManager (PyTorch Ecosystem).
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import List

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _resolve_module(model: nn.Module, path: str) -> nn.Module:
    """Resolve a dotted path string to an nn.Module.

    Args:
        model: Root module to start from.
        path: Dotted path like ``"neck.elan_up2"`` or ``"backbone.C3_p3"``.

    Returns:
        The nn.Module at the given path.

    Raises:
        AttributeError: If any segment of the path doesn't exist.
    """
    parts = path.split(".")
    current = model
    for part in parts:
        if hasattr(current, part):
            current = getattr(current, part)
        elif part.isdigit() and isinstance(current, (nn.ModuleList, nn.Sequential)):
            current = current[int(part)]
        else:
            raise AttributeError(
                f"Module '{type(model).__name__}' has no submodule at path '{path}' "
                f"(failed at segment '{part}'). "
                f"Available: {[n for n, _ in current.named_children()]}"
            )
    return current


class FeatureHookManager:
    """Capture intermediate feature maps from any model via forward hooks.

    Usage::

        manager = FeatureHookManager(model, ["neck.elan_up2", "neck.elan_down1"])
        output = model(input_tensor)  # hooks fire automatically
        features = manager.get_features()  # OrderedDict of path -> tensor
        manager.clear()  # reset for next batch

    The manager does not modify the model's forward method or parameters.
    """

    def __init__(self, model: nn.Module, tap_points: List[str]):
        self._features: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._hooks: List[torch.utils.hooks.RemovableHook] = []
        self._tap_points = list(tap_points)

        for path in tap_points:
            module = _resolve_module(model, path)
            hook = module.register_forward_hook(self._make_hook(path))
            self._hooks.append(hook)
            logger.debug(f"Registered distillation hook on '{path}' ({type(module).__name__})")

    def _make_hook(self, name: str):
        """Create a hook closure that stores the output under *name*."""
        def hook_fn(_module, _input, output):
            if isinstance(output, (tuple, list)):
                output = output[0]
            self._features[name] = output
        return hook_fn

    def get_features(self) -> OrderedDict[str, torch.Tensor]:
        """Return captured features as an ordered dict (path -> tensor)."""
        return self._features

    def get_feature_list(self) -> List[torch.Tensor]:
        """Return captured features as a list, in declared tap_points order."""
        result = []
        for path in self._tap_points:
            if path in self._features:
                result.append(self._features[path])
        return result

    def clear(self):
        """Clear captured features. Call after each training step."""
        self._features.clear()

    def remove(self):
        """Remove all hooks. Call when distillation is done."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._features.clear()

    @property
    def tap_points(self) -> List[str]:
        return list(self._tap_points)

    def __del__(self):
        self.remove()

    def __repr__(self) -> str:
        return (
            f"FeatureHookManager(tap_points={self._tap_points}, "
            f"captured={list(self._features.keys())})"
        )
