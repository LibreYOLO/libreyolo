"""Compare LibreQuickSRNet against the pinned BSD upstream implementation.

This script imports ``blocks.py`` and ``models.py`` directly from a local
checkout of ``quic/aimet-model-zoo``. It does not copy upstream source into
LibreYOLO.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import types
from pathlib import Path

import torch

from _conversion_utils import add_repo_root_to_path, load_checkpoint


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {name} from {path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_upstream_model(source_dir: str | Path, checkpoint_path: str | Path):
    """Load official QuickSRNet Medium 2x from a pinned upstream checkout."""

    source_dir = Path(source_dir)
    blocks_path = source_dir / "blocks.py"
    models_path = source_dir / "models.py"
    if not blocks_path.is_file() or not models_path.is_file():
        raise FileNotFoundError(
            "source_dir must contain the upstream QuickSRNet model/blocks.py "
            "and model/models.py files."
        )

    package_name = "_libreyolo_quicksrnet_upstream"
    package = types.ModuleType(package_name)
    package.__path__ = [str(source_dir)]
    sys.modules[package_name] = package
    _load_module(f"{package_name}.blocks", blocks_path)
    models = _load_module(f"{package_name}.models", models_path)

    upstream = models.QuickSRNetMedium(scaling_factor=2)
    checkpoint = load_checkpoint(checkpoint_path)
    upstream.load_state_dict(checkpoint["state_dict"], strict=True)
    return upstream.eval()


def compare(source_dir: str, checkpoint_path: str) -> float:
    """Return the maximum absolute tensor difference on a deterministic input."""

    add_repo_root_to_path()
    from libreyolo.models.quicksrnet import LibreQuickSRNet

    checkpoint = load_checkpoint(checkpoint_path)
    upstream = load_upstream_model(source_dir, checkpoint_path)
    native = LibreQuickSRNet(model_path=None, size="m2", device="cpu")
    native.model.load_state_dict(checkpoint["state_dict"], strict=True)
    native.model.eval()

    generator = torch.Generator().manual_seed(2026)
    image = torch.rand(1, 3, 37, 53, generator=generator)
    with torch.inference_mode():
        upstream_output = upstream(image)
        native_output = native.model(image)
    return float((upstream_output - native_output).abs().max().item())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify QuickSRNet tensor parity")
    parser.add_argument(
        "source_dir",
        help="Pinned aimet-model-zoo aimet_zoo_torch/quicksrnet/model directory",
    )
    parser.add_argument("checkpoint", help="Official Medium 2x checkpoint")
    arguments = parser.parse_args()
    maximum_difference = compare(arguments.source_dir, arguments.checkpoint)
    print(f"max_abs_diff={maximum_difference:.9g}")
    if maximum_difference != 0.0:
        raise SystemExit(1)
