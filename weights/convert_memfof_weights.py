"""Convert official MEMFOF checkpoints into LibreYOLO format.

The authors publish MEMFOF on Hugging Face as ``PyTorchModelHubMixin`` repos
(``egorchistov/optical-flow-MEMFOF-*``: ``model.safetensors`` + ``config.json``).
The architecture is vendored unchanged under ``libreyolo/models/memfof/_vendor``
(BSD-3-Clause), so the state-dict keys already match and this script only
wraps them in the LibreYOLO v1.0 checkpoint schema (task ``flow``, ``nc=1``,
``names={0: "flow"}``); the load is 0-missing / 0-unexpected.

Variants (LibreYOLO filename on the left, upstream repo on the right)::

    LibreMEMFOFm-flow.pt           egorchistov/optical-flow-MEMFOF-Tartan-T-TSKH  (canonical)
    LibreMEMFOFm-flow-tartan.pt    egorchistov/optical-flow-MEMFOF-Tartan
    LibreMEMFOFm-flow-tartan-t.pt  egorchistov/optical-flow-MEMFOF-Tartan-T
    LibreMEMFOFm-flow-kitti.pt     egorchistov/optical-flow-MEMFOF-Tartan-T-TSKH-kitti
    LibreMEMFOFm-flow-sintel.pt    egorchistov/optical-flow-MEMFOF-Tartan-T-TSKH-sintel
    LibreMEMFOFm-flow-spring.pt    egorchistov/optical-flow-MEMFOF-Tartan-T-TSKH-spring

Usage::

    # from a local file or snapshot directory
    python weights/convert_memfof_weights.py path/to/model.safetensors \
        weights/LibreMEMFOFm-flow.pt --variant tskh
    # or fetch the upstream repo (needs huggingface_hub)
    python weights/convert_memfof_weights.py --from-hub --variant tskh \
        weights/LibreMEMFOFm-flow.pt

The upstream weights carry no separate licence statement; LibreYOLO treats
them as covered by the code repository's BSD-3-Clause licence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from _conversion_utils import (
    add_repo_root_to_path,
    extract_state_dict,
    load_checkpoint,
    save_checkpoint,
)

UPSTREAM_REPOS = {
    "tskh": "egorchistov/optical-flow-MEMFOF-Tartan-T-TSKH",
    "tartan": "egorchistov/optical-flow-MEMFOF-Tartan",
    "tartan-t": "egorchistov/optical-flow-MEMFOF-Tartan-T",
    "kitti": "egorchistov/optical-flow-MEMFOF-Tartan-T-TSKH-kitti",
    "sintel": "egorchistov/optical-flow-MEMFOF-Tartan-T-TSKH-sintel",
    "spring": "egorchistov/optical-flow-MEMFOF-Tartan-T-TSKH-spring",
}
CANONICAL_VARIANT = "tskh"
# The network config every published checkpoint uses (upstream defaults).
_EXPECTED_CONFIG = {
    "backbone": "resnet34",
    "dim": 512,
    "corr_levels": 4,
    "corr_radius": 4,
    "num_blocks": 2,
}
_IMGSZ = 1920


def output_filename(variant: str) -> str:
    """Canonical LibreYOLO filename for a variant (TSKH is the bare name)."""
    if variant not in UPSTREAM_REPOS:
        raise ValueError(f"unknown MEMFOF variant {variant!r}; choose from {sorted(UPSTREAM_REPOS)}")
    suffix = "" if variant == CANONICAL_VARIANT else f"-{variant}"
    return f"LibreMEMFOFm-flow{suffix}.pt"


def _load_state_dict(path: Path) -> dict:
    if path.is_dir():
        config = path / "config.json"
        if config.is_file():
            _check_config(json.loads(config.read_text()))
        path = path / "model.safetensors"
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(path))
    return extract_state_dict(load_checkpoint(path))


def _check_config(config: dict) -> None:
    for key, expected in _EXPECTED_CONFIG.items():
        if key in config and config[key] != expected:
            raise ValueError(
                f"MEMFOF config {key}={config[key]!r} differs from the vendored "
                f"architecture ({expected!r}); refusing to convert."
            )


def fetch_from_hub(variant: str, revision: str | None = None) -> tuple[Path, str]:
    """Download an upstream snapshot; return its directory and resolved commit."""
    from huggingface_hub import HfApi, snapshot_download

    repo = UPSTREAM_REPOS[variant]
    sha = HfApi().model_info(repo, revision=revision).sha
    local = snapshot_download(repo, revision=sha, allow_patterns=["*.json", "*.safetensors"])
    return Path(local), sha


def convert_state_dict(state_dict: dict, *, variant: str, source: str | None = None,
                       source_revision: str | None = None) -> dict:
    """Wrap an upstream MEMFOF state dict in LibreYOLO v1.0 metadata."""
    add_repo_root_to_path()
    from libreyolo.models.memfof.model import LibreMEMFOF
    from libreyolo.utils.serialization import wrap_libreyolo_checkpoint

    state_dict = {k: v for k, v in state_dict.items() if isinstance(v, torch.Tensor)}
    if not LibreMEMFOF.can_load(state_dict):
        raise ValueError(
            "This does not look like a MEMFOF checkpoint (expected a 9-channel "
            "'cnet.resnet.conv1.weight' plus 'att.to_qk.weight')."
        )
    if variant not in UPSTREAM_REPOS:
        raise ValueError(f"unknown MEMFOF variant {variant!r}")
    extra = {"variant": variant}
    if source:
        extra["source"] = source
    if source_revision:
        extra["source_revision"] = source_revision
    return wrap_libreyolo_checkpoint(
        state_dict,
        model_family="memfof",
        size="m",
        task="flow",
        nc=1,
        names={0: "flow"},
        imgsz=_IMGSZ,
        **extra,
    )


def check_strict_load(checkpoint: dict) -> None:
    """Load into a fresh vendored network; fail on any missing or unexpected key."""
    add_repo_root_to_path()
    from libreyolo.models.memfof._vendor import MEMFOF

    result = MEMFOF().load_state_dict(checkpoint["model"], strict=False)
    if result.missing_keys or result.unexpected_keys:
        raise ValueError(
            f"key mismatch: missing={result.missing_keys[:5]} "
            f"unexpected={result.unexpected_keys[:5]}"
        )


def convert_weights(input_path: str | None, output_path: str, *, variant: str,
                    from_hub: bool = False, revision: str | None = None) -> dict:
    source = UPSTREAM_REPOS[variant]
    source_revision = revision
    if from_hub:
        local, source_revision = fetch_from_hub(variant, revision)
    else:
        if input_path is None:
            raise ValueError("pass an input path or --from-hub")
        local = Path(input_path)
    state_dict = _load_state_dict(local)
    print(f"Loaded {len(state_dict)} tensors from {local}")
    checkpoint = convert_state_dict(
        state_dict, variant=variant, source=source, source_revision=source_revision
    )
    check_strict_load(checkpoint)
    out = Path(output_path)
    tmp = out.with_suffix(".tmp")
    save_checkpoint(checkpoint, tmp)
    tmp.replace(out)
    print(f"Saved {out} (variant={variant}, source={source}@{source_revision})")
    return checkpoint


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert MEMFOF weights to LibreYOLO format")
    parser.add_argument("input", nargs="?", help="model.safetensors, a snapshot dir, or a .pth/.ckpt")
    parser.add_argument("output", nargs="?", help="Output .pt (default: canonical name for the variant)")
    parser.add_argument("--variant", default=CANONICAL_VARIANT, choices=sorted(UPSTREAM_REPOS))
    parser.add_argument("--from-hub", action="store_true", help="Download the upstream repo first")
    parser.add_argument("--revision", default=None, help="Upstream Hugging Face revision")
    args = parser.parse_args()
    if args.from_hub and args.output is None and args.input is not None:
        args.input, args.output = None, args.input
    output = args.output or output_filename(args.variant)
    convert_weights(args.input, output, variant=args.variant,
                    from_hub=args.from_hub, revision=args.revision)
