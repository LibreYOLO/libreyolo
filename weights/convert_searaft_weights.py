"""Convert official SEA-RAFT weights into LibreYOLO format.

SEA-RAFT (https://github.com/princeton-vl/SEA-RAFT, BSD-3-Clause) publishes its
checkpoints on Hugging Face (``MemorySlices/<name>/model.safetensors``) and in a
Google Drive model zoo (``<name>.pth``). Both are bare state dicts whose keys
already match the vendored model under ``libreyolo/models/searaft/_vendor``, so
this script only wraps them in the LibreYOLO v1.0 checkpoint schema; the load is
0-missing / 0-unexpected.

Canonical LibreYOLO checkpoints (all Tartan-C-T-TSKH-spring540x960):

    S  <- Tartan-C-T-TSKH-spring540x960-S   -> LibreSEARAFTs-flow.pt (4 iterations)
    M  <- Tartan-C-T-TSKH-spring540x960-M   -> LibreSEARAFTm-flow.pt (4 iterations)
    L  <- Tartan-C-T-TSKH-spring540x960-M   -> LibreSEARAFTl-flow.pt (12 iterations)

Upstream defines SEA-RAFT(L) as the M weights run with 12 iterations, so ``--size l``
takes an M checkpoint.

Usage::

    python weights/convert_searaft_weights.py model.safetensors \
        weights/LibreSEARAFTm-flow.pt --size m --verify
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from _conversion_utils import (
    add_repo_root_to_path,
    extract_state_dict,
    load_checkpoint,
    save_checkpoint,
)

_IMGSZ = 960
_SIZES = ("s", "m", "l")
_ENCODER_SIZE = {"s": "s", "m": "m", "l": "m"}  # size -> encoder the weights must have


def load_upstream_state_dict(path: str | Path) -> dict:
    path = Path(path)
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return dict(load_file(str(path)))
    return dict(extract_state_dict(load_checkpoint(path)))


def convert_weights(
    input_path: str,
    output_path: str,
    *,
    size: str | None = None,
    source: str | None = None,
) -> dict:
    add_repo_root_to_path()
    from libreyolo.models.searaft.model import LibreSEARAFT
    from libreyolo.utils.serialization import wrap_libreyolo_checkpoint

    print(f"Loading upstream SEA-RAFT weights from {input_path}")
    state_dict = load_upstream_state_dict(input_path)
    # Upstream DataParallel checkpoints prefix every key with ``module.``.
    if state_dict and all(k.startswith("module.") for k in state_dict):
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
    print(f"Found {len(state_dict)} tensors")

    if not LibreSEARAFT.can_load(state_dict):
        raise ValueError(
            f"{input_path} does not look like a SEA-RAFT checkpoint (expected a "
            "6-channel cnet.conv1, init_conv and a 6-channel flow_head)."
        )
    state_dict = LibreSEARAFT.restore_shared_keys(state_dict)
    encoder = LibreSEARAFT.detect_size(state_dict)
    if size is None:
        size = encoder
        print(f"Auto-detected size: {size}")
    if size not in _SIZES:
        raise ValueError(f"size must be one of {_SIZES}, got {size!r}")
    if encoder != _ENCODER_SIZE[size]:
        raise ValueError(
            f"--size {size} needs a SEA-RAFT-{_ENCODER_SIZE[size].upper()} "
            f"checkpoint, but {input_path} has the {str(encoder).upper()} encoder."
        )

    # Strict load into a fresh model: proves the keys and shapes match.
    model = LibreSEARAFT(None, size=size, device="cpu")
    model.model.load_state_dict(state_dict, strict=True)
    state_dict = {k: v.detach().clone().contiguous() for k, v in model.model.state_dict().items()}

    extra = {"source": source} if source else {}
    ckpt = wrap_libreyolo_checkpoint(
        state_dict,
        model_family="searaft",
        size=size,
        task="flow",
        nc=1,
        names={0: "flow"},
        imgsz=_IMGSZ,
        **extra,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(".tmp")
    save_checkpoint(ckpt, tmp)
    tmp.replace(output_path)
    print(f"Saved LibreYOLO-format checkpoint to {output_path}")
    return ckpt


def verify_conversion(converted_path: str) -> bool:
    add_repo_root_to_path()
    from libreyolo import LibreYOLO
    from libreyolo.utils.serialization import validate_checkpoint_metadata

    ckpt = torch.load(converted_path, map_location="cpu", weights_only=True)
    problems = validate_checkpoint_metadata(ckpt, strict=True)
    if problems:
        raise ValueError(f"checkpoint metadata problems: {problems}")
    m = LibreYOLO(converted_path, device="cpu")
    print(f"  family={m.FAMILY} size={m.size} task={m.task} names={m.names}")
    with torch.no_grad():
        flow, info = m.model(torch.zeros(1, 3, 128, 160), torch.zeros(1, 3, 128, 160))
    print(f"  forward OK - flow {tuple(flow.shape)}, info {tuple(info.shape)}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert SEA-RAFT weights to LibreYOLO format")
    parser.add_argument("input", help="Upstream checkpoint (.safetensors or .pth)")
    parser.add_argument("output", help="Output LibreYOLO checkpoint (.pt)")
    parser.add_argument("--size", default=None, choices=_SIZES,
                        help="Size code (default: from the encoder depth; l needs an M checkpoint)")
    parser.add_argument("--source", default=None, help="Optional provenance string stored in the checkpoint")
    parser.add_argument("--verify", action="store_true", help="Validate metadata and load after conversion")
    args = parser.parse_args()

    convert_weights(args.input, args.output, size=args.size, source=args.source)
    if args.verify:
        verify_conversion(args.output)
