"""Convert official WAFT checkpoints into LibreYOLO format.

WAFT (https://github.com/princeton-vl/WAFT, BSD-3-Clause) releases bare state
dicts on its Google Drive model zoo. The inference graph is vendored under
``libreyolo/models/waft/_vendor`` with upstream parameter names, so conversion
only wraps the state dict in the LibreYOLO v1.0 checkpoint schema; the load is
0-missing / 0-unexpected.

Size mapping (by parameter count, zero-shot ``tar-c-t`` / ``zero-shot`` files):

    t  a2/dinov3/zero-shot.pth  ->  LibreWAFTt-flow-dinov3.pt   (56.5 M)
    s  a2/dav2/zero-shot.pth    ->  LibreWAFTs-flow-dav2.pt     (56.9 M)
    m  a1/tar-c-t.pth           ->  LibreWAFTm-flow.pt          (64.2 M)
    l  a2/twins/zero-shot.pth   ->  LibreWAFTl-flow-twins.pt    (136.0 M)

Usage::

    python weights/convert_waft_weights.py a1/tar-c-t.pth weights/LibreWAFTm-flow.pt

Size is detected from the encoder keys (``--size`` overrides). ``--verify``
loads the result through ``LibreYOLO`` with ``strict=True``.
"""

from __future__ import annotations

import argparse

from _conversion_utils import (
    add_repo_root_to_path,
    extract_state_dict,
    load_checkpoint,
    save_checkpoint,
)

_IMGSZ = 640
_SIZE_TO_VARIANT = {"t": "dinov3", "s": "dav2", "m": None, "l": "twins"}


def canonical_filename(size: str) -> str:
    variant = _SIZE_TO_VARIANT[size]
    return f"LibreWAFT{size}-flow" + (f"-{variant}" if variant else "") + ".pt"


def convert_state_dict(state_dict: dict, size: str | None = None) -> dict:
    add_repo_root_to_path()
    from libreyolo import __version__
    from libreyolo.models.waft.model import LibreWAFT
    from libreyolo.utils.serialization import wrap_libreyolo_checkpoint

    if not LibreWAFT.can_load(state_dict):
        raise ValueError(
            "This does not look like a WAFT checkpoint (expected warp_linear, "
            "refine_transform, refine_net and a known encoder)."
        )
    detected = LibreWAFT.detect_size(state_dict)
    if size is not None and size != detected:
        raise ValueError(f"--size {size} does not match the checkpoint encoder (size {detected}).")
    size = detected
    extra = {"variant": _SIZE_TO_VARIANT[size]} if _SIZE_TO_VARIANT[size] else {}
    return wrap_libreyolo_checkpoint(
        dict(state_dict),
        model_family="waft",
        size=size,
        task="flow",
        nc=1,
        names={0: "flow"},
        imgsz=_IMGSZ,
        libreyolo_version=__version__,
        **extra,
    )


def convert_weights(input_path: str, output_path: str, *, size: str | None = None) -> dict:
    print(f"Loading upstream WAFT weights from {input_path}")
    state_dict = extract_state_dict(load_checkpoint(input_path))
    print(f"Found {len(state_dict)} tensors")
    ckpt = convert_state_dict(state_dict, size)
    expected = canonical_filename(ckpt["size"])
    if not output_path.endswith(expected):
        print(f"Note: the canonical filename for size {ckpt['size']} is {expected}")
    save_checkpoint(ckpt, output_path)
    print(f"Saved LibreYOLO-format checkpoint to {output_path}")
    return ckpt


def verify_conversion(converted_path: str) -> None:
    add_repo_root_to_path()
    import torch

    from libreyolo import LibreYOLO

    m = LibreYOLO(converted_path, device="cpu")
    ckpt = torch.load(converted_path, map_location="cpu", weights_only=False)
    m.model.load_state_dict(ckpt["model"], strict=True)
    print(f"family={m.FAMILY} size={m.size} task={m.task} names={m.names}: strict load ok")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="upstream WAFT .pth")
    parser.add_argument("output", help="output LibreYOLO .pt")
    parser.add_argument("--size", choices=sorted(_SIZE_TO_VARIANT), default=None)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    convert_weights(args.input, args.output, size=args.size)
    if args.verify:
        verify_conversion(args.output)


if __name__ == "__main__":
    main()
