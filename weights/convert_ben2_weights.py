"""Convert the published BEN2 Base checkpoint to LibreYOLO format.

The BEN2 safetensors keys already match the native port, so conversion is a
strict metadata wrap with learned parameters unchanged.

Usage::

    python weights/convert_ben2_weights.py model.safetensors \
        weights/LibreBEN2b-matte.pt --verify

BEN2 code and released Base weights are MIT:
https://github.com/PramaLLC/BEN2 and https://huggingface.co/PramaLLC/BEN2.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from _conversion_utils import add_repo_root_to_path, load_checkpoint, save_checkpoint

_IMGSZ = 1024


def _load_state_dict(input_path: str) -> dict[str, torch.Tensor]:
    if str(input_path).endswith(".safetensors"):
        from safetensors.torch import load_file

        return dict(load_file(input_path))
    raw = load_checkpoint(input_path)
    if isinstance(raw, dict):
        for key in ("model_state_dict", "state_dict", "model", "params", "net"):
            value = raw.get(key)
            if isinstance(value, dict):
                return dict(value)
        return dict(raw)
    if hasattr(raw, "state_dict"):
        return dict(raw.state_dict())
    raise TypeError(f"Unsupported checkpoint object: {type(raw)!r}")


def convert_weights(
    input_path: str,
    output_path: str,
    *,
    imgsz: int = _IMGSZ,
) -> dict:
    print(f"Loading BEN2 Base weights from {input_path}")
    state_dict = _load_state_dict(input_path)
    print(f"Found {len(state_dict)} parameter entries")

    add_repo_root_to_path()
    from libreyolo.models.ben2 import LibreBEN2
    from libreyolo.utils.serialization import wrap_libreyolo_checkpoint

    if not LibreBEN2.can_load(state_dict):
        raise ValueError("The checkpoint does not match the BEN2 Base key layout.")

    model = LibreBEN2(model_path=None, size="b", device="cpu")
    result = model.model.load_state_dict(state_dict, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            "BEN2 state dict did not load strictly: "
            f"missing={result.missing_keys[:8]}, "
            f"unexpected={result.unexpected_keys[:8]}"
        )

    checkpoint = wrap_libreyolo_checkpoint(
        model.model.state_dict(),
        model_family="ben2",
        size="b",
        task="matte",
        nc=1,
        names={0: "matte"},
        supported_tasks=("matte",),
        default_task="matte",
        imgsz=imgsz,
    )
    output = Path(output_path)
    temporary = output.with_suffix(output.suffix + ".tmp")
    save_checkpoint(checkpoint, temporary)
    temporary.replace(output)
    print(f"Saved LibreYOLO-format checkpoint to {output}")
    return checkpoint


def verify_conversion(converted_path: str) -> bool:
    add_repo_root_to_path()
    from libreyolo import LibreYOLO
    from libreyolo.utils.serialization import validate_checkpoint_metadata

    checkpoint = load_checkpoint(converted_path)
    validate_checkpoint_metadata(checkpoint, strict=True)
    model = LibreYOLO(converted_path, device="cpu")
    assert model.FAMILY == "ben2"
    assert model.size == "b" and model.task == "matte"
    print(f"Round-trip OK: family={model.FAMILY} size={model.size} task={model.task}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert BEN2 Base weights to LibreYOLO format"
    )
    parser.add_argument("input", help="Upstream BEN2 checkpoint")
    parser.add_argument("output", help="Output LibreYOLO checkpoint (.pt)")
    parser.add_argument(
        "--imgsz",
        type=int,
        default=_IMGSZ,
        help="Native input size recorded in metadata",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify metadata and factory round-trip",
    )
    args = parser.parse_args()

    convert_weights(args.input, args.output, imgsz=args.imgsz)
    if args.verify:
        verify_conversion(args.output)
