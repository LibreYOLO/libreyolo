"""Convert upstream DEIMv2 weights into LibreYOLO format."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from _conversion_utils import (
    add_repo_root_to_path,
    extract_state_dict,
    load_checkpoint,
    save_checkpoint,
    wrap_libreyolo_checkpoint,
)


def _load_safetensors_state_dict(input_path: str, size: str, nc: int) -> dict:
    add_repo_root_to_path()
    try:
        from safetensors.torch import load_model as load_safetensors_model
    except ImportError as e:
        raise ImportError(
            "Converting DEIMv2 safetensors requires safetensors. "
            "Install with: pip install safetensors"
        ) from e

    from libreyolo.models.deimv2.nn import LibreDEIMv2Model

    model = LibreDEIMv2Model(config=size, nb_classes=nc)
    missing, unexpected = load_safetensors_model(
        model,
        input_path,
        strict=True,
        device="cpu",
    )
    if missing or unexpected:
        raise RuntimeError(
            "Failed to load DEIMv2 safetensors exactly: "
            f"missing={sorted(missing)[:10]}, unexpected={sorted(unexpected)[:10]}"
        )
    return model.state_dict()


def convert_weights(input_path: str, output_path: str, size: str, nc: int = 80) -> dict:
    print(f"Loading upstream weights from {input_path}")
    if Path(input_path).suffix == ".safetensors":
        state_dict = _load_safetensors_state_dict(input_path, size, nc)
    else:
        raw = load_checkpoint(input_path)
        state_dict = extract_state_dict(raw)
    print(f"Found {len(state_dict)} parameter entries")

    libreyolo_ckpt = wrap_libreyolo_checkpoint(
        state_dict,
        model_family="deimv2",
        size=size,
        nc=nc,
    )

    save_checkpoint(libreyolo_ckpt, output_path)
    print(f"Saved LibreYOLO-format checkpoint to {output_path}")
    return libreyolo_ckpt


def verify_conversion(converted_path: str, size: str) -> bool:
    add_repo_root_to_path()
    from libreyolo import LibreDEIMv2

    print(f"\nLoading converted weights into LibreDEIMv2-{size}...")
    model = LibreDEIMv2(converted_path, size=size, device="cpu")
    model.model.eval()
    with torch.no_grad():
        out = model.model(torch.zeros(1, 3, model.input_size, model.input_size))
    assert "pred_logits" in out and "pred_boxes" in out
    assert out["pred_logits"].shape[0] == 1
    assert out["pred_logits"].shape[-1] == model.nb_classes
    assert out["pred_boxes"].shape[-1] == 4
    print("  forward pass OK")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert DEIMv2 weights to LibreYOLO format"
    )
    parser.add_argument(
        "input", help="Upstream DEIMv2 checkpoint (.pth/.bin/.safetensors)"
    )
    parser.add_argument("output", help="Output LibreYOLO checkpoint (.pt)")
    parser.add_argument(
        "--size",
        required=True,
        choices=["atto", "femto", "pico", "n", "s", "m", "l", "x"],
        help="DEIMv2 size",
    )
    parser.add_argument(
        "--nc", type=int, default=80, help="Number of classes (default: 80)"
    )
    parser.add_argument(
        "--verify", action="store_true", help="Verify round-trip after conversion"
    )
    args = parser.parse_args()

    convert_weights(args.input, args.output, args.size, args.nc)
    if args.verify:
        verify_conversion(args.output, args.size)
