"""Convert official Video Depth Anything weights into LibreYOLO format.

The released relative-depth checkpoints (``video_depth_anything_vit{s,b,l}.pth``)
are bare state dicts with the DINOv2 encoder under ``pretrained.*`` and the
temporal DPT head under ``head.*``. The native port keeps those names, so this
script only wraps the state dict in the LibreYOLO v1.0 checkpoint schema; the
learned parameters are unchanged and the load is strict.

Upstream: https://github.com/DepthAnything/Video-Depth-Anything (Apache-2.0
code). Weight licences: Small is Apache-2.0, Base and Large are CC-BY-NC-4.0.
Metric checkpoints (``metric_video_depth_anything_*``) are rejected: they share
the relative tensors' layout but predict metres.

Usage::

    python weights/convert_video_depth_anything_weights.py \
        video_depth_anything_vits.pth weights/LibreVideoDepthAnythings-depth.pt --verify
"""

from __future__ import annotations

import argparse

import torch

from _conversion_utils import (
    add_repo_root_to_path,
    extract_state_dict,
    load_checkpoint,
    save_checkpoint,
)

_IMGSZ = 518


def convert_weights(input_path: str, output_path: str, *, size: str | None = None) -> dict:
    add_repo_root_to_path()
    from libreyolo.models.video_depth_anything.model import LibreVideoDepthAnything
    from libreyolo.models.video_depth_anything.utils import check_not_metric
    from libreyolo.utils.serialization import wrap_libreyolo_checkpoint

    check_not_metric(input_path)
    print(f"Loading upstream Video Depth Anything weights from {input_path}")
    state_dict = extract_state_dict(load_checkpoint(input_path))
    if not LibreVideoDepthAnything.can_load(state_dict):
        raise ValueError(
            "This does not look like a Video Depth Anything checkpoint "
            "(expected 'pretrained.*' and 'head.motion_modules.*' keys)."
        )
    size = size or LibreVideoDepthAnything.detect_size(state_dict)
    if size is None:
        raise ValueError("Could not detect the encoder size; pass --size {s,b,l}.")
    print(f"Size: {size} ({len(state_dict)} tensors)")

    wrapped = wrap_libreyolo_checkpoint(
        state_dict,
        model_family=LibreVideoDepthAnything.FAMILY,
        size=size,
        task="depth",
        nc=1,
        names={0: "depth"},
        imgsz=_IMGSZ,
    )
    save_checkpoint(wrapped, output_path)
    print(f"Saved LibreYOLO-format checkpoint to {output_path}")
    return wrapped


def verify_conversion(converted_path: str) -> None:
    add_repo_root_to_path()
    from libreyolo import LibreYOLO

    model = LibreYOLO(converted_path, device="cpu")
    print(f"  family={model.FAMILY} size={model.size} task={model.task}")
    with torch.no_grad():
        out = model.model(torch.zeros(1, 2, 3, 70, 70))
    print(f"  forward OK - output shape {tuple(out.shape)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Video Depth Anything weights to LibreYOLO format"
    )
    parser.add_argument("input", help="Upstream checkpoint (.pth)")
    parser.add_argument("output", help="Output LibreYOLO checkpoint (.pt)")
    parser.add_argument("--size", choices=["s", "b", "l"], default=None)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    convert_weights(args.input, args.output, size=args.size)
    if args.verify:
        verify_conversion(args.output)
