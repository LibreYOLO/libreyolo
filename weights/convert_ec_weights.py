"""Convert upstream EdgeCrafter EC / ECPose / ECSeg COCO weights into
LibreYOLO format.

Upstream releases ship as ``{"model": state_dict}``. LibreYOLO checkpoints add
metadata (``model_family``, ``task``, ``supported_tasks``, ``default_task``,
``nc``, ``size``, ``names``) so the unified ``LibreYOLO()`` factory can route
without filename heuristics.

EC, ECPose, and ECSeg module names already match the LibreEC port
byte-for-byte, so this is a metadata wrap — no key remapping required.

Usage:
    python weights/convert_ec_weights.py downloads/ec_weights/ecdet_s.pth weights/LibreECs.pt --size s --task detect
    python weights/convert_ec_weights.py weights/ecpose_s.pth weights/LibreECs-pose.pt --size s --task pose
    python weights/convert_ec_weights.py weights/ecseg_s.pth weights/LibreECs-seg.pt --size s --task segment

Add ``--verify`` to load the converted weights into a LibreEC wrapper and
run a smoke forward pass.
"""

from __future__ import annotations

import argparse

import torch

from _conversion_utils import (
    add_repo_root_to_path,
    extract_state_dict,
    load_checkpoint,
    save_checkpoint,
    wrap_libreyolo_checkpoint,
)


_SUPPORTED_TASKS = ("detect", "pose", "segment")
_DEFAULT_TASK = "detect"

# Per-task class-count + class-name overrides.
_POSE_NAMES = {0: "person"}


def convert_weights(
    input_path: str, output_path: str, size: str, task: str = "detect", nc: int = 80,
) -> dict:
    print(f"Loading upstream weights from {input_path}")
    raw = load_checkpoint(input_path)
    state_dict = extract_state_dict(raw)
    if not isinstance(state_dict, dict):
        raise TypeError(f"Could not extract state dict from {input_path}")
    print(f"Found {len(state_dict)} parameter entries")

    if task == "pose":
        nc = 1
        names: dict[int, str] | None = _POSE_NAMES
    else:
        names = None  # built from nc by wrap_libreyolo_checkpoint

    libreyolo_ckpt = wrap_libreyolo_checkpoint(
        state_dict,
        model_family="ec",
        size=size,
        nc=nc,
        names=names,
        task=task,
        supported_tasks=_SUPPORTED_TASKS,
        default_task=_DEFAULT_TASK,
    )
    out = save_checkpoint(libreyolo_ckpt, output_path)
    print(f"Saved LibreYOLO-format checkpoint to {out}")
    return libreyolo_ckpt


def verify_conversion(converted_path: str, size: str, task: str) -> bool:
    add_repo_root_to_path()
    from libreyolo.models.ec.model import LibreEC

    print(f"\nLoading converted weights into LibreEC-{size} task={task}...")
    m = LibreEC(converted_path, size=size, device="cpu", task=task)
    print(f"  family={m.FAMILY} size={m.size} task={m.task} nc={m.nb_classes}")

    m.model.eval()
    with torch.no_grad():
        out = m.model(torch.zeros(1, 3, 640, 640))

    if task == "detect":
        assert "pred_logits" in out and "pred_boxes" in out
        assert out["pred_logits"].shape == (1, 300, 80)
        assert out["pred_boxes"].shape == (1, 300, 4)
        print("  detect forward pass OK — logits (1,300,80), boxes (1,300,4)")
    elif task == "pose":
        assert "pred_logits" in out and "pred_keypoints" in out
        assert out["pred_logits"].shape == (1, 60, 2)
        assert out["pred_keypoints"].shape == (1, 60, 34)
        print("  pose forward pass OK — logits (1,60,2), keypoints (1,60,34)")
    elif task == "segment":
        assert "pred_logits" in out and "pred_boxes" in out and "pred_masks" in out
        assert out["pred_logits"].shape == (1, 300, 80)
        assert out["pred_boxes"].shape == (1, 300, 4)
        assert out["pred_masks"].shape == (1, 300, 160, 160)
        print("  segment forward pass OK — masks (1,300,160,160)")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert EC weights to LibreYOLO format")
    parser.add_argument("input", help="Upstream EdgeCrafter checkpoint (.pth)")
    parser.add_argument("output", help="Output LibreYOLO checkpoint (.pt)")
    parser.add_argument("--size", required=True, choices=["s", "m", "l", "x"])
    parser.add_argument("--task", default="detect", choices=_SUPPORTED_TASKS)
    parser.add_argument("--nc", type=int, default=80)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    convert_weights(args.input, args.output, args.size, args.task, args.nc)
    if args.verify:
        verify_conversion(args.output, args.size, args.task)
