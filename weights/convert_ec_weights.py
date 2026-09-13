"""Convert upstream EdgeCrafter EC / ECPose / ECSeg COCO weights into
LibreYOLO format.

Upstream releases ship as ``{"model": state_dict}``. LibreYOLO checkpoints add
metadata (``model_family``, ``task``, ``nc``, ``size``, ``names``, ``imgsz``)
so the unified ``LibreYOLO()`` factory can route without filename heuristics.

EC, ECPose, and ECSeg module names already match the LibreEC port
byte-for-byte, so this is a metadata wrap — no key remapping required.

Usage:
    python weights/convert_ec_weights.py downloads/ec_weights/ecdet_s.pth weights/LibreECs.pt --size s --task detect
    python weights/convert_ec_weights.py weights/ecpose_s.pth weights/LibreECs-pose.pt --size s --task pose
    python weights/convert_ec_weights.py weights/ecseg_s.pth weights/LibreECs-seg.pt --size s --task segment
    python weights/convert_ec_weights.py ecdet_s_o3652coco.pth LibreECs-obj2coco.pt --size s --variant obj2coco

Add ``--verify`` to load the converted weights into a LibreEC wrapper and
run a smoke forward pass.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch

from _conversion_utils import (
    add_repo_root_to_path,
    extract_state_dict,
    save_checkpoint,
    wrap_libreyolo_checkpoint,
)

_SUPPORTED_TASKS = ("detect", "pose", "segment")
_DEFAULT_TASK = "detect"

# Per-task class-count + class-name overrides.
_POSE_NAMES = {0: "person"}


def convert_weights(
    input_path: str,
    output_path: str,
    size: str,
    task: str = "detect",
    nc: int = 80,
    variant: str | None = None,
) -> dict:
    if variant not in (None, "obj2coco"):
        raise ValueError(f"Unknown EC weight variant: {variant!r}")
    if task not in _SUPPORTED_TASKS:
        raise ValueError(f"Unknown EC task: {task!r}")
    add_repo_root_to_path()
    from libreyolo.models.ec.model import LibreEC
    from libreyolo.utils.serialization import (
        load_untrusted_torch_file,
        validate_checkpoint_metadata,
    )

    if LibreEC.detect_variant_from_filename(Path(output_path).name) != variant:
        raise ValueError("Output filename must match the requested EC weight variant")
    if variant is not None:
        suffix = {"detect": "", "segment": "-seg", "pose": "-pose"}[task]
        expected = f"LibreEC{size}{suffix}-{variant}.pt"
        if Path(output_path).name != expected:
            raise ValueError(f"Expected output filename {expected}")
    print(f"Loading upstream weights from {input_path}")
    raw = load_untrusted_torch_file(input_path, context="EC weight conversion")
    state_dict = extract_state_dict(raw)
    if not isinstance(state_dict, dict):
        raise TypeError(f"Could not extract state dict from {input_path}")
    print(f"Found {len(state_dict)} parameter entries")

    if task == "pose":
        nc = 1
        names: dict[int, str] | None = _POSE_NAMES
    else:
        names = None  # built from nc by wrap_libreyolo_checkpoint

    extra = {"num_keypoints": 17, "keypoint_dim": 3} if task == "pose" else {}
    if variant is not None:
        if nc != (1 if task == "pose" else 80):
            raise ValueError("obj2coco requires COCO classes (80, or 1 for pose)")
        detected_task = LibreEC.detect_checkpoint_task(state_dict) or "detect"
        if detected_task != task or LibreEC.detect_nb_classes(state_dict) != nc:
            raise ValueError("Checkpoint head does not match the requested COCO task")
        if LibreEC.detect_size(state_dict) != size:
            raise ValueError("Checkpoint size does not match --size")
        digest = hashlib.sha256()
        with open(input_path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        extra.update(
            {
                "weight_variant": variant,
                "license": "edgecrafter-non-commercial",
                "source": "https://github.com/capsule2077/edgecrafter/releases/tag/edgecrafterv1_o365",
                "source_sha256": digest.hexdigest(),
            }
        )

    libreyolo_ckpt = wrap_libreyolo_checkpoint(
        state_dict,
        model_family="ec",
        size=size,
        nc=nc,
        names=names,
        task=task,
        supported_tasks=_SUPPORTED_TASKS,
        default_task=_DEFAULT_TASK,
        **extra,
    )
    validate_checkpoint_metadata(libreyolo_ckpt, strict=True)
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
    parser = argparse.ArgumentParser(
        description="Convert EC weights to LibreYOLO format"
    )
    parser.add_argument("input", help="Upstream EdgeCrafter checkpoint (.pth)")
    parser.add_argument("output", help="Output LibreYOLO checkpoint (.pt)")
    parser.add_argument("--size", required=True, choices=["s", "m", "l", "x"])
    parser.add_argument("--task", default="detect", choices=_SUPPORTED_TASKS)
    parser.add_argument("--nc", type=int, default=80)
    parser.add_argument("--variant", choices=["obj2coco"], default=None)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    convert_weights(
        args.input, args.output, args.size, args.task, args.nc, args.variant
    )
    if args.verify:
        verify_conversion(args.output, args.size, args.task)
