"""Convert upstream LoopDVR (LoopDet / LoopSeg / LoopPose) weights into LibreYOLO format.

Upstream releases from ``Siruuui/LoopDVR`` (v1.0) ship as
``{"model_id": ..., "state_dict": state_dict}``. LibreYOLO checkpoints add
schema v1.0 metadata (``model_family``, ``task``, ``nc``, ``size``, ``names``,
``imgsz``) so ``LibreYOLO()`` routes without filename heuristics.

LoopDet, LoopSeg, and LoopPose module names mirror the native LibreLoopDVR port
verbatim (``looped_vit.*``, ``encoder.*``, ``decoder.*``), so conversion is a
pure metadata wrap with strict validation.

Usage:
    python weights/convert_loopdvr_weights.py loopdet-s.pth weights/LibreLoopDVRs.pt --size s --task detect
    python weights/convert_loopdvr_weights.py loopseg-s.pth weights/LibreLoopDVRs-seg.pt --size s --task segment
    python weights/convert_loopdvr_weights.py looppose-s.pth weights/LibreLoopDVRs-pose.pt --size s --task pose

Add ``--verify`` to load the converted checkpoint into LibreLoopDVR and run a
smoke forward pass.
"""

from __future__ import annotations

import argparse

import torch
from _conversion_utils import (
    add_repo_root_to_path,
    extract_state_dict,
    save_checkpoint,
    wrap_libreyolo_checkpoint,
)

_SUPPORTED_TASKS = ("detect", "pose", "segment")
_DEFAULT_TASK = "detect"
_POSE_NAMES = {0: "person"}


def convert_weights(
    input_path: str,
    output_path: str,
    size: str | None = None,
    task: str | None = None,
    nc: int | None = None,
) -> dict:
    add_repo_root_to_path()
    from libreyolo.models.loopdvr.model import LibreLoopDVR
    from libreyolo.utils.serialization import (
        load_untrusted_torch_file,
        validate_checkpoint_metadata,
    )

    print(f"Loading upstream LoopDVR weights from {input_path}")
    raw = load_untrusted_torch_file(input_path, context="LoopDVR weight conversion")
    state_dict = extract_state_dict(raw)
    if not isinstance(state_dict, dict):
        raise TypeError(f"Could not extract state dict from {input_path}")
    if not LibreLoopDVR.can_load(state_dict):
        raise ValueError(f"{input_path} does not appear to be a LoopDVR checkpoint")
    print(f"Found {len(state_dict)} parameter entries")

    detected_size = LibreLoopDVR.detect_size(state_dict)
    if size is None:
        size = detected_size
    elif detected_size is not None and detected_size != size:
        raise ValueError(
            f"Checkpoint tensor shapes indicate size={detected_size!r}, "
            f"which does not match --size={size!r}"
        )
    if size not in ("s", "m", "l", "x"):
        raise ValueError(f"Unknown LoopDVR size: {size!r}")

    detected_task = LibreLoopDVR.detect_checkpoint_task(state_dict) or _DEFAULT_TASK
    if task is None:
        task = detected_task
    elif detected_task != task:
        raise ValueError(
            f"Checkpoint head indicates task={detected_task!r}, "
            f"which does not match --task={task!r}"
        )
    if task not in _SUPPORTED_TASKS:
        raise ValueError(f"Unknown LoopDVR task: {task!r}")

    detected_nc = LibreLoopDVR.detect_nb_classes(state_dict)
    if task == "pose":
        resolved_nc = 1
        names: dict[int, str] | None = _POSE_NAMES
    else:
        resolved_nc = nc if nc is not None else (detected_nc or 80)
        if detected_nc is not None and resolved_nc != detected_nc:
            raise ValueError(
                f"Checkpoint head has {detected_nc} classes, got --nc={resolved_nc}"
            )
        names = None  # populated from nc by wrap_libreyolo_checkpoint

    extra = {"num_keypoints": 17, "keypoint_dim": 3} if task == "pose" else {}
    libreyolo_ckpt = wrap_libreyolo_checkpoint(
        state_dict,
        model_family="loopdvr",
        size=size,
        nc=resolved_nc,
        names=names,
        task=task,
        supported_tasks=_SUPPORTED_TASKS,
        default_task=_DEFAULT_TASK,
        **extra,
    )
    validate_checkpoint_metadata(libreyolo_ckpt, strict=True)
    out = save_checkpoint(libreyolo_ckpt, output_path)
    print(
        f"Saved LibreYOLO-format checkpoint to {out} "
        f"(size={size}, task={task}, nc={resolved_nc})"
    )
    return libreyolo_ckpt


def verify_conversion(converted_path: str, size: str, task: str) -> bool:
    add_repo_root_to_path()
    from libreyolo import LibreYOLO

    print(f"\nLoading converted weights via LibreYOLO({converted_path!r})...")
    m = LibreYOLO(converted_path, device="cpu")
    assert m.FAMILY == "loopdvr", f"Expected family='loopdvr', got {m.FAMILY!r}"
    assert m.size == size, f"Expected size={size!r}, got {m.size!r}"
    assert m.task == task, f"Expected task={task!r}, got {m.task!r}"
    print(f"  family={m.FAMILY} size={m.size} task={m.task} nc={m.nb_classes}")

    m.model.eval()
    with torch.no_grad():
        out = m.model(torch.zeros(1, 3, 640, 640))

    if task == "detect":
        assert out["pred_logits"].shape == (1, 300, m.nb_classes)
        assert out["pred_boxes"].shape == (1, 300, 4)
        print("  detect forward pass OK")
    elif task == "pose":
        assert out["pred_logits"].shape == (1, 60, 2)
        assert out["pred_keypoints"].shape == (1, 60, 34)
        print("  pose forward pass OK")
    elif task == "segment":
        assert out["pred_logits"].shape == (1, 300, m.nb_classes)
        assert out["pred_boxes"].shape == (1, 300, 4)
        assert out["pred_masks"].shape == (1, 300, 160, 160)
        print("  segment forward pass OK")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert LoopDVR upstream weights to LibreYOLO format"
    )
    parser.add_argument("input", help="Upstream LoopDVR checkpoint (.pth)")
    parser.add_argument("output", help="Output LibreYOLO checkpoint (.pt)")
    parser.add_argument("--size", choices=["s", "m", "l", "x"], default=None)
    parser.add_argument("--task", choices=_SUPPORTED_TASKS, default=None)
    parser.add_argument("--nc", type=int, default=None)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    ckpt = convert_weights(args.input, args.output, args.size, args.task, args.nc)
    if args.verify:
        verify_conversion(args.output, ckpt["size"], ckpt["task"])
