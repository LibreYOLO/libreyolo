"""Convert official MoGe-3 weights into LibreYOLO format.

Every tensor of the official checkpoint is kept unchanged (encoder, neck,
point, normal, mask, and metric-scale heads, and the sparse refiner), so one
converted file serves both the ``depth`` and the ``normal`` task; the loader
keeps only the tensors the requested task's graph needs. The converter adds
LibreYOLO checkpoint schema v1.0 metadata for the chosen task.

Official source:
  repository: https://github.com/microsoft/MoGe
  code commit: 74fbce054ebed49800de42d0ad0e83495065719a (MIT)
  ViT-L weights:
    https://huggingface.co/Ruicheng/moge-3-vitl
    revision 184008f877d7ad1ad4c2cd2182a9bd1f63d0e5be (MIT)
  ViT-g weights:
    https://huggingface.co/Ruicheng/moge-3-vitg
    revision 6ef26c5a4b4148dab5ccaacdb08b72dc66380475 (MIT)

Usage::

    python weights/convert_moge3_weights.py model.pt \\
        weights/LibreMoGe3l-depth.pt --task depth --verify
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _conversion_utils import (
    add_repo_root_to_path,
    extract_state_dict,
    load_checkpoint,
    save_checkpoint,
    wrap_libreyolo_checkpoint,
)

_EMBED_DIM_TO_SIZE = {1024: "l", 1536: "g"}
_IMGSZ = 840
_TASKS = ("depth", "normal")


def detect_size(state_dict: dict) -> str | None:
    cls_token = state_dict.get("encoder.backbone.cls_token")
    if cls_token is None or getattr(cls_token, "ndim", 0) < 1:
        return None
    return _EMBED_DIM_TO_SIZE.get(int(cls_token.shape[-1]))


def convert_weights(
    input_path: str,
    output_path: str,
    *,
    task: str = "depth",
    size: str | None = None,
) -> dict:
    if task not in _TASKS:
        raise ValueError(f"task must be one of {_TASKS}, got {task!r}")
    print(f"Loading official MoGe-3 weights from {input_path}")
    state_dict = extract_state_dict(load_checkpoint(input_path))
    if not isinstance(state_dict, dict) or not (
        "refiner.input_proj.weight" in state_dict
        and "encoder.backbone.cls_token" in state_dict
        and "normal_head.output_blocks.4.weight" in state_dict
        and "points_head.output_blocks.4.weight" in state_dict
    ):
        raise ValueError("This does not look like an official MoGe-3 checkpoint.")

    detected = detect_size(state_dict)
    if size is None:
        size = detected
    elif detected is not None and detected != size:
        raise ValueError(
            f"--size {size!r} contradicts checkpoint width (detected {detected!r})."
        )
    if size is None:
        raise ValueError("Could not detect MoGe-3 size; pass --size explicitly.")

    print(f"Keeping all {len(state_dict)} tensors unchanged (size={size}, task={task})")
    wrapped = wrap_libreyolo_checkpoint(
        dict(state_dict),
        model_family="moge3",
        size=size,
        nc=1,
        names={0: task},
        task=task,
        imgsz=_IMGSZ,
    )
    output = Path(output_path)
    temporary = output.with_suffix(output.suffix + ".tmp")
    save_checkpoint(wrapped, temporary)
    temporary.replace(output)
    print(f"Wrote {output}")
    return wrapped


def verify(output_path: str, device: str = "cpu") -> None:
    add_repo_root_to_path()
    from libreyolo import LibreYOLO

    model = LibreYOLO(output_path, device=device)
    image = np.random.default_rng(0).integers(0, 255, (224, 308, 3), dtype=np.uint8)
    result = model.predict(image, imgsz=224, verbose=False)
    if isinstance(result, list):
        result = result[0]
    if model.task == "normal":
        assert result.normal_map is not None
        assert result.normal_map.data.shape == (224, 308, 3)
        result.normal_map.assert_normalized(atol=1e-5)
    else:
        depth = result.depth_map
        assert depth is not None and depth.metric
        assert tuple(depth.data.shape) == (224, 308)
        assert tuple(depth.point_map.shape) == (224, 308, 3)
    print(
        f"Verified {output_path}: family={model.family}, size={model.size}, "
        f"task={model.task}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("input", help="Official MoGe-3 model.pt")
    parser.add_argument("output", help="Output LibreYOLO .pt checkpoint")
    parser.add_argument("--task", choices=_TASKS, default="depth")
    parser.add_argument("--size", choices=["l", "g"], default=None)
    parser.add_argument("--device", default="cpu", help="Device for --verify.")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    convert_weights(args.input, args.output, task=args.task, size=args.size)
    if args.verify:
        verify(args.output, device=args.device)
