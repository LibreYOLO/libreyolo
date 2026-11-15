"""Convert official FlowSeek checkpoints into LibreYOLO format.

Upstream (https://github.com/mattpoggi/flowseek, ``scripts/get_weights.sh``)
releases two networks, ``flowseek_T_*.pth`` (ResNet-18, Depth Anything V2
ViT-S) and ``flowseek_M_*.pth`` (ResNet-34, ViT-B). Sizes ``s`` and ``l`` reuse
the ``T`` and ``M`` weights with 12 refinement iterations instead of 4, as in
upstream ``config/eval/flowseek-[T,S,M,L].json``, so the size has to be given:

    t, s  <- flowseek_T_TartanCT_TSKH.pth
    m, l  <- flowseek_M_TartanCT_TSKH.pth

The released state dicts already embed the frozen Depth Anything V2 encoder
under ``dav2.*`` (bit-identical to ``depth_anything_v2_vit{s,b}.pth``), and the
key names match ``libreyolo/models/flowseek/_vendor``, so conversion only wraps
the tensors in the LibreYOLO v1.0 checkpoint schema. If a checkpoint lacks the
``dav2.*`` keys, pass ``--dav2 depth_anything_v2_vit{s,b}.pth`` to embed them,
so the resulting ``.pt`` is self-contained.

Weight licensing: the ViT-S encoder (t, s) is Apache-2.0; the ViT-B encoder
(m, l) is CC BY-NC 4.0, so ``m`` and ``l`` checkpoints are non-commercial.

Usage::

    python weights/convert_flowseek_weights.py flowseek_T_TartanCT_TSKH.pth \
        weights/LibreFlowSeekt-flow.pt --size t --verify
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch

from _conversion_utils import (
    add_repo_root_to_path,
    extract_state_dict,
    load_checkpoint,
    save_checkpoint,
)

_SIZE_TO_EMBED_DIM = {"t": 384, "s": 384, "m": 768, "l": 768}
_IMGSZ = 640


def _sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare_state_dict(state_dict: dict, size: str, dav2_state: dict | None = None) -> dict:
    """Validate an upstream FlowSeek state dict and embed DAv2 weights if needed."""
    state_dict = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}
    if dav2_state is not None:
        for key, value in dav2_state.items():
            state_dict.setdefault(f"dav2.{key}", value)

    add_repo_root_to_path()
    from libreyolo.models.flowseek.model import LibreFlowSeek, build_flowseek

    if not LibreFlowSeek.can_load(state_dict):
        raise ValueError(
            "This does not look like a FlowSeek checkpoint with its Depth Anything "
            "V2 encoder (expected bnet.*, merge_head.*, update_block.* and dav2.* "
            "keys). Pass --dav2 if the encoder weights are separate."
        )
    width = int(state_dict["dav2.pretrained.cls_token"].shape[-1])
    if width != _SIZE_TO_EMBED_DIM[size]:
        raise ValueError(
            f"Size {size!r} needs a ViT width of {_SIZE_TO_EMBED_DIM[size]}, but "
            f"the checkpoint encoder is {width} wide."
        )

    model = build_flowseek(size)
    result = model.load_state_dict(state_dict, strict=False)
    if result.missing_keys or result.unexpected_keys:
        raise ValueError(
            f"Key mismatch: missing={result.missing_keys[:10]} "
            f"unexpected={result.unexpected_keys[:10]}"
        )
    return state_dict


def convert_weights(
    input_path: str,
    output_path: str,
    *,
    size: str,
    dav2_path: str | None = None,
) -> dict:
    if size not in _SIZE_TO_EMBED_DIM:
        raise ValueError(f"size must be one of {sorted(_SIZE_TO_EMBED_DIM)}")
    print(f"Loading upstream FlowSeek weights from {input_path}")
    state_dict = extract_state_dict(load_checkpoint(input_path))
    dav2_state = extract_state_dict(load_checkpoint(dav2_path)) if dav2_path else None
    state_dict = prepare_state_dict(state_dict, size, dav2_state)
    print(f"{len(state_dict)} tensors, 0 missing, 0 unexpected")

    from libreyolo.utils.serialization import wrap_libreyolo_checkpoint

    ckpt = wrap_libreyolo_checkpoint(
        state_dict,
        model_family="flowseek",
        size=size,
        task="flow",
        nc=1,
        names={0: "flow"},
        imgsz=_IMGSZ,
    )
    ckpt["source"] = "https://github.com/mattpoggi/flowseek"
    ckpt["source_commit"] = "ba6407370bceeb96244cad4d60d37489f6eb471e"
    ckpt["source_sha256"] = _sha256(input_path)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(".tmp")
    save_checkpoint(ckpt, tmp)
    tmp.rename(output)
    print(f"Saved LibreYOLO-format checkpoint to {output}")
    return ckpt


def verify_conversion(converted_path: str) -> bool:
    add_repo_root_to_path()
    from libreyolo import LibreYOLO

    m = LibreYOLO(converted_path, device="cpu")
    print(f"  family={m.FAMILY} size={m.size} task={m.task} names={m.names} iters={m.iterations}")
    x = torch.zeros(2, 3, 128, 160)
    with torch.no_grad():
        out = m._forward_flow(x)
    print(f"  forward OK, flow shape {tuple(out.shape)}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert FlowSeek weights to LibreYOLO format")
    parser.add_argument("input", help="Upstream FlowSeek checkpoint (.pth)")
    parser.add_argument("output", help="Output LibreYOLO checkpoint (.pt)")
    parser.add_argument("--size", required=True, choices=sorted(_SIZE_TO_EMBED_DIM))
    parser.add_argument("--dav2", default=None, help="Depth Anything V2 weights, if not embedded")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    convert_weights(args.input, args.output, size=args.size, dav2_path=args.dav2)
    if args.verify:
        verify_conversion(args.output)
