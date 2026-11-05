"""Byte parity of LibreYOLO's Falcon Perception port against the upstream engine.

Runs the upstream batch engine (tiiuae/Falcon-Perception, Apache-2.0) and the
LibreYOLO port on identical (image, query) pairs on CPU in fp32, then compares:

  * every forward pass (prefill + each decode step): logits and hidden states
    (max_abs_diff and bitwise equality),
  * the generated token stream,
  * the raw normalised box values (x, y, h, w) as float bit patterns,
  * the binary masks as bytes (sha256) at upstream's own resolution and at the
    original image size.

Upstream needs triton and CUDA FlexAttention kernels, which do not exist on a
CPU box. The harness therefore stubs ``triton`` and replaces the two compiled
FlexAttention entry points and the block-mask builder with their eager
(uncompiled) PyTorch equivalents. Nothing in the upstream model math changes.

Usage:
    uv run python weights/parity_falcon_perception.py \\
        --upstream=/path/to/Falcon-Perception --weights600=/dir --weights300=/dir

Outputs go to results/falcon_perception_parity/ (cases.jsonl, summary.json).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import types
from pathlib import Path
from unittest import mock

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def stub_upstream(upstream_dir: str):
    """Make the upstream package importable and runnable on CPU."""
    sys.path.insert(0, upstream_dir)

    from torch.nn.attention import flex_attention as fa

    # attention.py / anyup.py need no triton; import them first so torch's
    # compiler stack never sees the stub below.
    import falcon_perception.anyup as anyup
    import falcon_perception.attention as att

    att.compiled_flex_attn_decode = fa.flex_attention
    att.compiled_flex_attn_prefill = fa.flex_attention
    att._compiled_create_block_mask = lambda *a, **k: fa.create_block_mask(
        *a, device="cpu", **k
    )
    anyup.compiled_flex_attn_prefill = fa.flex_attention

    # model.py defines a triton kernel at import time; give it a stub for the
    # duration of that import only.
    triton = types.ModuleType("triton")
    triton.jit = lambda f: f
    triton.cdiv = lambda a, b: (a + b - 1) // b
    tl = types.ModuleType("triton.language")
    tl.constexpr = object
    triton.language = tl
    sys.modules["triton"] = triton
    sys.modules["triton.language"] = tl
    try:
        import falcon_perception.model as um
    finally:
        del sys.modules["triton"], sys.modules["triton.language"]

    def squared_relu_gate(packed, hidden_dim):
        gate = packed[..., 0::2]
        up = packed[..., 1::2]
        gate = torch.where(gate > 0, gate, torch.zeros_like(gate))
        return gate * gate * up

    um.squared_relu_gate = squared_relu_gate
    return um


def sha(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]


def f32_bits(x: float) -> int:
    return int(np.float32(x).view(np.uint32))


def record_forward(model, store):
    orig = model.forward

    def wrapped(*a, **k):
        logits, h = orig(*a, **k)
        store.append((logits.detach().clone(), h.detach().clone()))
        return logits, h

    model.forward = wrapped
    return orig


def run_upstream(model, tokenizer, image, query, task, steps):
    from falcon_perception import build_prompt_for_task
    from falcon_perception.batch_inference import BatchInferenceEngine, process_batch_and_generate
    from falcon_perception.data import load_image
    from falcon_perception.visualization_utils import pair_bbox_entries

    pil = load_image(image).convert("RGB")
    prompt = build_prompt_for_task(query, task)
    inputs = process_batch_and_generate(
        tokenizer, [(pil, prompt)], max_length=4096, min_dimension=256, max_dimension=1024
    )
    engine = BatchInferenceEngine(model, tokenizer)
    record_forward(model, steps)
    tokens, aux = engine.generate(
        **inputs,
        max_new_tokens=2048,
        temperature=0.0,
        stop_token_ids=[tokenizer.eos_token_id, tokenizer.end_of_query_token_id],
        seed=42,
        task=task,
    )
    L = inputs["tokens"].shape[1]
    a = aux[0]
    masks = []
    final_masks = []
    if task == "segmentation":
        from falcon_perception.visualization_utils import (
            decode_coco_rle,
            detections_from_batch_aux,
        )

        masks = [decode_coco_rle(r) for r in a.masks_rle]
        # Upstream's own final step: RLE -> NEAREST resize to the original image.
        dets = detections_from_batch_aux(
            a, orig_hw=(pil.height, pil.width), segmentation=True
        )
        final_masks = [
            None if d["mask"] is None else d["mask"].numpy().astype(np.uint8) for d in dets
        ]
    return {
        "L": L,
        "tokens": tokens[0].cpu().numpy(),
        "boxes": pair_bbox_entries(a.bboxes_raw),
        "raw": a.bboxes_raw,
        "masks": masks,
        "final_masks": final_masks,
        "pil_size": pil.size,
    }


def run_ours(lib, image, query, task, steps):
    from PIL import Image

    from libreyolo.models.falcon_perception.generate import pair_bbox_entries

    pil = Image.open(image).convert("RGB")
    record_forward(lib.model, steps)
    gen = lib.run_query(pil, query, segmentation=(task == "segmentation"))
    masks = []
    if gen.masks is not None:
        masks = [m.to(torch.uint8).cpu().numpy() for m in gen.masks]
    from libreyolo.models.falcon_perception.model import _mask_to_original

    width, height = pil.size
    final_masks = []
    if task == "segmentation":
        final_masks = [
            _mask_to_original(gen.masks, i, width, height)
            for i in range(len(pair_bbox_entries(gen.boxes_raw)))
        ]
    return {
        "tokens": gen.tokens.numpy(),
        "boxes": pair_bbox_entries(gen.boxes_raw),
        "raw": gen.boxes_raw,
        "masks": masks,
        "final_masks": final_masks,
        "pil_size": pil.size,
    }


def compare(up, ours, up_steps, our_steps):
    out = {}
    out["n_forward_up"], out["n_forward_ours"] = len(up_steps), len(our_steps)
    n = min(len(up_steps), len(our_steps))
    max_l = max_h = 0.0
    bit_l = bit_h = True
    for (ul, uh), (ol, oh) in zip(up_steps[:n], our_steps[:n]):
        max_l = max(max_l, float((ul - ol).abs().max()))
        max_h = max(max_h, float((uh - oh).abs().max()))
        bit_l &= torch.equal(ul, ol)
        bit_h &= torch.equal(uh, oh)
    out["logits_max_abs_diff"], out["hidden_max_abs_diff"] = max_l, max_h
    out["logits_bitwise"], out["hidden_bitwise"] = bool(bit_l), bool(bit_h)

    L = up["L"]
    ut, ot = up["tokens"][:], ours["tokens"][:]
    k = len(ot)
    out["tokens_equal"] = bool(np.array_equal(ut[:k], ot)) and bool(
        (ut[k:] == ut[k:][:1]).all() if len(ut) > k else True
    )
    out["generated_tokens"] = int(k - L)

    ub, ob = up["boxes"], ours["boxes"]
    out["n_boxes_up"], out["n_boxes_ours"] = len(ub), len(ob)
    out["boxes_bitwise"] = len(ub) == len(ob) and all(
        all(f32_bits(a[key]) == f32_bits(b[key]) and a[key] == b[key] for key in "xyhw")
        for a, b in zip(ub, ob)
    )

    um, om = up["masks"], ours["masks"]
    out["n_masks_up"], out["n_masks_ours"] = len(um), len(om)
    out["masks_bytes_equal"] = len(um) == len(om) and all(
        np.array_equal(a.astype(np.uint8), b.astype(np.uint8)) for a, b in zip(um, om)
    )
    uf, of = up["final_masks"], ours["final_masks"]
    out["final_masks_bytes_equal"] = len(uf) == len(of) and all(
        (np.array_equal(a, b) if a is not None else not b.any()) for a, b in zip(uf, of)
    )
    out["mask_sha_up"] = [sha(m.astype(np.uint8)) for m in um]
    out["mask_sha_ours"] = [sha(m.astype(np.uint8)) for m in om]
    out["pass"] = bool(
        out["tokens_equal"]
        and out["boxes_bitwise"]
        and out["masks_bytes_equal"]
        and out["final_masks_bytes_equal"]
        and out["n_forward_up"] == out["n_forward_ours"]
    )
    out["pass_bitwise_logits"] = bool(out["pass"] and bit_l and bit_h)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", required=True)
    ap.add_argument("--weights600", required=True)
    ap.add_argument("--weights300", required=True)
    ap.add_argument(
        "--detect-only",
        action="store_true",
        help="skip segmentation (upstream's eager AnyUp needs GBs at large canvases)",
    )
    ap.add_argument("--out", default=str(REPO / "results" / "falcon_perception_parity"))
    ap.add_argument(
        "--cases",
        default=json.dumps(
            [
                ["libreyolo/assets/parkour.jpg", "person"],
                ["libreyolo/assets/guggenheim-bilbao.jpg", "building"],
            ]
        ),
    )
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    torch.set_num_threads(4)
    stub_upstream(args.upstream)
    from falcon_perception import load_from_hf_export

    from libreyolo.models.falcon_perception import LibreFalconPerception

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = json.loads(args.cases)
    rows = []
    t0 = time.time()

    for size, wdir, tasks in (
        ("600m", args.weights600, ("segmentation", "detection")),
        ("300m", args.weights300, ("detection",)),
    ):
        up_model, up_tok, _ = load_from_hf_export(hf_local_dir=wdir)
        up_model = up_model.to(device="cpu", dtype=torch.float32).eval()
        # Load the port from the same local snapshot instead of downloading.
        with mock.patch.object(
            LibreFalconPerception, "_ensure_weights", lambda self, d=wdir: d
        ):
            lib = LibreFalconPerception(size=size, device="cpu")
        for image, query in cases:
            for task in tasks:
                if args.detect_only and task == "segmentation":
                    continue
                img = str(REPO / image)
                up_steps, our_steps = [], []
                tt = time.time()
                up = run_upstream(up_model, up_tok, img, query, task, up_steps)
                up_model.forward = type(up_model).forward.__get__(up_model)
                ours = run_ours(lib, img, query, task, our_steps)
                lib.model.forward = type(lib.model).forward.__get__(lib.model)
                row = {"size": size, "image": image, "query": query, "task": task}
                row.update(compare(up, ours, up_steps, our_steps))
                row["seconds"] = round(time.time() - tt, 1)
                rows.append(row)
                print(json.dumps(row), flush=True)
                with open(out_dir / "cases.jsonl", "a") as f:
                    f.write(json.dumps(row) + "\n")
                del up_steps, our_steps

    summary = {
        "cases": len(rows),
        "passed": sum(r["pass"] for r in rows),
        "passed_bitwise_logits": sum(r["pass_bitwise_logits"] for r in rows),
        "seconds": round(time.time() - t0, 1),
        "torch": torch.__version__,
        "device": "cpu",
        "dtype": "float32",
        "upstream_commit": "c457916c9974efbacfa91f0f6ecc2c49c6543e56",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    sys.exit(0 if summary["passed_bitwise_logits"] == summary["cases"] else 1)


if __name__ == "__main__":
    main()
