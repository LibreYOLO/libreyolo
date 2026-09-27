"""Exact-parity gate for the PE Core port.

Cross-loads each pinned OpenCLIP-compatible PE Core snapshot into the native
LibreYOLO implementation and asserts ``max_abs_diff == 0.0`` against
unmodified ``open_clip_torch==3.2.0`` on fixed float32 CPU inputs, for:

* image embeddings
* text embeddings
* zero-shot logits
* fixed-frame video embeddings (mean of frame embeddings, normalized once)

Usage::

    python weights/parity_pe.py                 # all sizes
    python weights/parity_pe.py --sizes t16 s16 # a subset

``g14`` needs roughly 20 GB of RAM to hold both the port and the oracle in
float32; it is excluded from ``--sizes all-small``.
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from libreyolo.models.pe.nn import PE_CONFIGS, build_pe_model  # noqa: E402

# Pinned source snapshots. Both repo id and revision are part of the gate:
# a moved revision must fail loudly rather than silently re-parity.
PINNED_SOURCES = {
    "t16": ("timm/PE-Core-T-16-384", "7fe539ed578ac49a1c2b4f946e4b0747704c825a"),
    "s16": ("timm/PE-Core-S-16-384", "3249a38eb1c432ec19231c6fe1774acb6a4e4efe"),
    "b16": ("timm/PE-Core-B-16", "0038414f37721c5eafc1a5e0da802d291c909de3"),
    "l14": ("timm/PE-Core-L-14-336", "8eff41b3f687e50a323662c2dda5eb3588c6dd35"),
    "g14": ("timm/PE-Core-bigG-14-448", "17aa0c25addfa14198fa2ff73d845a22d433432e"),
}

WEIGHT_FILE = "open_clip_model.safetensors"


def fetch(size: str) -> str:
    from huggingface_hub import hf_hub_download

    repo, revision = PINNED_SOURCES[size]
    return hf_hub_download(repo, WEIGHT_FILE, revision=revision)


def _logits(img: torch.Tensor, txt: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return scale.exp() * (F.normalize(img, dim=-1) @ F.normalize(txt, dim=-1).T)


def check_size(size: str, frames: int = 4, batch: int = 2) -> bool:
    """Compare the port against the oracle on identical fixed inputs.

    The oracle is built, run, and *freed* before the port is built. Holding both
    in float32 at once costs ~20 GB for g14 and segfaults on a 32 GB machine.
    """
    import open_clip
    import safetensors.torch as st

    cfg = PE_CONFIGS[size]
    path = fetch(size)
    res = cfg.image_size

    # g14 is a 1.88B vision tower at 448px; keep the fixtures small.
    if size == "g14":
        batch, frames = 1, 2

    torch.manual_seed(0)
    images = torch.randn(batch, 3, res, res)
    tokens = torch.randint(0, 49000, (3, cfg.context_length))
    tokens[:, -1] = 49407  # EOT marker drives argmax pooling
    clips = torch.randn(batch, frames, 3, res, res)

    # --- oracle pass, then free ------------------------------------------
    oracle, _, _ = open_clip.create_model_and_transforms(
        cfg.open_clip_model_name, pretrained=path
    )
    oracle.eval()
    with torch.no_grad():
        ref_img = oracle.encode_image(images)
        ref_txt = oracle.encode_text(tokens)
        ref_log = _logits(ref_img, ref_txt, oracle.logit_scale)
        flat = clips.reshape(batch * frames, 3, res, res)
        ref_vid = F.normalize(
            oracle.encode_image(flat).reshape(batch, frames, -1).mean(dim=1), dim=-1
        )
    del oracle
    gc.collect()

    # --- port pass --------------------------------------------------------
    ours = build_pe_model(size)
    result = ours.load_state_dict(st.load_file(path), strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    ours.eval()
    with torch.no_grad():
        ours_img = ours.encode_image(images)
        ours_txt = ours.encode_text(tokens)
        ours_log = _logits(ours_img, ours_txt, ours.logit_scale)
        ours_vid = ours.encode_video(clips)

    ok = True
    for label, a, b in (
        ("image", ours_img, ref_img),
        ("text", ours_txt, ref_txt),
        ("logits", ours_log, ref_log),
        ("video", ours_vid, ref_vid),
    ):
        diff = (a - b).abs().max().item()
        status = "OK " if diff == 0.0 else "FAIL"
        print(
            f"  [{status}] {size:>4s} {label:<7s} shape={tuple(a.shape)} "
            f"max_abs_diff={diff:.6g}"
        )
        ok = ok and diff == 0.0

    del ours
    gc.collect()
    return ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="*", default=list(PE_CONFIGS))
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    failures = [size for size in args.sizes if not check_size(size)]
    if failures:
        print(f"PARITY FAILED for: {', '.join(failures)}")
        return 1
    print(f"PARITY OK for: {', '.join(args.sizes)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
