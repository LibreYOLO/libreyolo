"""Exact-parity gate for the V-JEPA 2 port.

Compares the native LibreYOLO encoder against unmodified Hugging Face
Transformers pinned at ``v5.1.0`` on fixed float32 CPU inputs, and requires
``max_abs_diff == 0.0`` for both the full final token tensor and the public
mean-pooled, L2-normalized vector.

The oracle must be the pinned version. Later Transformers releases refactor
the V-JEPA 2 attention dispatch and rotary helper, so a newer install is not
a valid oracle for this gate. Install it out-of-tree, for example::

    python -m pip install --target ./tf510 "transformers==5.1.0"
    python -m pip install --target ./tf510 --no-deps "kernels<0.10"
    PYTHONPATH=./tf510 python weights/parity_vjepa2.py --size l256

Usage::

    python weights/parity_vjepa2.py --size l256 [--frames 64] [--impl sdpa]
"""

from __future__ import annotations

import argparse
import sys

import torch
import torch.nn.functional as F

from _conversion_utils import add_repo_root_to_path

add_repo_root_to_path()

from libreyolo.models.vjepa2.nn import (  # noqa: E402
    VJEPA2_CONFIGS,
    LibreVJEPA2Encoder,
    VJEPA2Config,
)

# Pinned encoder snapshots. Revisions are exact and must not drift.
ENCODER_SOURCES: dict[str, tuple[str, str]] = {
    "l256": ("facebook/vjepa2-vitl-fpc64-256", "b3c1679b7c34d3255ef3547f27c7b226aefab26f"),
    "h256": ("facebook/vjepa2-vith-fpc64-256", "b5eac8703e3efdc1547fbb6ddfbeb133dc0bdee5"),
    "g256": ("facebook/vjepa2-vitg-fpc64-256", "875c192b7b704b87d1e1d99345769632dd5f739a"),
    "g384": ("facebook/vjepa2-vitg-fpc64-384", "12ca91694b230e0d4b5b0078af6f4ae1d51e933d"),
}

REQUIRED_ORACLE_VERSION = "5.1.0"

# The only top-level group the encoder conversion is allowed to drop. Anything
# else appearing here means the checkpoint layout changed and the conversion
# must be re-audited rather than silently discarding tensors.
ALLOWED_DROPPED_GROUPS = {"predictor"}


def run(size: str, frames: int, impl: str) -> None:
    import transformers

    if transformers.__version__ != REQUIRED_ORACLE_VERSION:
        raise SystemExit(
            f"parity oracle must be transformers=={REQUIRED_ORACLE_VERSION}, "
            f"found {transformers.__version__}. See this file's docstring."
        )
    from transformers.models.vjepa2.modeling_vjepa2 import VJEPA2Model

    repo, revision = ENCODER_SOURCES[size]
    torch.manual_seed(0)
    torch.set_grad_enabled(False)

    oracle = VJEPA2Model.from_pretrained(
        repo, revision=revision, dtype=torch.float32, attn_implementation=impl
    ).eval()
    cfg = oracle.config

    # Gate: never trust the size label for architecture. Validate the family
    # table against the pinned config before comparing anything.
    table = VJEPA2_CONFIGS[size]
    for key, expected in (
        ("hidden_size", cfg.hidden_size),
        ("num_attention_heads", cfg.num_attention_heads),
        ("num_hidden_layers", cfg.num_hidden_layers),
        ("mlp_ratio", cfg.mlp_ratio),
        ("crop_size", cfg.crop_size),
    ):
        if table[key] != expected:
            raise SystemExit(
                f"[{size}] architecture table {key}={table[key]!r} disagrees with "
                f"pinned config {expected!r}"
            )
    print(f"[{size}] architecture table matches pinned config @ {revision}")

    ours = LibreVJEPA2Encoder(VJEPA2Config.for_size(size, attn_implementation=impl)).eval()

    oracle_sd = oracle.state_dict()
    encoder_sd = {
        k[len("encoder."):]: v for k, v in oracle_sd.items() if k.startswith("encoder.")
    }
    dropped = {k.split(".")[0] for k in oracle_sd if not k.startswith("encoder.")}
    if not dropped <= ALLOWED_DROPPED_GROUPS:
        raise SystemExit(
            f"[{size}] unexpected non-encoder groups {sorted(dropped - ALLOWED_DROPPED_GROUPS)}; "
            "re-audit the conversion instead of dropping them"
        )
    ours.load_state_dict(encoder_sd, strict=True)
    params = sum(p.numel() for p in ours.parameters())
    print(
        f"[{size}] strict-loaded {len(encoder_sd)} tensors, {params / 1e6:.1f}M params, "
        f"dropped {sorted(dropped)}"
    )

    crop = cfg.crop_size
    x = torch.randn(1, frames, 3, crop, crop)

    reference_tokens = oracle.get_vision_features(x)
    our_tokens = ours(x)
    if reference_tokens.shape != our_tokens.shape:
        raise SystemExit(
            f"[{size}] token shape {tuple(our_tokens.shape)} != "
            f"{tuple(reference_tokens.shape)}"
        )
    token_diff = (reference_tokens - our_tokens).abs().max().item()
    print(
        f"[{size}] impl={impl} frames={frames} tokens={tuple(our_tokens.shape)} "
        f"TOKEN max_abs_diff={token_diff}"
    )
    if token_diff != 0.0:
        raise SystemExit(f"[{size}] token parity FAILED: {token_diff}")

    ref_vec = F.normalize(reference_tokens.mean(dim=1), dim=-1).float()
    our_vec = F.normalize(our_tokens.mean(dim=1), dim=-1).float()
    pooled_diff = (ref_vec - our_vec).abs().max().item()
    print(
        f"[{size}] pooled={tuple(our_vec.shape)} POOLED max_abs_diff={pooled_diff} "
        f"norm={our_vec.norm(dim=-1).item():.6f}"
    )
    if pooled_diff != 0.0:
        raise SystemExit(f"[{size}] pooled parity FAILED: {pooled_diff}")

    # A graph that loads but ignores time is a failed port, not a passing one.
    reversed_tokens = ours(x.flip(dims=[1]))
    delta = (our_tokens - reversed_tokens).abs().max().item()
    print(f"[{size}] temporal-order sensitivity max_abs_delta={delta:.6f}")
    if delta <= 1e-3:
        raise SystemExit(f"[{size}] model is insensitive to frame order ({delta})")

    print(f"[{size}] PARITY OK")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", choices=sorted(ENCODER_SOURCES), required=True)
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--impl", choices=["sdpa", "eager"], default="sdpa")
    args = parser.parse_args()
    run(args.size, args.frames, args.impl)


if __name__ == "__main__":
    sys.exit(main())
