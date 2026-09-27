"""Exact HVI-CIDNet parity against the pinned MIT upstream checkout.

Set ``LIBREYOLO_HVI_CIDNET_UPSTREAM`` to commit
``eb43d7d91e9a336c66856824ff9e4603ae41f408`` and
``LIBREYOLO_HVI_CIDNET_CHECKPOINT`` to the official Generalization
``model.safetensors``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

from convert_hvi_cidnet_weights import OFFICIAL_SHA256, OFFICIAL_SIZE, file_sha256


UPSTREAM_COMMIT = "eb43d7d91e9a336c66856824ff9e4603ae41f408"


def _assert_upstream_pin(upstream_dir: Path) -> None:
    commit = subprocess.check_output(
        ["git", "-C", str(upstream_dir), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if commit != UPSTREAM_COMMIT:
        raise RuntimeError(
            f"HVI-CIDNet upstream checkout must be {UPSTREAM_COMMIT}, got {commit}."
        )


def _assert_checkpoint_pin(checkpoint: Path) -> None:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"HVI-CIDNet checkpoint not found: {checkpoint}")
    source_size = checkpoint.stat().st_size
    if source_size != OFFICIAL_SIZE:
        raise ValueError(
            "HVI-CIDNet source size mismatch: "
            f"expected {OFFICIAL_SIZE} bytes, got {source_size}."
        )
    source_digest = file_sha256(checkpoint)
    if source_digest != OFFICIAL_SHA256:
        raise ValueError(
            "HVI-CIDNet source SHA-256 mismatch: "
            f"expected {OFFICIAL_SHA256}, got {source_digest}."
        )


def main() -> None:
    upstream_dir = os.environ.get("LIBREYOLO_HVI_CIDNET_UPSTREAM")
    checkpoint = os.environ.get("LIBREYOLO_HVI_CIDNET_CHECKPOINT")
    if not upstream_dir or not checkpoint:
        raise SystemExit(
            "Set LIBREYOLO_HVI_CIDNET_UPSTREAM and LIBREYOLO_HVI_CIDNET_CHECKPOINT."
        )
    upstream_root = Path(upstream_dir).resolve()
    checkpoint_path = Path(checkpoint).resolve()
    _assert_upstream_pin(upstream_root)
    _assert_checkpoint_pin(checkpoint_path)

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(upstream_root))

    from net.CIDNet import CIDNet as UpstreamCIDNet

    from libreyolo.models.hvi_cidnet.nn import CIDNet

    state_dict = load_file(checkpoint_path, device="cpu")
    upstream = UpstreamCIDNet()
    ours = CIDNet()
    upstream.load_state_dict(state_dict, strict=True)
    ours.load_state_dict(state_dict, strict=True)
    upstream.trans.alpha_s = 1.0
    upstream.trans.alpha = 1.0
    upstream.trans.gated = True
    upstream.trans.gated2 = True
    upstream.eval()
    ours.eval()

    torch.manual_seed(7)
    image = torch.rand(1, 3, 32, 40)
    with torch.inference_mode():
        expected = upstream(image)
        actual = ours(image)
    max_abs_diff = float((expected - actual).abs().max())
    if max_abs_diff != 0.0:
        raise AssertionError(f"HVI-CIDNet max_abs_diff={max_abs_diff}, expected 0.")
    print("HVI-CIDNet fp32 parity OK: max_abs_diff=0")


if __name__ == "__main__":
    main()
