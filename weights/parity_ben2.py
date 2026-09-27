"""Cross-load parity for LibreBEN2 and the pinned BEN2 Base reference.

Prerequisites are kept outside the LibreYOLO environment. Set:

``BEN2_UPSTREAM_DIR``
    Checkout of PramaLLC/BEN2 at commit
    ``2c99a5da477b5523585bfa5c893888a6e818a8f6``.
``BEN2_CHECKPOINT``
    Path to ``model.safetensors`` from Hugging Face revision
    ``e48a20765fb421d19dcdb0bf3cc61e802ca5ec8f``.

The comparison bypasses upstream's hard-coded CUDA autocast decorators so both
graphs run at fp32, then compares upstream's probability output with sigmoid of
LibreYOLO's contract-preserving logit output.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file


def main() -> int:
    upstream_dir = os.environ.get("BEN2_UPSTREAM_DIR")
    checkpoint = os.environ.get("BEN2_CHECKPOINT")
    if not upstream_dir or not checkpoint:
        raise SystemExit(
            "Set BEN2_UPSTREAM_DIR and BEN2_CHECKPOINT (see module docstring)."
        )

    repo_root = str(Path(__file__).resolve().parents[1])
    sys.path.insert(0, repo_root)
    sys.path.insert(0, upstream_dir)
    from BEN2 import BEN_Base

    from libreyolo.models.ben2.nn import LibreBEN2Model

    state_dict = load_file(checkpoint)
    upstream = BEN_Base().eval()
    ours = LibreBEN2Model().eval()
    upstream.load_state_dict(state_dict, strict=True)
    ours.load_state_dict(state_dict, strict=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    upstream.to(device)
    ours.to(device)
    # @inference_mode wraps @autocast in upstream. Invoke the undecorated
    # implementation so fp32 is compared with fp32.
    upstream_forward = BEN_Base.forward.__wrapped__.__wrapped__
    ok = True
    batches = (1, 2) if device.type == "cuda" else (1,)
    for batch in batches:
        torch.manual_seed(batch)
        x = torch.randn(batch, 3, 1024, 1024, device=device)
        with torch.inference_mode():
            expected = upstream_forward(upstream, x)
            actual = torch.sigmoid(ours(x))
        difference = (expected - actual).abs()
        max_abs = float(difference.max())
        mean_abs = float(difference.mean())
        print(f"batch={batch} max_abs_diff={max_abs:.3e} mean_abs_diff={mean_abs:.3e}")
        ok = ok and max_abs == 0.0
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
