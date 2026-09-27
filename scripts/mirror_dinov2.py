"""Build the LibreDINOv2 mirror checkpoint for the LibreYOLO Hugging Face org.

    .venv/Scripts/python.exe scripts/mirror_dinov2.py <out-dir>

Why this exists
---------------
LibreDINOv2 already asks for its weights at
https://huggingface.co/LibreYOLO/LibreDINOv2n/resolve/main/LibreDINOv2n.pt,
because the family does not override ``get_download_url``. Nothing was ever
uploaded there, so every first run instead reached out to Meta through the
backbone loader. Mirroring makes the documented path the real one.

What ships
----------
The DINOv2-S backbone, unmodified, wrapped in LibreYOLO checkpoint format. The
weights are Meta's Apache-2.0 release; this repackages them, it does not
retrain them. Semantic and classification heads are deliberately NOT included:
LibreYOLO publishes no trained head for this family, and shipping a randomly
initialised one inside a file called "weights" would be a lie. The loader
builds a fresh head exactly as it does today.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from libreyolo.models.dinov2.model import LibreDINOv2
from libreyolo.utils.serialization import (
    validate_checkpoint_metadata,
    wrap_libreyolo_checkpoint,
)

SIZE = "n"
UPSTREAM = "facebook/dinov2-small"


def main() -> int:
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "mirror-dinov2")
    out_dir.mkdir(parents=True, exist_ok=True)

    # task="embed" is the one head-free task, so the state dict that comes back
    # is the backbone and nothing else.
    print(f"building LibreDINOv2 size={SIZE} from {UPSTREAM} ...", flush=True)
    model = LibreDINOv2(None, size=SIZE, task="embed")
    state = model.model.state_dict()

    non_backbone = [k for k in state if not k.startswith("backbone.")]
    if non_backbone:
        raise SystemExit(f"refusing to ship non-backbone tensors: {non_backbone[:5]}")

    params = sum(v.numel() for v in state.values())
    print(f"  {len(state)} tensors, {params / 1e6:.2f}M parameters", flush=True)

    checkpoint = wrap_libreyolo_checkpoint(
        {k: v.cpu() for k, v in state.items()},
        model_family="dinov2",
        size=SIZE,
        task="semantic",  # the family default the filename implies
        nc=0,             # no trained head ships, so no class count is claimed
        imgsz=LibreDINOv2.INPUT_SIZES[SIZE],
    )

    errors = validate_checkpoint_metadata(checkpoint, strict=True)
    if errors:
        print("checkpoint metadata FAILED validation:", flush=True)
        for e in errors:
            print(f"  - {e}", flush=True)
        return 1
    print("  checkpoint metadata validates against the schema", flush=True)

    out = out_dir / f"LibreDINOv2{SIZE}.pt"
    torch.save(checkpoint, out)
    print(f"  wrote {out} ({out.stat().st_size / 1e6:.1f} MB)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
