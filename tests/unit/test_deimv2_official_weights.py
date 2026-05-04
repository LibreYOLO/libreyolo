"""Opt-in tests for official DEIMv2 safetensors checkpoints."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

pytestmark = [pytest.mark.unit, pytest.mark.deimv2, pytest.mark.slow]

DEIMV2_OFFICIAL_CASES = [
    ("atto", 320, 100, "Intellindust/DEIMv2_HGNetv2_ATTO_COCO"),
    ("femto", 416, 150, "Intellindust/DEIMv2_HGNetv2_FEMTO_COCO"),
    ("pico", 640, 200, "Intellindust/DEIMv2_HGNetv2_PICO_COCO"),
    ("n", 640, 300, "Intellindust/DEIMv2_HGNetv2_N_COCO"),
    ("s", 640, 300, "Intellindust/DEIMv2_DINOv3_S_COCO"),
    ("m", 640, 300, "Intellindust/DEIMv2_DINOv3_M_COCO"),
    ("l", 640, 300, "Intellindust/DEIMv2_DINOv3_L_COCO"),
    ("x", 640, 300, "Intellindust/DEIMv2_DINOv3_X_COCO"),
]


def _resolve_official_weight(size: str, repo_id: str) -> Path:
    ckpt_dir = os.environ.get("DEIMV2_OFFICIAL_CKPT_DIR")
    if ckpt_dir:
        root = Path(ckpt_dir)
        candidates = [
            root / f"{size}.safetensors",
            root / size / "model.safetensors",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        pytest.skip(f"No official DEIMv2 {size} checkpoint found under {root}")

    if os.environ.get("DEIMV2_ALLOW_HF_DOWNLOAD") == "1":
        pytest.importorskip("huggingface_hub")
        from huggingface_hub import hf_hub_download

        return Path(hf_hub_download(repo_id=repo_id, filename="model.safetensors"))

    pytest.skip(
        "Set DEIMV2_OFFICIAL_CKPT_DIR or DEIMV2_ALLOW_HF_DOWNLOAD=1 "
        "to run official DEIMv2 checkpoint tests."
    )


@pytest.mark.parametrize(
    ("size", "input_size", "queries", "repo_id"), DEIMV2_OFFICIAL_CASES
)
def test_deimv2_official_safetensors_load_and_forward(
    size, input_size, queries, repo_id
):
    """Every released HF safetensors checkpoint should load and run."""
    from libreyolo import LibreDEIMv2

    ckpt = _resolve_official_weight(size, repo_id)
    model = LibreDEIMv2(str(ckpt), size=size, device="cpu")
    model.model.eval()

    with torch.no_grad():
        out = model.model(torch.zeros(1, 3, input_size, input_size))

    assert out["pred_logits"].shape == (1, queries, 80)
    assert out["pred_boxes"].shape == (1, queries, 4)
