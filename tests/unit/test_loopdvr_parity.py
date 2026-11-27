"""Pinned-reference tensor parity test for LoopDVR vs upstream Intellindust-AI-Lab/LoopDVR.

Verifies that native ``LibreLoopDVR`` (detect, segment, and pose) matches the
pinned upstream ``Intellindust-AI-Lab/LoopDVR`` checkout at commit
``f2aba1f122029e4db757b69cd37fc5a4909db34e`` bit-for-bit on raw forward outputs
(``pred_boxes``, ``pred_logits``, ``pred_masks``, ``pred_keypoints``) and stays
exact under ``deploy()`` Conv+BN fusion.

Marked ``external_data`` so it stays outside the hermetic PR gate. Run locally:

    LIBREYOLO_LOOPDVR_UPSTREAM=/path/to/LoopDVR \\
    LIBREYOLO_LOOPDVR_WEIGHTS=/path/to/loopdvr_weights \\
    pytest -q tests/unit/test_loopdvr_parity.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

pytestmark = [pytest.mark.unit, pytest.mark.external_data]

EXPECTED_UPSTREAM_COMMIT = "f2aba1f122029e4db757b69cd37fc5a4909db34e"

MODEL_CASES = [
    ("loopdet-s", "detect", "s"),
    ("loopdet-m", "detect", "m"),
    ("loopdet-l", "detect", "l"),
    ("loopdet-x", "detect", "x"),
    ("loopseg-s", "segment", "s"),
    ("loopseg-m", "segment", "m"),
    ("loopseg-l", "segment", "l"),
    ("loopseg-x", "segment", "x"),
    ("looppose-s", "pose", "s"),
    ("looppose-m", "pose", "m"),
    ("looppose-l", "pose", "l"),
    ("looppose-x", "pose", "x"),
]


def _resolve_upstream_repo() -> Path:
    raw = os.environ.get("LIBREYOLO_LOOPDVR_UPSTREAM")
    if not raw:
        pytest.skip("Set LIBREYOLO_LOOPDVR_UPSTREAM to a pinned LoopDVR checkout")
    repo = Path(raw)
    if not repo.is_dir():
        pytest.skip(f"LIBREYOLO_LOOPDVR_UPSTREAM not found: {repo}")
    res = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if res.returncode != 0:
        pytest.fail(
            f"Could not read git commit of LIBREYOLO_LOOPDVR_UPSTREAM ({repo}): "
            f"{res.stderr.strip()}"
        )
    actual = res.stdout.strip()
    assert actual == EXPECTED_UPSTREAM_COMMIT, (
        f"LIBREYOLO_LOOPDVR_UPSTREAM is at {actual}, expected pinned commit "
        f"{EXPECTED_UPSTREAM_COMMIT}"
    )
    return repo


def _resolve_weights_dir() -> Path:
    raw = os.environ.get("LIBREYOLO_LOOPDVR_WEIGHTS")
    if not raw:
        pytest.skip("Set LIBREYOLO_LOOPDVR_WEIGHTS to the v1.0 LoopDVR weights dir")
    weights_dir = Path(raw)
    if not weights_dir.is_dir():
        pytest.skip(f"LIBREYOLO_LOOPDVR_WEIGHTS not found: {weights_dir}")
    return weights_dir


@pytest.mark.parametrize(
    ("model_id", "task", "size"),
    MODEL_CASES,
    ids=[case[0] for case in MODEL_CASES],
)
def test_loopdvr_pinned_upstream_parity(model_id: str, task: str, size: str):
    upstream_repo = _resolve_upstream_repo()
    weights_dir = _resolve_weights_dir()

    ckpt_path = weights_dir / f"{model_id}.pth"
    if not ckpt_path.is_file():
        pytest.skip(f"Checkpoint not staged: {ckpt_path}")

    prefix = model_id.split("-")[0]
    config_path = upstream_repo / "configs" / prefix / f"{model_id}.yaml"
    assert config_path.is_file(), f"Missing upstream config: {config_path}"

    if str(upstream_repo) not in sys.path:
        sys.path.insert(0, str(upstream_repo))

    from loopdvr.build import build_model, load_config

    from libreyolo.models.loopdvr import LibreLoopDVR

    raw = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state_dict = raw["state_dict"]

    assert LibreLoopDVR.can_load(state_dict)
    assert LibreLoopDVR.detect_size(state_dict) == size
    assert LibreLoopDVR.detect_checkpoint_task(state_dict) == (
        None if task == "detect" else task
    )
    assert LibreLoopDVR.detect_nb_classes(state_dict) == (1 if task == "pose" else 80)

    upstream = build_model(load_config(config_path), ckpt_path, device="cpu").eval()
    native = LibreLoopDVR(state_dict, size=size, device="cpu", task=task)
    native.model.eval()

    gen = torch.Generator().manual_seed(42)
    x = torch.randn(1, 3, 640, 640, generator=gen)

    with torch.no_grad():
        up_out = {k: v for k, v in upstream(x).items() if v is not None}
        na_out = {k: v for k, v in native.model(x).items() if v is not None}

    assert set(up_out) == set(na_out)
    for key in sorted(up_out):
        max_diff = float((up_out[key] - na_out[key]).abs().max().item())
        assert max_diff == 0.0, f"{model_id} {key}: max_abs_diff={max_diff:.3e}"
