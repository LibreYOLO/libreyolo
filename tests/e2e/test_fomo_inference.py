"""
LibreFOMO pretrained-checkpoint inference tests.

  1. Loads each pretrained variant (s/m/l) via the ``LibreYOLO`` factory.
  2. Verifies checkpoint schema (``validate_checkpoint_metadata``).
  3. Runs inference twice on a synthetic image and checks:
       - ``result.boxes is None``
       - ``result.points is not None``
       - Outputs are non-NaN / non-Inf
       - Point count is stable between the two passes
  4. Checks the public API: ``result.summary()``, ``result.to_json()``,
     ``result.points.xy``, ``result.points.xyn``.

Marker: ``fomo``  (gate with ``-m fomo`` or ``-m e2e and fomo``)

These tests require a network connection to download the pretrained checkpoints
from HuggingFace the first time they run.
"""

from pathlib import Path

import pytest
import torch
from PIL import Image
import numpy as np

from libreyolo import LibreYOLO
from libreyolo.utils.serialization import validate_checkpoint_metadata

from .conftest import cuda_cleanup, requires_cuda

pytestmark = [pytest.mark.e2e, pytest.mark.fomo]


# ---------------------------------------------------------------------------
# Pretrained checkpoint matrix
# ---------------------------------------------------------------------------

LIBREFOMO_CHECKPOINTS = [
    ("librefomo", "s", "LibreFOMOs.pt"),
    ("librefomo", "m", "LibreFOMOm.pt"),
    ("librefomo", "l", "LibreFOMOl.pt"),
]

LIBREFOMO_EXPECTED_IMGSZ = {"s": 96, "m": 192, "l": 224}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _synthetic_image(size: int) -> Image.Image:
    """Return a small synthetic RGB image with a bright patch."""
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    cx, cy = size // 2, size // 2
    r = max(4, size // 12)
    arr[cy - r : cy + r, cx - r : cx + r] = [200, 180, 160]
    return Image.fromarray(arr)


def _assert_point_output_valid(family: str, result, pass_num: int):
    """Assert the basic contracts of a FOMO result object."""
    assert result.boxes is None, (
        f"{family} pass {pass_num}: expected no boxes, got {result.boxes}"
    )
    assert result.points is not None, (
        f"{family} pass {pass_num}: result.points must not be None"
    )
    data = result.points.data
    assert data.ndim == 2, f"points.data must be 2-D, got shape {data.shape}"
    assert data.shape[1] == 4, (
        f"point rows must be (x, y, cls, conf), got shape {tuple(data.shape)}"
    )
    if data.shape[0] > 0:
        assert torch.isfinite(data).all(), (
            f"{family} pass {pass_num}: non-finite values in point output"
        )
    # Public API smoke
    assert result.points.xy.shape[1] == 2
    assert result.points.xyn.shape[1] == 2
    assert isinstance(result.summary(), list)
    assert isinstance(result.to_json(), str)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "family,size,weights",
    [
        pytest.param(f, s, w, marks=pytest.mark.fomo, id=f"{f}-{s}")
        for f, s, w in LIBREFOMO_CHECKPOINTS
    ],
)
def test_fomo_pretrained_checkpoint_schema(family, size, weights):
    """Download and validate the checkpoint schema for each pretrained variant.

    Checks all required LibreYOLO v1.0 metadata keys without running inference
    (fast — CPU only, weight download only done once).
    """
    # Force fresh download — LibreYOLO will cache under weights/
    weights_path = Path("weights") / weights
    if weights_path.exists():
        weights_path.unlink()

    model = LibreYOLO(weights, device="cpu")

    # Family / task
    assert model.family == "librefomo", f"Expected librefomo, got {model.family!r}"
    assert model.task == "point", f"Expected task='point', got {model.task!r}"
    assert model.size == size, f"Expected size={size!r}, got {model.size!r}"
    assert model.input_size == LIBREFOMO_EXPECTED_IMGSZ[size]
    assert model.model.training is False, "Model must be in eval mode after loading"

    # Checkpoint schema
    assert weights_path.exists(), f"Weights not downloaded to {weights_path}"
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=True)
    errors = validate_checkpoint_metadata(ckpt, strict=False)
    assert errors == [], f"Checkpoint schema errors for {weights}: {errors}"

    assert ckpt["model_family"] == "librefomo"
    assert ckpt["task"] == "point"
    assert ckpt["size"] == size
    assert ckpt["imgsz"] == LIBREFOMO_EXPECTED_IMGSZ[size]
    assert ckpt["nc"] == 1
    assert isinstance(ckpt["names"], dict) and ckpt["names"][0] == "person"

    print(
        f"\n  {weights}: schema OK — "
        f"task={ckpt['task']!r}, size={ckpt['size']!r}, "
        f"imgsz={ckpt['imgsz']}, nc={ckpt['nc']}",
        flush=True,
    )
    del model
    cuda_cleanup()


@requires_cuda
@pytest.mark.parametrize(
    "family,size,weights",
    [
        pytest.param(f, s, w, marks=pytest.mark.fomo, id=f"{f}-{s}")
        for f, s, w in LIBREFOMO_CHECKPOINTS
    ],
)
def test_fomo_inference_is_stable(family, size, weights):
    """Load pretrained checkpoint and verify stable point-localization output.

    Runs two inference passes on the same synthetic image and asserts:
      - Both passes return ``result.points`` (no boxes)
      - Output count is identical between passes
      - No NaN / Inf values
      - Public API (summary, to_json, xy, xyn) works correctly
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LibreYOLO(weights, device=device)

    image = _synthetic_image(LIBREFOMO_EXPECTED_IMGSZ[size])

    try:
        first = model.predict(image, conf=0.01, max_det=100)
        second = model.predict(image, conf=0.01, max_det=100)

        _assert_point_output_valid(family, first, pass_num=1)
        _assert_point_output_valid(family, second, pass_num=2)

        # Count stability
        assert len(first) == len(second), (
            f"{family}-{size}: point count changed between passes: "
            f"{len(first)} → {len(second)}"
        )

        # Coordinate stability
        if len(first) > 0:
            n = min(5, len(first))
            first_xy = first.points.xy[:n]
            second_xy = second.points.xy[:n]
            torch.testing.assert_close(first_xy, second_xy, rtol=1e-4, atol=1e-4)

        print(
            f"\n  {weights} (device={device}): "
            f"n_points={len(first)}, stable=OK",
            flush=True,
        )
    finally:
        del model
        cuda_cleanup()
