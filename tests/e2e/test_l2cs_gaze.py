"""Non-gated L2CS / Gaze360 inference check.

L2CS weights are derived from the Gaze360 dataset, whose license forbids
redistribution, so they are not mirrored on the LibreYOLO HF org and have no
plain-HTTP auto-download route. This suite is therefore deliberately NOT part of
the gated ``general_nightly`` matrix (where a skip is turned into a failure):
it runs when the checkpoint is staged locally and skips cleanly otherwise, so
gaze inference stays covered without making it a non-provisionable hard gate.
"""

from pathlib import Path

import pytest
import torch
from PIL import Image

from .conftest import cuda_cleanup

# Marked with the per-family ``l2cs`` marker (not ``general_nightly``) so
# targeted family jobs / `pytest -m l2cs` include this gaze check, while the
# gated nightly never turns its weight-absent skip into a failure.
pytestmark = [pytest.mark.e2e, pytest.mark.l2cs]

_L2CS_WEIGHTS = "LibreL2CSr50.pt"


def _staged_l2cs_weights() -> str | None:
    for candidate in (Path(_L2CS_WEIGHTS), Path("weights") / _L2CS_WEIGHTS):
        if candidate.exists():
            return str(candidate)
    return None


def _tensor(data):
    return (
        data.detach().cpu()
        if isinstance(data, torch.Tensor)
        else torch.as_tensor(data)
    )


def test_l2cs_gaze_inference_is_stable():
    weights = _staged_l2cs_weights()
    if weights is None:
        pytest.skip(
            "L2CS/Gaze360 weights not staged locally (non-redistributable); "
            f"place {_L2CS_WEIGHTS} in ./ or ./weights to run this gaze check."
        )

    from libreyolo import LibreL2CS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LibreL2CS(weights, size="r50", device=device)
    try:
        image = Image.new("RGB", (96, 96), color=(128, 128, 128))
        kwargs = {"face_boxes": [(8, 8, 88, 88)]}
        first = model(image, **kwargs)
        second = model(image, **kwargs)

        assert first.gaze is not None, "l2cs did not return gaze output"
        assert second.gaze is not None, "l2cs did not return gaze output"
        assert len(first.gaze) == 1
        assert len(second.gaze) == 1
        torch.testing.assert_close(
            _tensor(first.gaze.data),
            _tensor(second.gaze.data),
            rtol=1e-5,
            atol=1e-5,
        )
    finally:
        del model
        cuda_cleanup()
