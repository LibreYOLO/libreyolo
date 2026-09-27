"""External tensor parity for the pinned QuickSRNet Medium 2x source."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.unit,
    pytest.mark.quicksrnet,
    pytest.mark.external_data,
]


@pytest.mark.skipif(
    not (
        os.environ.get("QUICKSRNET_UPSTREAM_SOURCE")
        and os.environ.get("QUICKSRNET_OFFICIAL_CKPT")
    ),
    reason=(
        "Set QUICKSRNET_UPSTREAM_SOURCE and QUICKSRNET_OFFICIAL_CKPT to run "
        "parity against the pinned BSD upstream."
    ),
)
def test_tensor_parity_against_pinned_upstream():
    script = Path(__file__).resolve().parents[2] / "weights" / "parity_quicksrnet.py"
    weights_dir = str(script.parent)
    if weights_dir not in sys.path:
        sys.path.insert(0, weights_dir)
    spec = importlib.util.spec_from_file_location("parity_quicksrnet", script)
    assert spec is not None and spec.loader is not None
    parity = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parity)
    maximum_difference = parity.compare(
        os.environ["QUICKSRNET_UPSTREAM_SOURCE"],
        os.environ["QUICKSRNET_OFFICIAL_CKPT"],
    )
    assert maximum_difference == 0.0
