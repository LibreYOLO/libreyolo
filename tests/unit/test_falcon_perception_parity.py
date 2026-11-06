"""Byte parity of the Falcon Perception port against the upstream engine.

Opt in by pointing ``FALCON_PERCEPTION_UPSTREAM`` at a checkout of
tiiuae/Falcon-Perception (commit c457916) and ``FALCON_PERCEPTION_WEIGHTS_600M`` /
``FALCON_PERCEPTION_WEIGHTS_300M`` at the two snapshot directories. The harness
(weights/parity_falcon_perception.py) compares every forward pass bitwise plus
generated tokens, raw box floats and mask bytes, on CPU in fp32.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.external_data]

REPO = Path(__file__).resolve().parents[2]
UPSTREAM = os.environ.get("FALCON_PERCEPTION_UPSTREAM")
W600 = os.environ.get("FALCON_PERCEPTION_WEIGHTS_600M")
W300 = os.environ.get("FALCON_PERCEPTION_WEIGHTS_300M")


@pytest.mark.skipif(
    not (UPSTREAM and W600 and W300),
    reason="set FALCON_PERCEPTION_UPSTREAM and FALCON_PERCEPTION_WEIGHTS_{600M,300M}",
)
def test_bitwise_parity_with_upstream(tmp_path):
    from PIL import Image

    cases = []
    for name, query in (("parkour", "person"), ("guggenheim-bilbao", "building")):
        img = tmp_path / f"{name}.jpg"
        Image.open(REPO / "libreyolo" / "assets" / f"{name}.jpg").convert("RGB").resize(
            (256, 256)
        ).save(img)
        cases.append([str(img), query])
    out = tmp_path / "out"
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO / "weights" / "parity_falcon_perception.py"),
            f"--upstream={UPSTREAM}",
            f"--weights600={W600}",
            f"--weights300={W300}",
            f"--out={out}",
            f"--cases={json.dumps(cases)}",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    summary = json.loads((out / "summary.json").read_text())
    assert summary["passed"] == summary["cases"] == summary["passed_bitwise_logits"]
