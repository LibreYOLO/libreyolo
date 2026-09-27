"""Manual DetAny3D parity against locally staged upstream reference outputs.

Set LIBREYOLO_DETANY3D_CHECKPOINT, _RUNTIME, _PYTHON, _REFERENCE and
_GROUNDING_CHECKPOINT. The reference JSON sits beside its source images.
No model weights, dataset images or reference-output arrays are redistributed.
"""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from tests.e2e.conftest import require_test_weights

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.detany3d,
    pytest.mark.external_data,
    pytest.mark.network,
]


def _sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def test_upstream_prompt_modes_and_geometry():
    keys = ("CHECKPOINT", "RUNTIME", "PYTHON", "REFERENCE", "GROUNDING_CHECKPOINT")
    paths = {key: os.environ.get("LIBREYOLO_DETANY3D_" + key) for key in keys}
    if not all(paths.values()):
        pytest.skip(
            "Set LIBREYOLO_DETANY3D_CHECKPOINT, _RUNTIME, _PYTHON, _REFERENCE and _GROUNDING_CHECKPOINT."
        )
    checkpoint = Path(
        require_test_weights(str(Path(paths["CHECKPOINT"]).expanduser().resolve()))
    )
    reference_path = Path(paths["REFERENCE"]).expanduser().resolve()
    reference = json.loads(reference_path.read_text())
    assert reference["schema_version"] == 1
    assert reference["upstream_revision"] == "10e484be0837d80aa33cff6ed95a9be834a3f794"
    assert reference["checkpoint_sha256"] == _sha256(checkpoint)
    assert reference["grounding_sha256"] == _sha256(Path(paths["GROUNDING_CHECKPOINT"]))
    assert reference["config_sha256"] == _sha256(
        Path(paths["RUNTIME"]) / "detect_anything/configs/demo.yaml"
    )
    cases = reference["cases"]
    assert len(cases) == 9
    assert any(case["crop"][0] > 0 for case in cases)
    assert any(not case["labels"] for case in cases)
    assert any(
        "bboxes" in case["prompt"] and "text" in case["prompt"] for case in cases
    )
    assert any("points" in case["prompt"] for case in cases)
    from libreyolo import LibreDetAny3D

    model = None
    conf = None
    try:
        for case in cases:
            if model is None or conf != case["conf"]:
                if model is not None:
                    model.close()
                conf = case["conf"]
                model = LibreDetAny3D(
                    checkpoint,
                    runtime_path=paths["RUNTIME"],
                    runtime_python=paths["PYTHON"],
                    grounding_checkpoint=paths["GROUNDING_CHECKPOINT"],
                    device="cpu",
                    conf=conf,
                )
                model.set_classes(["car", "person"])
            assert Path(case["image"]).name == case["image"]
            image = reference_path.parent / case["image"]
            assert _sha256(image) == case["image_sha256"]
            result = model(image, **case["prompt"])
            labels = [result.names[int(index)] for index in result.boxes.cls]
            assert labels == case["labels"]
            np.testing.assert_allclose(
                result.boxes.conf, case["scores"], atol=1e-5, rtol=1e-5
            )
            np.testing.assert_allclose(
                result.boxes.xyxy,
                np.asarray(case["boxes"]).reshape(-1, 4),
                atol=0.05,
                rtol=1e-5,
            )
            np.testing.assert_allclose(
                result.boxes3d.xyz,
                np.asarray(case["centers"]).reshape(-1, 3),
                atol=0.002,
                rtol=1e-4,
            )
            np.testing.assert_allclose(
                result.boxes3d.dimensions,
                np.asarray(case["dimensions"]).reshape(-1, 3),
                atol=0.001,
                rtol=1e-4,
            )
            np.testing.assert_allclose(
                result.boxes3d.intrinsics, case["intrinsics"], atol=0.01, rtol=1e-5
            )
            if len(result):
                distance = np.linalg.norm(
                    result.boxes3d.corners[:, :, None]
                    - np.asarray(case["corners"])[:, None],
                    axis=-1,
                )
                assert distance.min(1).max() < 0.005
                assert distance.min(2).max() < 0.005
    finally:
        if model is not None:
            model.close()
