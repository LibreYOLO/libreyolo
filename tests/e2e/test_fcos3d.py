"""Replay final-detection parity against locally staged upstream CPU outputs.

Set LIBREYOLO_FCOS3D_CHECKPOINT and LIBREYOLO_FCOS3D_REFERENCE. The latter
is the JSON bundle described in ADR 0023, beside its six source images.
Reference outputs and dataset images are not redistributed by LibreYOLO.
"""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from tests.e2e.conftest import require_test_weights

pytestmark = [pytest.mark.e2e, pytest.mark.fcos3d, pytest.mark.external_data]


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_upstream_cpu_detections():
    checkpoint_ref = os.environ.get("LIBREYOLO_FCOS3D_CHECKPOINT")
    reference_ref = os.environ.get("LIBREYOLO_FCOS3D_REFERENCE")
    if not checkpoint_ref or not reference_ref:
        pytest.skip("Set LIBREYOLO_FCOS3D_CHECKPOINT and LIBREYOLO_FCOS3D_REFERENCE.")
    checkpoint = Path(
        require_test_weights(str(Path(checkpoint_ref).expanduser().resolve()))
    )
    reference_path = Path(reference_ref).expanduser().resolve()
    reference = json.loads(reference_path.read_text())
    assert reference["schema_version"] == 1
    assert reference["checkpoint_sha256"] == _sha256(checkpoint)
    assert (
        reference["reference"]["mmdetection3d"]
        == "fe25f7a51d36e3702f961e198894580d83c4387b"
    )
    assert reference["reference"]["mmcv"] == "a8073c74bf83d62ec36a103f835faa4837fb6585"
    cameras = {
        "CAM_FRONT",
        "CAM_FRONT_LEFT",
        "CAM_FRONT_RIGHT",
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_BACK_RIGHT",
    }
    cases = reference["cases"]
    assert len(cases) == 18
    assert {(case["camera"], case["conf"]) for case in cases} == {
        (camera, conf) for camera in cameras for conf in (0.05, 0.15, 0.3)
    }

    from libreyolo import LibreFCOS3D

    model = LibreFCOS3D(checkpoint, device="cpu")
    for case in cases:
        image = reference_path.parent / case["image"]
        assert _sha256(image) == case["image_sha256"]
        result = model(
            image,
            intrinsics=case["intrinsics"],
            conf=case["conf"],
            iou=0.8,
            max_det=200,
        )
        expected = case["expected"]
        assert len(result) == case["count"]
        np.testing.assert_array_equal(result.boxes.cls, expected["labels"])
        np.testing.assert_allclose(
            result.boxes.conf, expected["scores"], atol=1e-5, rtol=1e-5
        )
        np.testing.assert_allclose(
            result.boxes3d.xyz,
            np.asarray(expected["centers"]).reshape(-1, 3),
            atol=2e-4,
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            result.boxes3d.dimensions,
            np.asarray(expected["dimensions"]).reshape(-1, 3),
            atol=1e-4,
            rtol=1e-5,
        )
        if len(result):
            corners = np.asarray(expected["corners"])
            distance = np.linalg.norm(
                result.boxes3d.corners[:, :, None] - corners[:, None], axis=-1
            )
            assert distance.min(axis=1).max() < 3e-4
            assert distance.min(axis=2).max() < 3e-4
