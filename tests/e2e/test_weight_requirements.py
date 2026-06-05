"""Tests for e2e test-weight preflight behavior."""

import pytest

from .conftest import require_test_weights

pytestmark = pytest.mark.e2e


def test_public_libreyolo_weight_path_can_autodownload(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    assert require_test_weights("weights/LibreDEIMn.pt", expected_family="deim") == (
        "weights/LibreDEIMn.pt"
    )


def test_yolonas_native_filename_can_autodownload(monkeypatch, tmp_path):
    # YOLO-NAS weights live on Deci's public CDN, not the LibreYOLO HF org, so
    # their native filename must resolve a download route rather than skip.
    monkeypatch.chdir(tmp_path)

    assert require_test_weights("downloads/yolonas/yolo_nas_s_coco.pth") == (
        "downloads/yolonas/yolo_nas_s_coco.pth"
    )


def test_missing_local_only_weight_path_skips(monkeypatch, tmp_path):
    # A path-style weight with no public download route at all (L2CS/Gaze360 is
    # not mirrored and has no plain-HTTP URL) must still skip cleanly.
    monkeypatch.chdir(tmp_path)

    with pytest.raises(pytest.skip.Exception):
        require_test_weights("downloads/l2cs/LibreL2CSr50.pt")


def test_missing_unavailable_libreyolo_weight_path_skips(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    with pytest.raises(pytest.skip.Exception):
        require_test_weights("weights/LibreDAMOYOLOl.pt", expected_family="damoyolo")
