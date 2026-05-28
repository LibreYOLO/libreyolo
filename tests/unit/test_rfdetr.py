"""Unit tests for LibreRFDETR filename-based size detection."""

import pytest

from libreyolo.models.rfdetr.model import LibreRFDETR

pytestmark = pytest.mark.unit


class TestRFDETRSizeDetectionFromFilename:
    """Test LibreRFDETR.detect_size_from_filename."""

    @pytest.mark.parametrize("size", ["n", "s", "m", "l"])
    def test_all_detect_sizes_via_libre_prefix(self, size):
        assert LibreRFDETR.detect_size_from_filename(f"LibreRFDETR{size}.pt") == size

    @pytest.mark.parametrize(
        "filename,expected",
        [
            ("rf-detr-nano.pth", "n"),
            ("rf-detr-small.pth", "s"),
            ("rf-detr-medium.pth", "m"),
            ("rf-detr-large-2026.pth", "l"),
            ("rf-detr-seg-nano.pt", "n"),
            ("rf-detr-seg-small.pt", "s"),
            ("rf-detr-seg-medium.pt", "m"),
            ("rf-detr-seg-large.pt", "l"),
            ("rf-detr-seg-xlarge.pt", "x"),
            ("rf-detr-seg-xxlarge.pt", "xx"),
        ],
    )
    def test_upstream_filenames(self, filename, expected):
        assert LibreRFDETR.detect_size_from_filename(filename) == expected

    def test_unknown_filename_returns_none(self):
        assert LibreRFDETR.detect_size_from_filename("yolov8n.pt") is None

    def test_rtdetr_filename_returns_none(self):
        assert LibreRFDETR.detect_size_from_filename("rtdetr_r50vd_6ep_coco.pth") is None
