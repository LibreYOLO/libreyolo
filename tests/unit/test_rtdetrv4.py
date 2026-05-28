"""Unit tests for LibreRTDETRv4.detect_size_from_filename."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


class TestRTDETRv4SizeDetectionFromFilename:
    """Test LibreRTDETRv4.detect_size_from_filename."""

    @pytest.mark.parametrize("size", ["s", "m", "l", "x"])
    def test_all_sizes_via_libre_prefix(self, size):
        from libreyolo.models.rtdetrv4.model import LibreRTDETRv4

        assert LibreRTDETRv4.detect_size_from_filename(f"LibreRTDETRv4{size}.pt") == size

    @pytest.mark.parametrize(
        "filename,expected",
        [
            ("rtv4_s_coco.pth", "s"),
            ("rtv4_m_coco.pth", "m"),
            ("rtv4_hgnetv2_s_coco.pth", "s"),
            ("rtv4_hgnetv2_l_coco.pth", "l"),
            ("rtv4_hgnetv2_x_coco.pth", "x"),
        ],
    )
    def test_upstream_v4_checkpoints(self, filename, expected):
        from libreyolo.models.rtdetrv4.model import LibreRTDETRv4

        assert LibreRTDETRv4.detect_size_from_filename(filename) == expected

    def test_no_false_positive_for_two_char_suffix(self):
        # "sl" must not match "s" or "l"
        from libreyolo.models.rtdetrv4.model import LibreRTDETRv4

        assert LibreRTDETRv4.detect_size_from_filename("rtv4_hgnetv2_sl_coco.pth") is None

    def test_unknown_filename_returns_none(self):
        from libreyolo.models.rtdetrv4.model import LibreRTDETRv4

        assert LibreRTDETRv4.detect_size_from_filename("yolov8n.pt") is None
