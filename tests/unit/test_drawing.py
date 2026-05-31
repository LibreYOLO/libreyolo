"""Unit tests for drawing utilities."""

from PIL import Image, ImageDraw
import pytest

from libreyolo.utils.drawing import _get_font, draw_boxes

pytestmark = pytest.mark.unit


def _text_height(font) -> int:
    img = Image.new("RGB", (800, 200), "white")
    bbox = ImageDraw.Draw(img).textbbox((0, 0), "person: 0.99", font=font)
    return bbox[3] - bbox[1]


def test_get_font_scales_with_requested_size():
    _get_font.cache_clear()
    small = _get_font(12)
    large = _get_font(48)

    assert _text_height(large) > _text_height(small) * 2


def test_draw_boxes_scales_label_on_large_images():
    small = Image.new("RGB", (640, 640), "white")
    large = Image.new("RGB", (2560, 2560), "white")
    box = [[50, 80, 300, 300]]

    small_out = draw_boxes(small, box, [0.99], [0])
    large_out = draw_boxes(large, box, [0.99], [0])

    assert small_out.size == small.size
    assert large_out.size == large.size
    assert _text_height(_get_font(48)) > _text_height(_get_font(12)) * 2
