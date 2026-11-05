"""Falcon Perception preprocessing: canvas buckets, resize, normalisation, prompt."""

from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from libreyolo.models.falcon_perception.preprocess import (
    anyup_canvas_size,
    build_prompt,
    preprocess_image,
    resize_image_if_necessary,
    tokenize_prompt,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "h, w, canvas, out",
    [(100, 200, 256, 128), (300, 300, 512, 256), (700, 768, 768, 384), (1024, 900, 1024, 512)],
)
def test_anyup_canvas_buckets(h, w, canvas, out):
    got_canvas, (oh, ow) = anyup_canvas_size(h, w)
    assert (got_canvas, oh, ow) == (canvas, out, out)
    assert got_canvas >= min(max(h, w), 1024)


def test_resize_and_preprocess_contract():
    small = resize_image_if_necessary(Image.new("RGB", (100, 50)), 256, 1024)
    assert min(small.size) >= 256
    big = resize_image_if_necessary(Image.new("RGB", (3000, 1500)), 256, 1024)
    assert max(big.size) == 1024
    px = preprocess_image(Image.new("RGB", (301, 203), (255, 0, 128)))
    assert px.dtype == np.float32 and px.shape[0] == 1
    assert px.shape[1] % 16 == 0 and px.shape[2] % 16 == 0 and px.shape[3] == 3
    assert px.min() >= -1.0 and px.max() <= 1.0


def test_prompt_template():
    assert build_prompt("cat") == (
        "<|image|>Segment these expressions in the image:<|start_of_query|>cat<|REF_SEG|>"
    )


IMG = 227
FAKE_TOK = SimpleNamespace(
    image_token="<|image|>",
    image_token_id=IMG,
    image_cls_token_id=244,
    image_reg_1_token_id=245,
    image_reg_2_token_id=246,
    image_reg_3_token_id=247,
    image_reg_4_token_id=248,
    end_of_image_token_id=230,
    bos_id=None,
    encode=lambda text: [5] * max(1, len(text) // 10),
)


def test_full_size_square_image_keeps_its_block():
    # 1024x1024 is 4096 patch tokens: more than a 4096 prompt budget, well inside a real one.
    px = preprocess_image(resize_image_if_necessary(Image.new("RGB", (1024, 1024))))
    ids = tokenize_prompt(build_prompt("cat"), px[0], FAKE_TOK, 16, 8192 - 2048)
    assert int((ids == IMG).sum()) == 4096


def test_image_that_does_not_fit_raises_instead_of_vanishing():
    px = preprocess_image(resize_image_if_necessary(Image.new("RGB", (1024, 1024))))
    with pytest.raises(ValueError, match="image needs"):
        tokenize_prompt(build_prompt("cat"), px[0], FAKE_TOK, 16, 4096)
