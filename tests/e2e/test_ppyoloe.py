"""PP-YOLOE public route: image sources, batching, conversion and Results."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from tests.e2e.conftest import PPYOLOE_SIZES, require_test_weights

pytestmark = [pytest.mark.e2e, pytest.mark.ppyoloe]

WEIGHTS = "LibrePPYOLOEs.pt"


@pytest.fixture(scope="module")
def model():
    from libreyolo import LibreYOLO

    weights = require_test_weights(WEIGHTS, expected_family="ppyoloe")
    return LibreYOLO(weights)


def test_catalog_covers_every_released_size():
    assert PPYOLOE_SIZES == ["s", "m", "l", "x"]


def test_public_api_shape(model, sample_image):
    assert model.family == "ppyoloe"
    assert model.task == "detect"
    assert model.nb_classes == 80
    assert model.names[0] == "person"

    result = model.predict(sample_image, conf=0.5)
    assert not isinstance(result, list)
    assert result.boxes is not None
    assert result.masks is None and result.keypoints is None
    assert result.orig_shape is not None
    assert result.path is not None
    assert len(result.boxes) > 0

    boxes = np.asarray(result.boxes.xyxy)
    height, width = result.orig_shape[:2]
    assert (boxes[:, 0] >= 0).all() and (boxes[:, 1] >= 0).all()
    assert (boxes[:, 2] <= width + 1e-3).all() and (boxes[:, 3] <= height + 1e-3).all()
    assert (boxes[:, 2] > boxes[:, 0]).all() and (boxes[:, 3] > boxes[:, 1]).all()
    # Class probability is the confidence: there is no objectness multiplier.
    assert (np.asarray(result.boxes.conf) <= 1.0).all()


@pytest.mark.parametrize("source", ["path", "pil", "numpy", "tensor"])
def test_image_source_matrix(model, sample_image, source):
    image = Image.open(sample_image).convert("RGB")
    payload = {
        "path": sample_image,
        "pil": image,
        "numpy": np.array(image),
        "tensor": torch.from_numpy(np.array(image)).permute(2, 0, 1),
    }[source]
    # A single image yields a single Results, not a list.
    result = model.predict(payload, conf=0.5)
    assert not isinstance(result, list)
    assert len(result.boxes) > 0


def test_list_input_preserves_order_and_count(model, sample_image):
    image = Image.open(sample_image).convert("RGB")
    small = image.resize((320, 240))
    results = model.predict([sample_image, small], conf=0.5)
    assert len(results) == 2
    assert results[0].orig_shape[:2] == (image.height, image.width)
    assert results[1].orig_shape[:2] == (240, 320)


def test_conf_and_iou_overrides_take_effect(model, sample_image):
    strict = model.predict(sample_image, conf=0.9)
    loose = model.predict(sample_image, conf=0.05)
    assert len(loose.boxes) >= len(strict.boxes)
    if len(loose.boxes):
        assert float(np.asarray(loose.boxes.conf).min()) >= 0.05
    if len(strict.boxes):
        assert float(np.asarray(strict.boxes.conf).min()) >= 0.9


def test_non_square_source_is_restored_to_the_original_canvas(model, sample_image):
    """Stretch resize means x and y unscale by different factors."""
    image = Image.open(sample_image).convert("RGB").resize((900, 300))
    result = model.predict(image, conf=0.25)
    assert result.orig_shape[:2] == (300, 900)
    boxes = np.asarray(result.boxes.xyxy)
    if boxes.size:
        assert boxes[:, [0, 2]].max() <= 900 + 1e-3
        assert boxes[:, [1, 3]].max() <= 300 + 1e-3


def test_save_writes_an_annotated_image(model, sample_image, tmp_path):
    """The UI result card renders whatever save=True writes, so exercise that."""
    out = tmp_path / "annotated"
    result = model(sample_image, conf=0.5, save=True, output_path=str(out))
    if isinstance(result, list):
        result = result[0]
    assert len(result.boxes) > 0

    written = sorted(p for p in tmp_path.rglob("*") if p.suffix.lower() in {".jpg", ".png"})
    assert written, "save=True wrote no annotated image"
    annotated = np.array(Image.open(written[0]).convert("RGB"))
    original = np.array(Image.open(sample_image).convert("RGB"))
    assert annotated.shape == original.shape
    assert not np.array_equal(annotated, original)


def test_released_checkpoint_converts_and_reloads(tmp_path):
    """Native download route -> lean LibreYOLO checkpoint -> identical results."""
    from libreyolo import LibreYOLO
    from libreyolo.models.ppyoloe.convert import (
        convert_upstream,
        unwrap_ppyoloe_checkpoint,
    )
    from libreyolo.utils.serialization import (
        load_untrusted_torch_file,
        wrap_libreyolo_checkpoint,
    )

    weights = require_test_weights(WEIGHTS, expected_family="ppyoloe")
    native = LibreYOLO(weights, device="cpu")

    raw = load_untrusted_torch_file(
        str(native.model_path), map_location="cpu", context="ppyoloe conversion test"
    )
    state = convert_upstream(unwrap_ppyoloe_checkpoint(raw))
    wrapped = wrap_libreyolo_checkpoint(
        state, model_family="ppyoloe", size="s", task="detect", nc=80, imgsz=640
    )
    # A lean checkpoint carries no optimizer or scaler state.
    assert "optimizer" not in wrapped and "scaler" not in wrapped

    path = tmp_path / "LibrePPYOLOEs.pt"
    torch.save(wrapped, path)

    reloaded = LibreYOLO(str(path), device="cpu")
    assert reloaded.family == "ppyoloe"
    assert reloaded.size == "s"
    assert reloaded.nb_classes == 80

    x = torch.zeros(1, 3, 640, 640)
    with torch.no_grad():
        (a_boxes, a_scores), _ = native.model.eval()(x)
        (b_boxes, b_scores), _ = reloaded.model.eval()(x)
    assert torch.equal(a_boxes, b_boxes)
    assert torch.equal(a_scores, b_scores)
