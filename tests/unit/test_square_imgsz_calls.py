"""Square-only families reject a rectangular imgsz with a clear error.

Before this, ``predict(imgsz=(h, w))`` on these families crashed deep inside
preprocessing with a ``TypeError`` or an OpenCV resize error, and
``val(imgsz=(h, w))`` did the same on DEIMv2 and TinyFormer. Rectangular
``val()`` works on D-FINE, DEIM and EC, so it stays allowed there.

RF-DETR predicts at ``(h, w)`` like upstream ``predict(shape=(h, w))``: a plain
resize with normalized boxes, so no call is gated for it.
"""

import numpy as np
import pytest
from PIL import Image

from libreyolo import (
    LibreDEIM,
    LibreDEIMv2,
    LibreDFINE,
    LibreEC,
    LibreRFDETR,
    LibreTinyFormer,
)
from libreyolo.preprocess.rfdetr import preprocess_numpy as rfdetr_preprocess
from libreyolo.utils.image_size import reject_rectangular_imgsz

pytestmark = pytest.mark.unit

CASES = [
    pytest.param(LibreDFINE, "n", {"predict"}, id="dfine"),
    pytest.param(LibreDEIM, "n", {"predict"}, id="deim"),
    pytest.param(LibreEC, "s", {"predict"}, id="ec"),
    pytest.param(LibreDEIMv2, "atto", {"predict", "val"}, id="deimv2"),
    pytest.param(LibreTinyFormer, "s", {"predict", "val"}, id="tinyformer"),
]


@pytest.mark.parametrize("cls,size,calls", CASES)
def test_square_imgsz_calls_are_declared(cls, size, calls):
    assert cls.SQUARE_IMGSZ_CALLS == frozenset(calls)


@pytest.mark.parametrize("cls,size,calls", CASES)
def test_rectangular_predict_raises_clear_error(cls, size, calls):
    model = cls(None, size=size, device="cpu")
    image = Image.new("RGB", (64, 48))
    with pytest.raises(ValueError, match=r"predict\(\) does not support rectangular"):
        model(image, imgsz=(320, 640))


@pytest.mark.parametrize("cls,size,calls", CASES)
def test_rectangular_val_is_gated_only_where_it_crashes(cls, size, calls):
    model = cls(None, size=size, device="cpu")
    if "val" in calls:
        # Raised before the dataset is resolved, so no data is needed.
        with pytest.raises(ValueError, match=r"val\(\) does not support rectangular"):
            model.val(data="missing.yaml", imgsz=(320, 640))
    else:
        reject_rectangular_imgsz(model, (320, 640), "val")


def test_square_and_scalar_sizes_pass():
    model = LibreDEIMv2(None, size="atto", device="cpu")
    for imgsz in (320, (320, 320), [320, 320]):
        reject_rectangular_imgsz(model, imgsz, "predict")
        reject_rectangular_imgsz(model, imgsz, "val")


def test_rfdetr_predicts_at_rectangular_imgsz():
    assert LibreRFDETR.SQUARE_IMGSZ_CALLS == frozenset()
    model = LibreRFDETR(None, size="n", device="cpu")
    result = model(Image.new("RGB", (64, 48)), imgsz=(256, 448))
    assert result.orig_shape == (48, 64)


def test_rfdetr_rectangular_imgsz_must_fit_the_patch_grid():
    model = LibreRFDETR(None, size="n", device="cpu")
    with pytest.raises(ValueError, match="not divisible by 32"):
        model(Image.new("RGB", (64, 48)), imgsz=(300, 448))


@pytest.mark.parametrize(
    "input_size,expected", [(384, (3, 384, 384)), ((256, 448), (3, 256, 448))]
)
def test_rfdetr_preprocess_resizes_to_height_width(input_size, expected):
    chw, ratio = rfdetr_preprocess(np.zeros((48, 64, 3), dtype=np.uint8), input_size)
    assert chw.shape == expected
    assert ratio == 1.0
