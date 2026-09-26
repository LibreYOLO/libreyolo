"""Rectangular imgsz: supported where upstream supports it, a clear error elsewhere.

RF-DETR, D-FINE, DEIM and EC predict at ``(h, w)`` like their upstreams
(RF-DETR ``predict(shape=(h, w))``; the D-FINE line's ``eval_spatial_size`` is
``[h, w]``): a plain resize with normalized boxes. Before this they crashed in
preprocessing with a ``TypeError`` or an OpenCV resize error.

DEIMv2 and TinyFormer still validate ``imgsz`` as a single int, so
``predict()`` and ``val()`` reject a rectangle with a clear error instead of
that ``TypeError``.
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
from libreyolo.preprocess.deim import preprocess_numpy as deim_preprocess
from libreyolo.preprocess.dfine import preprocess_numpy as dfine_preprocess
from libreyolo.preprocess.ec import preprocess_numpy as ec_preprocess
from libreyolo.preprocess.rfdetr import preprocess_numpy as rfdetr_preprocess
from libreyolo.utils.image_size import reject_rectangular_imgsz

pytestmark = pytest.mark.unit

SQUARE_ONLY = [
    pytest.param(LibreDEIMv2, "atto", id="deimv2"),
    pytest.param(LibreTinyFormer, "s", id="tinyformer"),
]

# RF-DETR builds offline from an empty state dict; ``None`` would download
# its pretrained backbone, which the PR gate blocks.
RECTANGULAR = [
    pytest.param(LibreRFDETR, "n", (256, 448), id="rfdetr"),
    pytest.param(LibreDFINE, "n", (320, 640), id="dfine"),
    pytest.param(LibreDEIM, "n", (320, 640), id="deim"),
    pytest.param(LibreEC, "s", (320, 640), id="ec"),
]


@pytest.mark.parametrize("cls,size", SQUARE_ONLY)
def test_square_only_families_declare_their_calls(cls, size):
    assert cls.SQUARE_IMGSZ_CALLS == frozenset({"predict", "val"})


@pytest.mark.parametrize("cls,size", SQUARE_ONLY)
def test_rectangular_predict_raises_clear_error(cls, size):
    model = cls(None, size=size, device="cpu")
    with pytest.raises(ValueError, match=r"predict\(\) does not support rectangular"):
        model(Image.new("RGB", (64, 48)), imgsz=(320, 640))


@pytest.mark.parametrize("cls,size", SQUARE_ONLY)
def test_rectangular_val_raises_clear_error(cls, size):
    model = cls(None, size=size, device="cpu")
    # Raised before the dataset is resolved, so no data is needed.
    with pytest.raises(ValueError, match=r"val\(\) does not support rectangular"):
        model.val(data="missing.yaml", imgsz=(320, 640))


def test_square_and_scalar_sizes_pass():
    model = LibreDEIMv2(None, size="atto", device="cpu")
    for imgsz in (320, (320, 320), [320, 320]):
        reject_rectangular_imgsz(model, imgsz, "predict")
        reject_rectangular_imgsz(model, imgsz, "val")


@pytest.mark.parametrize("cls,size,imgsz", RECTANGULAR)
def test_rectangular_families_predict_at_height_width(cls, size, imgsz):
    assert cls.SQUARE_IMGSZ_CALLS == frozenset()
    model = cls({} if cls is LibreRFDETR else None, size=size, device="cpu")
    result = model(Image.new("RGB", (64, 48)), imgsz=imgsz)
    assert result.orig_shape == (48, 64)


def test_rfdetr_rectangular_imgsz_must_fit_the_patch_grid():
    model = LibreRFDETR({}, size="n", device="cpu")
    with pytest.raises(ValueError, match="not divisible by 32"):
        model(Image.new("RGB", (64, 48)), imgsz=(300, 448))


@pytest.mark.parametrize(
    "preprocess",
    [rfdetr_preprocess, dfine_preprocess, deim_preprocess, ec_preprocess],
    ids=["rfdetr", "dfine", "deim", "ec"],
)
@pytest.mark.parametrize(
    "input_size,expected", [(384, (3, 384, 384)), ((256, 448), (3, 256, 448))]
)
def test_preprocess_resizes_to_height_width(preprocess, input_size, expected):
    chw, ratio = preprocess(np.zeros((48, 64, 3), dtype=np.uint8), input_size)
    assert chw.shape == expected
    assert ratio == 1.0
