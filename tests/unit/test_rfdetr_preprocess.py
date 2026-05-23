from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

pytestmark = pytest.mark.unit


def test_rfdetr_native_preprocess_matches_numpy_helper():
    from libreyolo.models.rfdetr.model import LibreRFDETR
    from libreyolo.models.rfdetr.utils import preprocess_numpy

    rgb = np.random.default_rng(0).integers(0, 256, (37, 53, 3), dtype=np.uint8)
    model = LibreRFDETR(model_path={}, size="n", device="cpu")

    tensor, _, original_size, ratio = model._preprocess(
        Image.fromarray(rgb, mode="RGB"),
        input_size=64,
    )
    expected, _ = preprocess_numpy(rgb, 64)

    assert original_size == (53, 37)
    assert ratio == 1.0
    torch.testing.assert_close(tensor[0], torch.from_numpy(expected), atol=0, rtol=0)


def test_rfdetr_val_preprocessor_matches_numpy_helper_and_scales_targets():
    from libreyolo.models.rfdetr.utils import preprocess_numpy
    from libreyolo.validation.preprocessors import RFDETRValPreprocessor

    bgr = np.random.default_rng(1).integers(0, 256, (37, 53, 3), dtype=np.uint8)
    rgb = bgr[:, :, ::-1]
    targets = np.array([[5, 7, 40, 30, 2]], dtype=np.float32)
    preprocessor = RFDETRValPreprocessor((64, 64), max_labels=4)

    image, labels = preprocessor(bgr, targets, (64, 64))
    expected_image, _ = preprocess_numpy(rgb, 64)

    np.testing.assert_array_equal(image, expected_image)
    np.testing.assert_allclose(
        labels[0],
        np.array(
            [
                5 * 64 / 53,
                7 * 64 / 37,
                40 * 64 / 53,
                30 * 64 / 37,
                2,
            ],
            dtype=np.float32,
        ),
    )
    assert labels[1:].sum() == 0


def test_rfdetr_val_preprocessor_requests_original_image():
    from libreyolo.validation.preprocessors import RFDETRValPreprocessor

    preprocessor = RFDETRValPreprocessor((64, 64), max_labels=4)

    assert preprocessor.wants_unresized_image is True
