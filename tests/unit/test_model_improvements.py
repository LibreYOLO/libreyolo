"""Unit tests for the model-improvements PR (yolox eps, picodet cv2, yolo9).

Covers the two non-yolo9 regressions fixed alongside the yolo9 protocol work:

* PicoDet now resizes with ``cv2.INTER_LINEAR`` (was PIL bilinear) — the two
  kernels produce measurably different pixels.
* YOLOX applies BatchNorm ``eps=1e-3`` / ``momentum=0.03`` to *every* size
  (matching the official ``Exp.get_model``), not just nano.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image

pytestmark = pytest.mark.unit


class TestPicoDetCv2Resize:
    def test_cv2_resize_differs_from_pil_bilinear(self):
        """PicoDet preprocess must use cv2, whose output != PIL bilinear."""
        from libreyolo.models.picodet.utils import (
            IMAGENET_MEAN,
            IMAGENET_STD,
            preprocess_numpy,
        )

        rng = np.random.default_rng(0)
        img = rng.integers(0, 256, size=(73, 91, 3), dtype=np.uint8)

        chw, ratio = preprocess_numpy(img, input_size=64)
        assert ratio == 1.0
        assert chw.shape == (3, 64, 64)

        # Reconstruct the OLD PIL-bilinear path and confirm it diverges.
        pil = Image.fromarray(img).resize((64, 64), Image.Resampling.BILINEAR)
        pil_arr = np.array(pil, dtype=np.float32)
        pil_arr -= np.array(IMAGENET_MEAN, dtype=np.float32)
        pil_arr /= np.array(IMAGENET_STD, dtype=np.float32)
        pil_chw = pil_arr.transpose(2, 0, 1)

        assert not np.allclose(chw, pil_chw, atol=1e-3)

        # And confirm cv2 is exactly what preprocess_numpy produced.
        import cv2

        cv2_arr = cv2.resize(
            img, (64, 64), interpolation=cv2.INTER_LINEAR
        ).astype(np.float32)
        cv2_arr -= np.array(IMAGENET_MEAN, dtype=np.float32)
        cv2_arr /= np.array(IMAGENET_STD, dtype=np.float32)
        np.testing.assert_allclose(chw, cv2_arr.transpose(2, 0, 1), rtol=0, atol=1e-5)


class TestYOLOXBatchNormEps:
    @pytest.mark.parametrize("size", ["t", "s", "m", "l", "x"])
    def test_batchnorm_eps_applied_to_all_sizes(self, size):
        """Every YOLOX size must get BN eps=1e-3 / momentum=0.03, not just nano."""
        from libreyolo.models.yolox.model import LibreYOLOX

        model = LibreYOLOX(size=size)
        bns = [m for m in model.model.modules() if isinstance(m, nn.BatchNorm2d)]
        assert len(bns) > 0
        for m in bns:
            assert m.eps == pytest.approx(1e-3)
            assert m.momentum == pytest.approx(0.03)
