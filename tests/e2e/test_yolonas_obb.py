"""Task-appropriate trained-checkpoint smoke coverage for YOLO-NAS-R (OBB)."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from libreyolo import LibreYOLO

from .conftest import YOLONAS_OBB_PARAMS


pytestmark = [pytest.mark.e2e, pytest.mark.network, pytest.mark.yolonas]


@pytest.mark.parametrize("family,size,weights", YOLONAS_OBB_PARAMS)
def test_yolonas_obb_trained_checkpoint_smoke(family, size, weights):
    model = LibreYOLO(weights)
    assert (model.FAMILY, model.size, model.task) == (family, size, "obb")
    assert model.input_size == 1024
    assert model.names[0] == "plane"

    # Non-square on purpose: the OBB preprocessor pads bottom-right, and the
    # inverse transform has to land results back on the original canvas.
    image = Image.fromarray(np.full((480, 800, 3), 127, dtype=np.uint8))
    result = model.predict(image, conf=0.0, max_det=5)[0]

    assert result.obb is not None
    assert result.obb.data.shape[1] == 7
    assert result.boxes.data.shape[1] == 6
    assert len(result.obb) == len(result.boxes) <= 5
    assert result.orig_shape == (480, 800)

    xywhr = np.asarray(result.obb.xywhr)
    if len(xywhr):
        assert (xywhr[:, 0] >= 0).all() and (xywhr[:, 0] <= 800).all()
        assert (xywhr[:, 1] >= 0).all() and (xywhr[:, 1] <= 480).all()
        # Public contract: long side first, angle in [-pi/2, pi/2).
        assert (xywhr[:, 2] >= xywhr[:, 3]).all()
        assert (xywhr[:, 4] >= -np.pi / 2).all() and (xywhr[:, 4] < np.pi / 2).all()


@pytest.mark.parametrize("family,size,weights", YOLONAS_OBB_PARAMS)
def test_yolonas_obb_trained_checkpoint_takes_a_gradient_step(family, size, weights):
    """The public checkpoint trains: one real forward/backward on its own head."""
    import torch

    model = LibreYOLO(weights)
    from libreyolo.models.yolonas.obb_loss import YOLONASOBBLoss

    loss_fn = YOLONASOBBLoss(num_classes=model.nb_classes).to(model.device)
    net = model.model.train()
    images = torch.rand(1, 3, 320, 320, device=model.device)
    targets = torch.zeros(1, 4, 6, device=model.device)
    targets[0, 0] = torch.tensor(
        [0.0, 160.0, 160.0, 80.0, 40.0, 0.3], device=model.device
    )

    loss, components = loss_fn(net(images), targets)
    loss.backward()

    assert torch.isfinite(loss) and loss.item() > 0
    assert components.shape == (4,)
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
