"""Public-checkpoint semantic prediction, validation, and export for PP-LiteSeg.

The four released checkpoints are Cityscapes-derived and NON-COMMERCIAL; these
tests only run inference against them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from libreyolo import LibreYOLO
from libreyolo.models.ppliteseg.nn import SIZE_CONFIGS

from .conftest import (
    PPLITESEG_SEMANTIC_PARAMS,
    PPLITESEG_SMOKE_PARAMS,
    cuda_cleanup,
    require_test_weights,
)

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.ppliteseg,
    pytest.mark.external_data,
    pytest.mark.network,
]

CITYSCAPES_NC = 19


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.mark.parametrize("family,size,weights", PPLITESEG_SEMANTIC_PARAMS)
def test_public_checkpoint_predicts_on_its_native_rectangle(
    family, size, weights, sample_image
):
    weights = require_test_weights(weights, expected_family=family)
    model = LibreYOLO(weights, device=_device())
    try:
        assert model.FAMILY == family
        assert model.size == size
        # The native canvas is rectangular and must survive the round trip.
        assert tuple(model.input_size) == SIZE_CONFIGS[size]["imgsz"]
        assert model.nb_classes == CITYSCAPES_NC
        assert model.weight_license == "Cityscapes dataset terms, non-commercial"

        result = model.predict(sample_image)
        with Image.open(sample_image) as image:
            expected_shape = (image.height, image.width)
        assert result.boxes is None
        assert result.masks is None
        assert result.semantic_mask is not None
        assert tuple(result.semantic_mask.data.shape) == expected_shape
        assert result.semantic_mask.data.unique().numel() >= 2
        # Ignore is a target-only concept; predictions are always real classes.
        assert int(result.semantic_mask.data.min()) >= 0
        assert int(result.semantic_mask.data.max()) < CITYSCAPES_NC
        assert result.names == model.names
        assert model.names[0] == "road"
    finally:
        del model
        cuda_cleanup()


@pytest.mark.parametrize("family,size,weights", PPLITESEG_SMOKE_PARAMS)
def test_semantic_inference_is_stable(family, size, weights, sample_image):
    weights = require_test_weights(weights, expected_family=family)
    model = LibreYOLO(weights, device=_device())
    try:
        first = model.predict(sample_image).semantic_mask.data.cpu()
        second = model.predict(sample_image).semantic_mask.data.cpu()
        assert torch.equal(first, second)
    finally:
        del model
        cuda_cleanup()


@pytest.mark.parametrize("family,size,weights", PPLITESEG_SMOKE_PARAMS)
def test_arbitrary_image_shapes_and_batches(family, size, weights, tmp_path):
    weights = require_test_weights(weights, expected_family=family)
    model = LibreYOLO(weights, device=_device())
    try:
        rng = np.random.default_rng(0)
        shapes = [(97, 53), (480, 640), (31, 29), (600, 400)]
        paths = []
        for index, (height, width) in enumerate(shapes):
            path = tmp_path / f"shape{index}.png"
            Image.fromarray(
                rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
            ).save(path)
            paths.append(str(path))
            result = model.predict(str(path))
            assert tuple(result.semantic_mask.data.shape) == (height, width)

        # A batch of differently-shaped images restores each one separately.
        results = model.predict(paths)
        assert len(results) == len(shapes)
        for result, shape in zip(results, shapes):
            assert tuple(result.semantic_mask.data.shape) == shape
    finally:
        del model
        cuda_cleanup()


def _make_semantic_dataset(root: Path) -> Path:
    """Two repository-authored image/mask pairs with an ignore band."""
    from libreyolo.models.ppliteseg.model import CITYSCAPES_NAMES

    image_dir = root / "images" / "val"
    mask_dir = root / "masks" / "val"
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    for index in range(2):
        image = np.zeros((64, 128, 3), dtype=np.uint8)
        image[:, :64] = (40 + index * 10, 40, 40)
        image[:, 64:] = (160, 150 + index * 10, 140)
        Image.fromarray(image).save(image_dir / f"sample{index}.png")
        mask = np.zeros((64, 128), dtype=np.uint8)
        mask[:, 64:] = 2  # building
        mask[:4] = 255  # ignore band
        Image.fromarray(mask, mode="L").save(mask_dir / f"sample{index}.png")
    data = {
        "path": str(root),
        "train": "images/val",
        "val": "images/val",
        "masks_dir": "masks",
        "nc": CITYSCAPES_NC,
        "names": dict(CITYSCAPES_NAMES),
    }
    yaml_path = root / "data.yaml"
    yaml_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return yaml_path


@pytest.mark.smoke
def test_public_t50_semantic_validation(tmp_path):
    weights = require_test_weights(
        "LibrePPLiteSegt50-sem.pt",
        expected_family="ppliteseg",
    )
    device = _device()
    model = LibreYOLO(weights, device=device)
    try:
        metrics = model.val(
            data=str(_make_semantic_dataset(tmp_path)),
            batch=1,
            workers=0,
            device=device,
            verbose=False,
        )
        assert 0.0 <= metrics["metrics/mIoU"] <= 1.0
        assert 0.0 <= metrics["metrics/pixel_accuracy"] <= 1.0
    finally:
        del model
        cuda_cleanup()


@pytest.mark.export_backend
@pytest.mark.onnx
@pytest.mark.parametrize("family,size,weights", PPLITESEG_SMOKE_PARAMS)
def test_onnx_export_keeps_the_rectangle_and_drops_aux_heads(
    family, size, weights, sample_image, tmp_path
):
    onnx = pytest.importorskip("onnx")
    weights = require_test_weights(weights, expected_family=family)
    model = LibreYOLO(weights, device=_device())
    try:
        eager_mask = model.predict(sample_image).semantic_mask.data.cpu().numpy()
        height, width = SIZE_CONFIGS[size]["imgsz"]
        path = model.export(format="onnx", output_path=str(tmp_path / "model.onnx"))
    finally:
        del model
        cuda_cleanup()

    graph = onnx.load(path)
    metadata = {prop.key: prop.value for prop in graph.metadata_props}
    assert metadata["model_family"] == "ppliteseg"
    assert metadata["task"] == "semantic"
    assert int(metadata["nc"]) == CITYSCAPES_NC
    assert int(metadata["imgsz_h"]) == height
    assert int(metadata["imgsz_w"]) == width
    assert int(metadata["imgsz"]) == max(height, width)
    # Auxiliary heads are training-only: exactly one graph output.
    assert len(graph.graph.output) == 1
    assert graph.graph.output[0].name == "semantic_logits"

    backend = LibreYOLO(path)
    try:
        assert tuple(backend.input_size) == (height, width)
        backend_mask = backend.predict(sample_image).semantic_mask.data.cpu().numpy()
        agreement = float((backend_mask == eager_mask).mean())
        assert agreement >= 0.95, f"ONNX mask agreement {agreement:.4f} below 0.95"
    finally:
        del backend
        cuda_cleanup()
