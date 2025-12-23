import types
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from libreyolo import LIBREYOLO8, LIBREYOLO11

pytestmark = pytest.mark.unit


def _mock_postprocess(*args, **kwargs):
    """Mock postprocess that returns the expected schema."""
    return {
        "boxes": [[10, 20, 30, 40], [50, 60, 70, 80]],
        "scores": [0.9, 0.8],
        "classes": [0, 1],
        "num_detections": 2,
    }


def build_stub(model_cls, save_feature_maps=False, feature_dir=None):
    """Create a lightweight model instance that skips weight loading."""
    inst = model_cls.__new__(model_cls)
    inst.size = "n"
    inst.reg_max = 16
    inst.nb_classes = 80
    inst.save_feature_maps = save_feature_maps
    inst.save_eigen_cam = False
    inst.feature_maps = {}
    inst.hooks = []

    class DummyModel:
        def __call__(self, x):
            return {"x8": {}, "x16": {}, "x32": {}}

    inst.model = DummyModel()

    if save_feature_maps:
        def _save(self, image_path):
            path = Path(feature_dir) if feature_dir else Path("feature_maps_stub")
            path.mkdir(parents=True, exist_ok=True)
            (path / "metadata.json").write_text("{}")
            return str(path)

        inst._save_feature_maps = types.MethodType(_save, inst)

    return inst


@pytest.mark.parametrize("model_cls", [LIBREYOLO8, LIBREYOLO11])
def test_predict_schema(model_cls):
    model = build_stub(model_cls)
    image = np.zeros((16, 16, 3), dtype=np.uint8)

    # Mock postprocess to avoid implementation details
    if model_cls == LIBREYOLO8:
        with patch("libreyolo.v8.model.postprocess", side_effect=_mock_postprocess):
            detections = model.predict(image, save=False)
    else:
        with patch("libreyolo.v11.model.postprocess", side_effect=_mock_postprocess):
            detections = model.predict(image, save=False)

    for key in ["boxes", "scores", "classes", "num_detections"]:
        assert key in detections

    assert detections["num_detections"] == len(detections["boxes"]) == len(detections["scores"]) == len(
        detections["classes"]
    )


@pytest.mark.parametrize("model_cls", [LIBREYOLO8, LIBREYOLO11])
def test_predict_includes_feature_map_path_when_enabled(model_cls, tmp_path):
    fm_dir = tmp_path / "feature_maps"
    model = build_stub(model_cls, save_feature_maps=True, feature_dir=fm_dir)

    # Mock postprocess to avoid implementation details
    if model_cls == LIBREYOLO8:
        with patch("libreyolo.v8.model.postprocess", side_effect=_mock_postprocess):
            detections = model.predict(np.zeros((8, 8, 3), dtype=np.uint8), save=False)
    else:
        with patch("libreyolo.v11.model.postprocess", side_effect=_mock_postprocess):
            detections = model.predict(np.zeros((8, 8, 3), dtype=np.uint8), save=False)

    assert detections.get("feature_maps_path") == str(fm_dir)
    assert fm_dir.exists()

