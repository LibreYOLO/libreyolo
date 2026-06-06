"""End-to-end smoke test for the LibreVLM / LFM2-VL detector.

Skipped unless transformers is installed and the ``vlm`` marker is selected
(``pytest -m vlm``). Downloads the 450M LFM2.5-VL weights on first run.
"""

import pytest

pytestmark = pytest.mark.vlm

pytest.importorskip("transformers", reason="LibreVLM requires the 'vlm' extra")


def test_lfm2_predict_returns_results():
    from libreyolo import LibreVLM, Results, SAMPLE_IMAGE

    model = LibreVLM("lfm2-vl-450m", device="cpu")
    result = model.predict(SAMPLE_IMAGE)

    assert isinstance(result, Results)
    # Same Results contract as any YOLO model: xyxy boxes with conf and cls.
    assert result.boxes.xyxy.shape[1] == 4
    assert len(result.boxes.conf) == len(result.boxes.cls) == len(result.boxes.xyxy)
    # Names default to COCO-80.
    assert model.names[0] == "person"


def test_lfm2_feels_like_yolo_api():
    """The user-facing surface mirrors a YOLO model."""
    from libreyolo import LibreVLM

    model = LibreVLM("lfm2-vl-450m", device="cpu")
    # Callable form and predict() alias both exist and accept the same args.
    assert callable(model)
    assert hasattr(model, "predict")
    assert hasattr(model, "track")
    assert model.task == "detect"
