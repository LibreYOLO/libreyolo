"""End-to-end smoke test for the LibreVLM / LFM2-VL detector.

Heavy: it downloads the 450M LFM2.5-VL weights and loads the model, so it is
gated behind the vlm/network/external_data markers and skipped if transformers
is unavailable. The model is loaded once via a module-scoped fixture.
"""

import pytest

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.vlm,
    pytest.mark.network,
    pytest.mark.external_data,
]

pytest.importorskip("transformers", reason="LibreVLM requires the 'vlm' extra")


@pytest.fixture(scope="module")
def lfm2_model():
    from libreyolo import LibreVLM

    return LibreVLM("lfm2-vl-450m", device="cpu")


def test_lfm2_predict_returns_results(lfm2_model):
    from libreyolo import Results, SAMPLE_IMAGE

    result = lfm2_model.predict(SAMPLE_IMAGE)

    assert isinstance(result, Results)
    # Same Results contract as any YOLO model: xyxy boxes with conf and cls.
    assert result.boxes.xyxy.shape[1] == 4
    assert len(result.boxes.conf) == len(result.boxes.cls) == len(result.boxes.xyxy)
    # Names default to COCO-80.
    assert lfm2_model.names[0] == "person"


def test_lfm2_feels_like_yolo_api(lfm2_model):
    """The user-facing surface mirrors a YOLO model."""
    assert callable(lfm2_model)
    assert hasattr(lfm2_model, "predict")
    assert hasattr(lfm2_model, "track")
    assert lfm2_model.task == "detect"
