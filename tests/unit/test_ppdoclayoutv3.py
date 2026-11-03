"""Unit tests for the PP-DocLayoutV3 family (no weights, no network)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.unit


def _native_state() -> dict[str, torch.Tensor]:
    from libreyolo.models.ppdoclayoutv3.nn import PPDocLayoutV3Model

    with torch.device("meta"):
        return PPDocLayoutV3Model(nc=25).state_dict()


def _upstream_state() -> dict[str, torch.Tensor]:
    return {f"model.{k}": v for k, v in _native_state().items()}


def _rtdetr_state() -> dict[str, torch.Tensor]:
    return {
        "backbone.stages.0.blocks.0.layers.0.conv.weight": torch.zeros(1),
        "encoder.input_proj.0.0.weight": torch.zeros(256, 512, 1, 1),
        "decoder.input_proj.0.conv.weight": torch.zeros(1),
        "decoder.dec_score_head.0.weight": torch.zeros(80, 256),
        "decoder.dec_score_head.0.bias": torch.zeros(80),
    }


def _picodet_state() -> dict[str, torch.Tensor]:
    from libreyolo.models.picodet.nn import LibrePICODETModel

    with torch.device("meta"):
        return LibrePICODETModel(size="s", nb_classes=80).state_dict()


def _ppocr_state() -> dict[str, torch.Tensor]:
    return {
        "det.backbone.stem.stem1.conv.weight": torch.zeros(1),
        "det.head.binarize.conv1.weight": torch.zeros(1),
        "rec.head.ctc_head.fc.weight": torch.zeros(1),
    }


def test_can_load_native_and_metadata():
    from libreyolo import LibrePPDocLayoutV3

    state = _native_state()
    assert LibrePPDocLayoutV3.can_load(state) is True
    assert LibrePPDocLayoutV3.detect_size(state) == "l"
    assert LibrePPDocLayoutV3.detect_nb_classes(state) == 25
    assert LibrePPDocLayoutV3.can_load(_upstream_state()) is False


@pytest.mark.parametrize("factory", [_rtdetr_state, _picodet_state, _ppocr_state])
def test_rejects_sibling_paddle_and_detr_checkpoints(factory):
    from libreyolo import LibrePPDocLayoutV3

    state = factory()
    assert LibrePPDocLayoutV3.can_load(state) is False
    assert LibrePPDocLayoutV3.convert_upstream_state_dict(state) is None


def test_siblings_accept_their_own_layout_only():
    from libreyolo import LibrePICODET, LibrePPOCR, LibreRTDETR

    ours = _native_state()
    assert LibreRTDETR.can_load(_rtdetr_state()) is True
    assert LibrePICODET.can_load(_picodet_state()) is True
    assert LibrePPOCR.can_load(_ppocr_state()) is True
    for sibling in (LibreRTDETR, LibrePICODET, LibrePPOCR):
        assert sibling.can_load(ours) is False


def test_no_other_registered_family_claims_our_checkpoints():
    from libreyolo.models import try_ensure_rfdetr
    from libreyolo.models.base import BaseModel

    try_ensure_rfdetr()  # registers RF-DETR (broad can_load) when available
    from libreyolo.models.ppdoclayoutv3.model import LibrePPDocLayoutV3

    native, upstream = _native_state(), _upstream_state()
    for cls in BaseModel._registry:
        if cls is LibrePPDocLayoutV3:
            continue
        assert not cls.can_load(native), cls.FAMILY
        try:
            claim = cls.convert_upstream_state_dict(upstream)
        except Exception:  # noqa: BLE001  (a raising recognizer claims nothing)
            claim = None
        assert not claim, cls.FAMILY


def test_upstream_recognizer_strips_prefix():
    from libreyolo import LibrePPDocLayoutV3

    converted = LibrePPDocLayoutV3.convert_upstream_state_dict(_upstream_state())
    assert converted is not None
    assert set(converted) == set(_native_state())


def test_filename_and_download_url():
    from libreyolo import LibrePPDocLayoutV3

    assert LibrePPDocLayoutV3.detect_size_from_filename("LibrePPDocLayoutV3l.pt") == "l"
    assert LibrePPDocLayoutV3.detect_size_from_filename("LibreRTDETRl.pt") is None
    assert LibrePPDocLayoutV3.get_download_url("LibrePPDocLayoutV3l.pt") == (
        "https://huggingface.co/LibreYOLO/LibrePPDocLayoutV3l/resolve/main/"
        "LibrePPDocLayoutV3l.pt"
    )
    assert LibrePPDocLayoutV3.default_checkpoint_names(25)[22] == "text"


def _synthetic_outputs():
    # Four queries, two classes. Reading order: query 2, 0, 1, 3.
    logits = torch.tensor([[[4.0, -9.0], [3.0, 2.5], [-9.0, 5.0], [-9.0, -9.0]]])
    boxes = torch.tensor(
        [
            [
                [0.5, 0.5, 0.2, 0.2],
                [0.5, 0.8, 0.2, 0.2],
                [0.5, 0.2, 0.2, 0.2],
                [0.1, 0.1, 0.1, 0.1],
            ]
        ]
    )
    votes = torch.tensor([[1.0, 2.0, 0.0, 3.0]])
    return {"pred_logits": logits, "pred_boxes": boxes, "order_votes": votes}


def test_postprocess_sorts_by_reading_order_and_keeps_multilabel():
    from libreyolo.postprocess.ppdoclayoutv3 import postprocess

    det = postprocess(_synthetic_outputs(), conf_thres=0.5, original_size=(100, 200))
    # Query 1 passes for both classes (upstream top-k over query x class).
    assert det["num_detections"] == 4
    np.testing.assert_array_equal(det["classes"], [1, 0, 0, 1])
    np.testing.assert_array_equal(det["reading_order"], [1, 2, 3, 4])
    np.testing.assert_allclose(det["boxes"][0], [40.0, 20.0, 60.0, 60.0], atol=1e-5)


def test_postprocess_max_det_keeps_reading_order_ranks():
    from libreyolo.postprocess.ppdoclayoutv3 import postprocess

    det = postprocess(_synthetic_outputs(), conf_thres=0.5, max_det=2)
    np.testing.assert_array_equal(det["reading_order"], [1, 2])
    empty = postprocess(_synthetic_outputs(), conf_thres=0.9999)
    assert empty["num_detections"] == 0 and empty["reading_order"].shape == (0,)


def test_reading_order_votes_rank_pairwise_logits():
    from libreyolo.postprocess.ppdoclayoutv3 import order_ranks, reading_order_votes

    # Upper triangle: logits[i, j] > 0 means i precedes j. Order: 2, 0, 1.
    logits = torch.full((3, 3), -20.0)
    logits[0, 1] = 20.0  # 0 before 1; 2 before 0 and 2 before 1 stay negative
    ranks = order_ranks(reading_order_votes(logits[None])[0])
    assert ranks.tolist() == [1, 2, 0]


def test_backend_decode_matches_torch_postprocess():
    from libreyolo.backends.base import BaseBackend
    from libreyolo.postprocess.ppdoclayoutv3 import postprocess

    torch.manual_seed(0)
    outputs = {
        "pred_logits": torch.randn(1, 300, 25) * 3,
        "pred_boxes": torch.rand(1, 300, 4),
        "order_votes": torch.rand(1, 300) * 300,
    }
    want = postprocess(outputs, conf_thres=0.3, original_size=(640, 480))
    got = BaseBackend._parse_ppdoclayoutv3(
        [v.numpy() for v in outputs.values()], 640, 480, 0.3
    )
    np.testing.assert_allclose(got[0], want["boxes"], atol=1e-4)
    np.testing.assert_allclose(got[1], want["scores"], atol=1e-6)
    np.testing.assert_array_equal(got[2], want["classes"])
    np.testing.assert_array_equal(got[6], want["reading_order"])


def test_results_reading_order_slices_with_boxes():
    from libreyolo.utils.results import Boxes, Results

    boxes = Boxes(
        torch.rand(3, 4), torch.tensor([0.9, 0.8, 0.7]), torch.tensor([0.0, 1.0, 2.0])
    )
    result = Results(
        boxes=boxes, orig_shape=(10, 10), reading_order=torch.tensor([1, 2, 3])
    )
    sub = result[[2, 0]]
    assert sub.reading_order.tolist() == [3, 1]
    assert [row["reading_order"] for row in result.summary()] == [1, 2, 3]
    assert isinstance(result.numpy().reading_order, np.ndarray)
    assert "reading_order=3" in repr(result)
    assert Results(boxes=boxes, orig_shape=(10, 10)).reading_order is None


def test_tiny_forward_and_predict_threads_reading_order():
    from libreyolo import LibrePPDocLayoutV3

    torch.manual_seed(0)
    model = LibrePPDocLayoutV3(size="l", device="cpu")
    model.model.eval()
    with torch.no_grad():
        out = model.model(torch.rand(1, 3, 256, 256), return_masks=True)
    assert out["pred_logits"].shape == (1, 300, 25)
    assert out["pred_boxes"].shape == (1, 300, 4)
    assert out["order_logits"].shape == (1, 300, 300)
    assert out["pred_masks"].shape == (1, 300, 64, 64)

    image = (np.random.default_rng(0).random((120, 90, 3)) * 255).astype(np.uint8)
    result = model(image, conf=0.0, imgsz=256, classes=[3, 7])
    assert set(result.boxes.cls.tolist()) <= {3.0, 7.0}
    order = result.reading_order.tolist()
    assert len(order) == len(result) and order == sorted(order)


def test_train_is_rejected():
    from libreyolo import LibrePPDocLayoutV3

    with pytest.raises(NotImplementedError):
        LibrePPDocLayoutV3.train(object())
