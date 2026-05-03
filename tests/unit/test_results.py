"""Unit tests for Results and Boxes classes."""

import pytest
import torch
import numpy as np

from libreyolo.utils.results import Boxes, Keypoints, Masks, OBB, Probs, Results

pytestmark = pytest.mark.unit


class TestBoxes:
    """Tests for the Boxes wrapper class."""

    def test_empty_boxes(self):
        boxes = Boxes(
            torch.zeros((0, 4)),
            torch.zeros((0,)),
            torch.zeros((0,)),
        )
        assert len(boxes) == 0
        assert boxes.xyxy.shape == (0, 4)
        assert boxes.conf.shape == (0,)
        assert boxes.cls.shape == (0,)

    def test_populated_boxes(self):
        b = torch.tensor([[10.0, 20.0, 50.0, 60.0], [100.0, 200.0, 300.0, 400.0]])
        c = torch.tensor([0.9, 0.8])
        cl = torch.tensor([0.0, 5.0])
        boxes = Boxes(b, c, cl)

        assert len(boxes) == 2
        assert torch.equal(boxes.xyxy, b)
        assert torch.equal(boxes.conf, c)
        assert torch.equal(boxes.cls, cl)

    def test_xywh(self):
        b = torch.tensor([[10.0, 20.0, 50.0, 60.0]])
        boxes = Boxes(b, torch.tensor([0.9]), torch.tensor([0.0]))

        xywh = boxes.xywh
        assert xywh.shape == (1, 4)
        assert xywh[0, 0].item() == pytest.approx(30.0)  # cx = (10+50)/2
        assert xywh[0, 1].item() == pytest.approx(40.0)  # cy = (20+60)/2
        assert xywh[0, 2].item() == pytest.approx(40.0)  # w = 50-10
        assert xywh[0, 3].item() == pytest.approx(40.0)  # h = 60-20

    def test_data(self):
        b = torch.tensor([[10.0, 20.0, 50.0, 60.0]])
        c = torch.tensor([0.9])
        cl = torch.tensor([3.0])
        boxes = Boxes(b, c, cl)

        data = boxes.data
        assert data.shape == (1, 6)
        assert data[0, 4].item() == pytest.approx(0.9)
        assert data[0, 5].item() == pytest.approx(3.0)

    def test_tracking_id_data(self):
        boxes = Boxes(
            torch.tensor([[10.0, 20.0, 50.0, 60.0]]),
            torch.tensor([0.9]),
            torch.tensor([3.0]),
            id=torch.tensor([7]),
        )

        assert boxes.is_track
        assert boxes.id.tolist() == [7]
        assert boxes.data.shape == (1, 7)
        assert boxes.data[0, 4].item() == 7

    def test_normalized_boxes_require_orig_shape(self):
        boxes = Boxes(
            torch.tensor([[10.0, 20.0, 50.0, 60.0]]),
            torch.tensor([0.9]),
            torch.tensor([3.0]),
            orig_shape=(100, 200),
        )

        assert boxes.xyxyn[0, 0].item() == pytest.approx(0.05)
        assert boxes.xywhn[0, 2].item() == pytest.approx(0.2)

    def test_cpu(self):
        b = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        boxes = Boxes(b, torch.tensor([0.5]), torch.tensor([1.0]))
        cpu_boxes = boxes.cpu()
        assert cpu_boxes.xyxy.device.type == "cpu"
        assert len(cpu_boxes) == 1

    def test_numpy(self):
        b = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        boxes = Boxes(b, torch.tensor([0.5]), torch.tensor([1.0]))
        np_boxes = boxes.numpy()
        assert isinstance(np_boxes.xyxy, np.ndarray)
        assert isinstance(np_boxes.conf, np.ndarray)
        assert isinstance(np_boxes.cls, np.ndarray)

    def test_repr(self):
        b = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        boxes = Boxes(b, torch.tensor([0.5]), torch.tensor([1.0]))
        r = repr(boxes)
        assert "Boxes" in r
        assert "n=1" in r


class TestResults:
    """Tests for the Results class."""

    def _make_results(self, n=3):
        b = torch.rand(n, 4) * 100
        c = torch.rand(n)
        cl = torch.randint(0, 5, (n,)).float()
        boxes = Boxes(b, c, cl)
        return Results(
            boxes=boxes,
            orig_shape=(480, 640),
            path="/tmp/test.jpg",
            names={0: "cat", 1: "dog", 2: "bird", 3: "fish", 4: "horse"},
        )

    def test_empty_results(self):
        boxes = Boxes(
            torch.zeros((0, 4)),
            torch.zeros((0,)),
            torch.zeros((0,)),
        )
        result = Results(boxes=boxes, orig_shape=(480, 640))
        assert len(result) == 0
        assert result.path is None
        assert result.names == {}
        assert result.masks is None
        assert result.keypoints is None
        assert result.probs is None
        assert result.obb is None
        assert result.speed == {}

    def test_populated_results(self):
        result = self._make_results(5)
        assert len(result) == 5
        assert result.path == "/tmp/test.jpg"
        assert result.orig_shape == (480, 640)
        assert result.names[0] == "cat"

    def test_cpu(self):
        result = self._make_results(2)
        cpu_result = result.cpu()
        assert cpu_result.boxes.xyxy.device.type == "cpu"
        assert cpu_result.path == result.path
        assert cpu_result.orig_shape == result.orig_shape

    def test_getitem_preserves_flat_payloads(self):
        result = self._make_results(3)
        result.track_id = torch.tensor([10, 11, 12])
        result.boxes._id = result.track_id

        sliced = result[[0, 2]]

        assert len(sliced) == 2
        assert sliced.boxes.is_track
        assert sliced.boxes.id.tolist() == [10, 12]

    def test_select_preserves_all_instance_payloads(self):
        result = self._make_results(3)
        result.masks = Masks(torch.ones((3, 480, 640), dtype=torch.uint8), result.orig_shape)
        result.keypoints = Keypoints(torch.ones((3, 2, 3)), result.orig_shape)
        result.obb = OBB(torch.ones((3, 5)), result.orig_shape)

        sliced = result._select([0, 2])

        assert len(sliced.boxes) == 2
        assert len(sliced.masks) == 2
        assert len(sliced.keypoints) == 2
        assert len(sliced.obb) == 2

    def test_update_mutates_flat_slots(self):
        result = self._make_results(1)
        new_boxes = Boxes(
            torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
            torch.tensor([0.5]),
            torch.tensor([1.0]),
        )

        returned = result.update(boxes=new_boxes, track_id=torch.tensor([42]))

        assert returned is result
        assert result.boxes.id.tolist() == [42]
        assert result.track_id.tolist() == [42]

    def test_summary_and_json(self):
        result = self._make_results(1)

        rows = result.summary()
        payload = result.to_json()

        assert rows[0]["class"] == int(result.boxes.cls[0])
        assert "confidence" in rows[0]
        assert isinstance(payload, str)

    def test_classify_probs_result(self):
        probs = Probs(torch.tensor([0.1, 0.7, 0.2]))
        result = Results(
            boxes=None,
            orig_shape=(1, 1),
            probs=probs,
            names={0: "a", 1: "b", 2: "c"},
        )

        assert len(result) == 1
        assert result.probs.top1 == 1
        assert result.probs.top5 == [1, 2, 0]
        assert result.summary()[0]["name"] == "b"

    def test_keypoints_and_obb_accessors(self):
        keypoints = Keypoints(torch.tensor([[[10.0, 20.0, 0.9]]]), (100, 200))
        obb = OBB(torch.tensor([[10.0, 20.0, 30.0, 40.0, 0.0, 0.8, 2.0]]), (100, 200))

        assert keypoints.xy.shape == (1, 1, 2)
        assert keypoints.xyn[0, 0, 0].item() == pytest.approx(0.05)
        assert keypoints.conf[0, 0].item() == pytest.approx(0.9)
        assert keypoints.has_visible[0, 0].item() is True
        assert obb.xywhr.shape == (1, 5)
        assert obb.conf[0].item() == pytest.approx(0.8)
        assert obb.cls[0].item() == pytest.approx(2.0)
        assert obb.id is None
        assert obb.is_track is False
        assert obb.xyxyxyxy.shape == (1, 4, 2)
        assert obb.xyxy.shape == (1, 4)
        assert obb.xyxyxyxyn[0, 0, 0].item() == pytest.approx(-0.025)

    def test_repr(self):
        result = self._make_results(2)
        r = repr(result)
        assert "Results" in r
        assert "test.jpg" in r


class TestClassesFilter:
    """Tests for the classes filter in Results wrapping."""

    def test_filter_reduces_detections(self):
        b = torch.tensor(
            [[0, 0, 10, 10], [20, 20, 30, 30], [40, 40, 50, 50]], dtype=torch.float32
        )
        c = torch.tensor([0.9, 0.8, 0.7])
        cl = torch.tensor([0.0, 1.0, 0.0])
        boxes = Boxes(b, c, cl)
        Results(boxes=boxes, orig_shape=(100, 100))

        # Manually apply filter (same logic as base_model._apply_classes_filter)
        mask = cl == 0.0
        filtered = Boxes(b[mask], c[mask], cl[mask])
        assert len(filtered) == 2

    def test_filter_empty(self):
        b = torch.tensor([[0, 0, 10, 10]], dtype=torch.float32)
        c = torch.tensor([0.9])
        cl = torch.tensor([5.0])

        mask = cl == 0.0
        filtered = Boxes(b[mask], c[mask], cl[mask])
        assert len(filtered) == 0
