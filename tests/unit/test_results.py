"""Unit tests for Results and Boxes classes."""

import pytest
import torch
import numpy as np

from libreyolo.common.results import Boxes, Results

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

    def test_repr(self):
        result = self._make_results(2)
        r = repr(result)
        assert "Results" in r
        assert "test.jpg" in r


class TestClassesFilter:
    """Tests for the classes filter in Results wrapping."""

    def test_filter_reduces_detections(self):
        b = torch.tensor([[0, 0, 10, 10], [20, 20, 30, 30], [40, 40, 50, 50]], dtype=torch.float32)
        c = torch.tensor([0.9, 0.8, 0.7])
        cl = torch.tensor([0.0, 1.0, 0.0])
        boxes = Boxes(b, c, cl)
        result = Results(boxes=boxes, orig_shape=(100, 100))

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
