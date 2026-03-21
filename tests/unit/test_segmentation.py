"""Unit tests for segmentation support: Masks class, Results with masks, factory detection."""

import pytest
import torch
import numpy as np

from libreyolo.utils.results import Boxes, Masks, Results

pytestmark = pytest.mark.unit


class TestMasks:
    """Tests for the Masks wrapper class."""

    def test_empty_masks(self):
        masks = Masks(torch.zeros((0, 100, 100), dtype=torch.bool), (100, 100))
        assert len(masks) == 0
        assert masks.data.shape == (0, 100, 100)
        assert masks.orig_shape == (100, 100)

    def test_populated_masks(self):
        m = torch.randint(0, 2, (3, 64, 64), dtype=torch.bool)
        masks = Masks(m, (64, 64))
        assert len(masks) == 3
        assert masks.data.shape == (3, 64, 64)
        assert torch.equal(masks.data, m)

    def test_cpu(self):
        m = torch.ones((2, 32, 32), dtype=torch.bool)
        masks = Masks(m, (32, 32))
        cpu_masks = masks.cpu()
        assert cpu_masks.data.device.type == "cpu"
        assert len(cpu_masks) == 2

    def test_numpy(self):
        m = torch.ones((2, 32, 32), dtype=torch.bool)
        masks = Masks(m, (32, 32))
        np_masks = masks.numpy()
        assert isinstance(np_masks.data, np.ndarray)
        assert np_masks.data.shape == (2, 32, 32)

    def test_numpy_already_numpy(self):
        m = np.ones((2, 32, 32), dtype=np.uint8)
        masks = Masks(m, (32, 32))
        np_masks = masks.numpy()
        assert np_masks is masks  # should return self

    def test_cpu_already_numpy(self):
        m = np.ones((2, 32, 32), dtype=np.uint8)
        masks = Masks(m, (32, 32))
        cpu_masks = masks.cpu()
        assert cpu_masks is masks  # should return self

    def test_repr(self):
        m = torch.ones((3, 100, 200), dtype=torch.bool)
        masks = Masks(m, (100, 200))
        r = repr(masks)
        assert "Masks" in r
        assert "n=3" in r
        assert "(3, 100, 200)" in r

    def test_xy_contours(self):
        # Create a simple square mask
        m = torch.zeros((1, 100, 100), dtype=torch.bool)
        m[0, 20:80, 20:80] = True
        masks = Masks(m, (100, 100))
        contours = masks.xy
        assert len(contours) == 1
        assert contours[0].shape[1] == 2  # (M, 2) points
        assert len(contours[0]) > 0  # has points

    def test_xyn_normalized(self):
        m = torch.zeros((1, 100, 200), dtype=torch.bool)
        m[0, 20:80, 40:160] = True
        masks = Masks(m, (100, 200))
        contours = masks.xyn
        assert len(contours) == 1
        # All coordinates should be in [0, 1]
        assert contours[0][:, 0].max() <= 1.0
        assert contours[0][:, 1].max() <= 1.0
        assert contours[0][:, 0].min() >= 0.0
        assert contours[0][:, 1].min() >= 0.0

    def test_xy_empty_mask(self):
        m = torch.zeros((1, 50, 50), dtype=torch.bool)  # all zeros
        masks = Masks(m, (50, 50))
        contours = masks.xy
        assert len(contours) == 1
        assert contours[0].shape == (0, 2)  # empty contour


class TestResultsWithMasks:
    """Tests for Results class with segmentation masks."""

    def _make_results_with_masks(self, n=3, h=100, w=200):
        boxes = Boxes(
            torch.rand(n, 4) * 100,
            torch.rand(n),
            torch.randint(0, 2, (n,)).float(),
        )
        masks = Masks(
            torch.randint(0, 2, (n, h, w), dtype=torch.bool),
            (h, w),
        )
        return Results(
            boxes=boxes,
            orig_shape=(h, w),
            path="/tmp/test.jpg",
            names={0: "fire", 1: "smoke"},
            masks=masks,
        )

    def test_results_with_masks(self):
        result = self._make_results_with_masks(5)
        assert len(result) == 5
        assert result.masks is not None
        assert len(result.masks) == 5

    def test_results_without_masks(self):
        boxes = Boxes(torch.rand(3, 4), torch.rand(3), torch.zeros(3))
        result = Results(boxes=boxes, orig_shape=(100, 100))
        assert result.masks is None

    def test_cpu_with_masks(self):
        result = self._make_results_with_masks(2)
        cpu_result = result.cpu()
        assert cpu_result.masks is not None
        assert cpu_result.masks.data.device.type == "cpu"
        assert cpu_result.boxes.xyxy.device.type == "cpu"

    def test_repr_with_masks(self):
        result = self._make_results_with_masks(2)
        r = repr(result)
        assert "Results" in r
        assert "masks=" in r
        assert "Masks" in r

    def test_repr_without_masks(self):
        boxes = Boxes(torch.rand(2, 4), torch.rand(2), torch.zeros(2))
        result = Results(boxes=boxes, orig_shape=(100, 100))
        r = repr(result)
        assert "masks=" not in r


class TestClassesFilterWithMasks:
    """Tests for class filtering when masks are present."""

    def test_filter_with_masks(self):
        from libreyolo.models.base.inference import InferenceRunner

        boxes = torch.tensor([[0, 0, 10, 10], [20, 20, 30, 30], [40, 40, 50, 50]],
                             dtype=torch.float32)
        conf = torch.tensor([0.9, 0.8, 0.7])
        cls = torch.tensor([0.0, 1.0, 0.0])
        masks = torch.randint(0, 2, (3, 64, 64), dtype=torch.bool)

        filtered_boxes, filtered_conf, filtered_cls, filtered_masks = (
            InferenceRunner._apply_classes_filter(boxes, conf, cls, [0], masks)
        )

        assert len(filtered_boxes) == 2
        assert len(filtered_masks) == 2

    def test_filter_without_masks(self):
        from libreyolo.models.base.inference import InferenceRunner

        boxes = torch.tensor([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=torch.float32)
        conf = torch.tensor([0.9, 0.8])
        cls = torch.tensor([0.0, 1.0])

        filtered_boxes, filtered_conf, filtered_cls, filtered_masks = (
            InferenceRunner._apply_classes_filter(boxes, conf, cls, [0])
        )

        assert len(filtered_boxes) == 1
        assert filtered_masks is None


class TestFactorySegDetection:
    """Tests for -seg suffix detection in filenames."""

    def test_detect_size_from_seg_filename(self):
        from libreyolo.models.rfdetr.model import LibreYOLORFDETR

        assert LibreYOLORFDETR.detect_size_from_filename("LibreRFDETRs-seg.pt") == "s"
        assert LibreYOLORFDETR.detect_size_from_filename("LibreRFDETRn-seg.pt") == "n"
        assert LibreYOLORFDETR.detect_size_from_filename("LibreRFDETRm-seg.pt") == "m"
        assert LibreYOLORFDETR.detect_size_from_filename("LibreRFDETRl-seg.pt") == "l"

    def test_detect_task_from_seg_filename(self):
        from libreyolo.models.rfdetr.model import LibreYOLORFDETR

        assert LibreYOLORFDETR.detect_task_from_filename("LibreRFDETRs-seg.pt") == "seg"
        assert LibreYOLORFDETR.detect_task_from_filename("LibreRFDETRs.pt") is None

    def test_det_filename_still_works(self):
        from libreyolo.models.rfdetr.model import LibreYOLORFDETR

        assert LibreYOLORFDETR.detect_size_from_filename("LibreRFDETRs.pt") == "s"
        assert LibreYOLORFDETR.detect_task_from_filename("LibreRFDETRs.pt") is None

    def test_download_url_seg(self):
        from libreyolo.models.rfdetr.model import LibreYOLORFDETR

        url = LibreYOLORFDETR.get_download_url("LibreRFDETRs-seg.pt")
        assert url == "https://huggingface.co/LibreYOLO/LibreRFDETRs-seg/resolve/main/LibreRFDETRs-seg.pt"

    def test_download_url_det(self):
        from libreyolo.models.rfdetr.model import LibreYOLORFDETR

        url = LibreYOLORFDETR.get_download_url("LibreRFDETRs.pt")
        assert url == "https://huggingface.co/LibreYOLO/LibreRFDETRs/resolve/main/LibreRFDETRs.pt"

    def test_other_families_unaffected(self):
        from libreyolo.models.yolox.model import LibreYOLOX
        from libreyolo.models.yolo9.model import LibreYOLO9

        assert LibreYOLOX.detect_size_from_filename("LibreYOLOXs.pt") == "s"
        assert LibreYOLOX.detect_task_from_filename("LibreYOLOXs.pt") is None
        assert LibreYOLO9.detect_size_from_filename("LibreYOLO9s.pt") == "s"
        assert LibreYOLO9.detect_task_from_filename("LibreYOLO9s.pt") is None
