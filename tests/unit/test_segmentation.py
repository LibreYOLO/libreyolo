"""Unit tests for segmentation support: Masks class, Results with masks, factory detection."""

import pytest
import torch
import numpy as np
from PIL import Image

from libreyolo.utils.drawing import draw_masks
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


class TestPolygonLabelParsing:
    """Tests for polygon→bbox derivation in label parsers."""

    def test_parse_yolo_label_polygon_format(self):
        """parse_yolo_label_line derives bbox from polygon vertices."""
        from libreyolo.data.yolo_coco_api import parse_yolo_label_line

        # Triangle polygon: (0.2,0.3) (0.8,0.3) (0.5,0.9)
        line = "0 0.2 0.3 0.8 0.3 0.5 0.9"
        result = parse_yolo_label_line(line, img_w=100, img_h=100, num_classes=2)
        assert result is not None
        cls_id, x1, y1, x2, y2, area = result
        assert cls_id == 0
        # bbox of polygon: cx=0.5, cy=0.6, w=0.6, h=0.6
        # pixel: x1=20, y1=30, x2=80, y2=90
        assert abs(x1 - 20) < 1
        assert abs(y1 - 30) < 1
        assert abs(x2 - 80) < 1
        assert abs(y2 - 90) < 1

    def test_parse_yolo_label_detection_format(self):
        """parse_yolo_label_line still works for standard 5-column detection."""
        from libreyolo.data.yolo_coco_api import parse_yolo_label_line

        line = "1 0.5 0.5 0.4 0.6"
        result = parse_yolo_label_line(line, img_w=200, img_h=200, num_classes=2)
        assert result is not None
        cls_id, x1, y1, x2, y2, area = result
        assert cls_id == 1
        # cx=0.5, cy=0.5, w=0.4, h=0.6 → pixel: x1=60, y1=40, x2=140, y2=160
        assert abs(x1 - 60) < 1
        assert abs(y1 - 40) < 1
        assert abs(x2 - 140) < 1
        assert abs(y2 - 160) < 1

    def test_yolo_dataset_polygon_format(self):
        """YOLODataset._load_label derives bbox from polygon vertices."""
        import tempfile
        from pathlib import Path

        from PIL import Image

        # Create a minimal dataset with a polygon label
        with tempfile.TemporaryDirectory() as tmpdir:
            img_dir = Path(tmpdir) / "images" / "train"
            lbl_dir = Path(tmpdir) / "labels" / "train"
            img_dir.mkdir(parents=True)
            lbl_dir.mkdir(parents=True)

            # 100x100 dummy image
            Image.new("RGB", (100, 100)).save(img_dir / "test.jpg")
            # Polygon label: square from (0.2,0.2) to (0.8,0.8)
            (lbl_dir / "test.txt").write_text(
                "0 0.2 0.2 0.8 0.2 0.8 0.8 0.2 0.8\n"
            )

            from libreyolo.data.dataset import YOLODataset

            ds = YOLODataset(data_dir=tmpdir, split="train", img_size=(100, 100))
            _, target, _, _ = ds[0]
            # target shape: (N, 5) with [x1, y1, x2, y2, cls]
            assert len(target) == 1
            x1, y1, x2, y2, cls = target[0]
            assert cls == 0
            assert abs(x1 - 20) < 1
            assert abs(y1 - 20) < 1
            assert abs(x2 - 80) < 1
            assert abs(y2 - 80) < 1


class TestDrawMasks:
    """Tests for draw_masks visualization function."""

    def _make_img(self, w=200, h=100):
        return Image.new("RGB", (w, h), color=(128, 128, 128))

    def test_single_mask(self):
        img = self._make_img()
        masks = np.zeros((1, 100, 200), dtype=bool)
        masks[0, 20:80, 40:160] = True
        result = draw_masks(img, masks, classes=[0])
        assert isinstance(result, Image.Image)
        assert result.size == (200, 100)
        assert result.mode == "RGB"

    def test_multiple_masks_different_classes(self):
        img = self._make_img()
        masks = np.zeros((3, 100, 200), dtype=bool)
        masks[0, 10:30, 10:50] = True
        masks[1, 40:60, 60:120] = True
        masks[2, 70:90, 130:190] = True
        result = draw_masks(img, masks, classes=[0, 1, 2])
        assert result.size == (200, 100)
        # Masked regions should differ from uniform gray background
        arr = np.array(result)
        assert not np.all(arr == 128)

    def test_empty_masks(self):
        img = self._make_img()
        masks = np.zeros((0, 100, 200), dtype=bool)
        result = draw_masks(img, masks, classes=[])
        assert result.size == img.size
        # No masks → image should be unchanged
        assert np.array_equal(np.array(result), np.array(img))

    def test_all_false_mask(self):
        img = self._make_img()
        masks = np.zeros((1, 100, 200), dtype=bool)  # all False
        result = draw_masks(img, masks, classes=[0])
        assert result.size == img.size

    def test_alpha_zero_transparent(self):
        img = self._make_img()
        masks = np.ones((1, 100, 200), dtype=bool)
        result = draw_masks(img, masks, classes=[0], alpha=0.0)
        # Alpha 0 = fully transparent, image should be unchanged
        assert np.array_equal(np.array(result), np.array(img))

    def test_alpha_one_opaque(self):
        img = self._make_img()
        masks = np.ones((1, 100, 200), dtype=bool)
        result = draw_masks(img, masks, classes=[0], alpha=1.0)
        # Alpha 1 = fully opaque, masked pixels should NOT match original
        assert not np.array_equal(np.array(result), np.array(img))

    def test_does_not_modify_original(self):
        img = self._make_img()
        original_arr = np.array(img).copy()
        masks = np.ones((1, 100, 200), dtype=bool)
        draw_masks(img, masks, classes=[0])
        assert np.array_equal(np.array(img), original_arr)


class TestDetectNumOutputs:
    """Tests for ONNX segmentation output detection."""

    def test_detection_model_two_outputs(self):
        from libreyolo.export.onnx import _detect_num_outputs

        class DetModel(torch.nn.Module):
            def forward(self, x):
                return torch.zeros(1, 4), torch.zeros(1, 80)

        model = DetModel()
        dummy = torch.zeros(1, 3, 64, 64)
        assert _detect_num_outputs(model, dummy) == 2

    def test_segmentation_model_three_outputs(self):
        from libreyolo.export.onnx import _detect_num_outputs

        class SegModel(torch.nn.Module):
            def forward(self, x):
                return torch.zeros(1, 4), torch.zeros(1, 80), torch.zeros(1, 1, 64, 64)

        model = SegModel()
        dummy = torch.zeros(1, 3, 64, 64)
        assert _detect_num_outputs(model, dummy) == 3

    def test_single_output_model(self):
        from libreyolo.export.onnx import _detect_num_outputs

        class SingleModel(torch.nn.Module):
            def forward(self, x):
                return torch.zeros(1, 85, 100)

        model = SingleModel()
        dummy = torch.zeros(1, 3, 64, 64)
        assert _detect_num_outputs(model, dummy) == 1


class TestDetectSegmentation:
    """Tests for auto-detection of segmentation from weights."""

    def test_detect_seg_from_checkpoint(self):
        """_detect_segmentation returns True for checkpoints with seg keys."""
        import tempfile
        from pathlib import Path

        from libreyolo.models.rfdetr.model import LibreYOLORFDETR

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "seg_model.pt")
            torch.save(
                {"model": {"segmentation_head.weight": torch.zeros(1)}},
                path,
            )
            assert LibreYOLORFDETR._detect_segmentation(path) is True

    def test_detect_det_from_checkpoint(self):
        """_detect_segmentation returns False for detection-only checkpoints."""
        import tempfile
        from pathlib import Path

        from libreyolo.models.rfdetr.model import LibreYOLORFDETR

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "det_model.pt")
            torch.save(
                {"model": {"class_embed.weight": torch.zeros(1)}},
                path,
            )
            assert LibreYOLORFDETR._detect_segmentation(path) is False

    def test_detect_seg_from_filename(self):
        """Filename-based detection avoids loading weights."""
        from libreyolo.models.rfdetr.model import LibreYOLORFDETR

        assert LibreYOLORFDETR.detect_task_from_filename("LibreRFDETRn-seg.pt") == "seg"
        assert LibreYOLORFDETR.detect_task_from_filename("LibreRFDETRn.pt") is None
