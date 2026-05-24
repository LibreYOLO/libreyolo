"""Unit tests for segmentation support: Masks class, Results with masks, factory detection."""

import pytest
import torch
import numpy as np
from PIL import Image

from libreyolo.data.utils import polygon_to_cxcywh
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

        boxes = torch.tensor(
            [[0, 0, 10, 10], [20, 20, 30, 30], [40, 40, 50, 50]], dtype=torch.float32
        )
        conf = torch.tensor([0.9, 0.8, 0.7])
        cls = torch.tensor([0.0, 1.0, 0.0])
        masks = torch.randint(0, 2, (3, 64, 64), dtype=torch.bool)

        filtered_boxes, filtered_conf, filtered_cls, filtered_masks, filtered_kpts = (
            InferenceRunner._apply_classes_filter(boxes, conf, cls, [0], masks)
        )

        assert len(filtered_boxes) == 2
        assert len(filtered_masks) == 2
        assert filtered_kpts is None

    def test_filter_without_masks(self):
        from libreyolo.models.base.inference import InferenceRunner

        boxes = torch.tensor([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=torch.float32)
        conf = torch.tensor([0.9, 0.8])
        cls = torch.tensor([0.0, 1.0])

        filtered_boxes, filtered_conf, filtered_cls, filtered_masks, filtered_kpts = (
            InferenceRunner._apply_classes_filter(boxes, conf, cls, [0])
        )

        assert len(filtered_boxes) == 1
        assert filtered_masks is None
        assert filtered_kpts is None


class TestFactorySegDetection:
    """Tests for -seg suffix detection in filenames."""

    def test_detect_size_from_seg_filename(self):
        from libreyolo.models.rfdetr.model import LibreRFDETR

        assert LibreRFDETR.detect_size_from_filename("LibreRFDETRs-seg.pt") == "s"
        assert LibreRFDETR.detect_size_from_filename("LibreRFDETRn-seg.pt") == "n"
        assert LibreRFDETR.detect_size_from_filename("LibreRFDETRm-seg.pt") == "m"
        assert LibreRFDETR.detect_size_from_filename("LibreRFDETRl-seg.pt") == "l"
        assert LibreRFDETR.detect_size_from_filename("LibreRFDETRx-seg.pt") == "x"
        assert LibreRFDETR.detect_size_from_filename("LibreRFDETRxx-seg.pt") == "xx"

    def test_detect_task_from_seg_filename(self):
        from libreyolo.models.rfdetr.model import LibreRFDETR

        assert LibreRFDETR.detect_task_from_filename("LibreRFDETRs-seg.pt") == "segment"
        assert LibreRFDETR.detect_task_from_filename("LibreRFDETRs.pt") is None

    def test_det_filename_still_works(self):
        from libreyolo.models.rfdetr.model import LibreRFDETR

        assert LibreRFDETR.detect_size_from_filename("LibreRFDETRs.pt") == "s"
        assert LibreRFDETR.detect_task_from_filename("LibreRFDETRs.pt") is None

    def test_download_url_seg(self):
        from libreyolo.models.rfdetr.model import LibreRFDETR

        url = LibreRFDETR.get_download_url("LibreRFDETRs-seg.pt")
        assert (
            url
            == "https://huggingface.co/LibreYOLO/LibreRFDETRs-seg/resolve/main/LibreRFDETRs-seg.pt"
        )

    def test_download_url_upstream_default_weights(self):
        from libreyolo.models.rfdetr.model import LibreRFDETR

        assert (
            LibreRFDETR.get_download_url("rf-detr-seg-nano.pt")
            == "https://storage.googleapis.com/rfdetr/rf-detr-seg-n-ft.pth"
        )
        assert (
            LibreRFDETR.get_download_url("rf-detr-seg-xlarge.pt")
            == "https://storage.googleapis.com/rfdetr/rf-detr-seg-xl-ft.pth"
        )
        assert (
            LibreRFDETR.get_download_url("rf-detr-seg-xxlarge.pt")
            == "https://storage.googleapis.com/rfdetr/rf-detr-seg-2xl-ft.pth"
        )

    def test_download_url_det(self):
        from libreyolo.models.rfdetr.model import LibreRFDETR

        url = LibreRFDETR.get_download_url("LibreRFDETRs.pt")
        assert (
            url
            == "https://huggingface.co/LibreYOLO/LibreRFDETRs/resolve/main/LibreRFDETRs.pt"
        )

    def test_other_families_unaffected(self):
        from libreyolo.models.yolox.model import LibreYOLOX
        from libreyolo.models.yolo9.model import LibreYOLO9

        assert LibreYOLOX.detect_size_from_filename("LibreYOLOXs.pt") == "s"
        assert LibreYOLOX.detect_task_from_filename("LibreYOLOXs.pt") is None
        assert LibreYOLOX.detect_size_from_filename("LibreYOLOXs-seg.pt") is None
        assert LibreYOLOX.get_download_url("LibreYOLOXs-seg.pt") is None
        assert LibreYOLO9.detect_size_from_filename("LibreYOLO9s.pt") == "s"
        assert LibreYOLO9.detect_task_from_filename("LibreYOLO9s.pt") is None
        assert LibreYOLO9.detect_size_from_filename("LibreYOLO9s-seg.pt") == "s"
        assert LibreYOLO9.detect_task_from_filename("LibreYOLO9s-seg.pt") == "segment"


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
            (lbl_dir / "test.txt").write_text("0 0.2 0.2 0.8 0.2 0.8 0.8 0.2 0.8\n")

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

    def test_yolo_dataset_preserves_segments_when_requested(self):
        import tempfile
        from pathlib import Path

        from PIL import Image
        from libreyolo.data.dataset import YOLODataset

        with tempfile.TemporaryDirectory() as tmpdir:
            img_dir = Path(tmpdir) / "images" / "train"
            lbl_dir = Path(tmpdir) / "labels" / "train"
            img_dir.mkdir(parents=True)
            lbl_dir.mkdir(parents=True)

            Image.new("RGB", (100, 100)).save(img_dir / "test.jpg")
            (lbl_dir / "test.txt").write_text("0 0.2 0.2 0.8 0.2 0.8 0.8 0.2 0.8\n")

            default_ds = YOLODataset(data_dir=tmpdir, split="train", img_size=(100, 100))
            seg_ds = YOLODataset(
                data_dir=tmpdir,
                split="train",
                img_size=(100, 100),
                load_segments=True,
            )

            assert default_ds.segments is None
            assert seg_ds.segments is not None
            assert len(seg_ds.segments[0][0]) == 1
            assert seg_ds.segments[0][0][0].shape == (4, 2)
            assert seg_ds.segments[0][0][0][0].tolist() == [20.0, 20.0]

            item = seg_ds[0]
            assert len(item) == 5
            _, _, _, _, segments = item
            assert len(segments[0]) == 1
            assert segments[0][0].shape == (4, 2)
            assert segments[0][0][2].tolist() == [80.0, 80.0]
            # Polygon-sourced rings should NOT carry a dataset-resident dense
            # mask: that would OOM on full COCO (issue #270). Crop fidelity
            # is preserved by materializing the mask lazily inside the
            # RF-DETR transform when the crop branch fires.
            assert getattr(segments[0][0], "dense_mask", None) is None

    def test_yolo_dataset_bbox_rows_become_rectangle_segments_when_requested(self):
        import tempfile
        from pathlib import Path

        from libreyolo.data.dataset import YOLODataset

        with tempfile.TemporaryDirectory() as tmpdir:
            img_dir = Path(tmpdir) / "images" / "train"
            lbl_dir = Path(tmpdir) / "labels" / "train"
            img_dir.mkdir(parents=True)
            lbl_dir.mkdir(parents=True)

            Image.new("RGB", (100, 80)).save(img_dir / "test.jpg")
            (lbl_dir / "test.txt").write_text("0 0.5 0.5 0.4 0.5\n")

            seg_ds = YOLODataset(
                data_dir=tmpdir,
                split="train",
                img_size=(100, 100),
                load_segments=True,
            )

            ring = seg_ds.segments[0][0][0]
            assert ring.shape == (4, 2)
            assert ring.tolist() == [
                [30.0, 20.0],
                [70.0, 20.0],
                [70.0, 60.0],
                [30.0, 60.0],
            ]
            assert getattr(ring, "dense_mask", None) is None

    def test_yolo_collate_preserves_segments_when_present(self):
        from libreyolo.data.dataset import yolox_collate_fn

        img = np.zeros((3, 32, 32), dtype=np.float32)
        target = np.zeros((2, 5), dtype=np.float32)
        segment = [[np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)]]

        batch = [
            (img, target, (32, 32), 0, segment),
            (img, target, (32, 32), 1, []),
        ]

        imgs, targets, img_infos, img_ids, segments = yolox_collate_fn(batch)

        assert imgs.shape == (2, 3, 32, 32)
        assert targets.shape == (2, 2, 5)
        assert img_infos == ((32, 32), (32, 32))
        assert img_ids == (0, 1)
        assert segments[0][0][0].tolist() == [[1.0, 2.0], [3.0, 4.0]]

    def test_yolo_collate_stacks_segmentation_masks(self):
        from libreyolo.data.dataset import yolox_collate_fn

        img = np.zeros((3, 32, 32), dtype=np.float32)
        target = np.zeros((2, 5), dtype=np.float32)
        masks = np.zeros((2, 8, 8), dtype=np.float32)
        masks[0, 2:6, 2:6] = 1

        batch = [
            (img, target, (32, 32), 0, masks),
            (img, target, (32, 32), 1, masks),
        ]

        imgs, targets, img_infos, img_ids, stacked_masks = yolox_collate_fn(batch)

        assert imgs.shape == (2, 3, 32, 32)
        assert targets.shape == (2, 2, 5)
        assert stacked_masks.shape == (2, 2, 8, 8)
        assert stacked_masks[0, 0].sum() == 16

    def test_yolo9_seg_transform_rasterizes_polygons(self):
        from libreyolo.models.yolo9.transforms import YOLO9TrainTransform

        image = np.zeros((64, 64, 3), dtype=np.uint8)
        targets = np.array([[16, 16, 48, 48, 0]], dtype=np.float32)
        segments = [
            [np.array([[16, 16], [48, 16], [48, 48], [16, 48]], dtype=np.float32)]
        ]
        transform = YOLO9TrainTransform(
            max_labels=4,
            flip_prob=0.0,
            hsv_prob=0.0,
            mask_downsample_ratio=4,
        )

        img, labels, masks = transform(image, targets, (64, 64), segments)

        assert img.shape == (3, 64, 64)
        assert labels.shape == (4, 5)
        assert masks.shape == (4, 16, 16)
        assert labels[0, 0] == 0
        assert masks[0].sum() > 0

    def test_rfdetr_seg_transform_keeps_full_resolution_masks(self):
        from libreyolo.models.rfdetr.seg_transforms import RFDETRSegTransform

        image = np.zeros((64, 64, 3), dtype=np.uint8)
        targets = np.array([[16, 16, 48, 48, 0]], dtype=np.float32)
        segments = [
            [np.array([[16, 16], [48, 16], [48, 48], [16, 48]], dtype=np.float32)]
        ]
        transform = RFDETRSegTransform(
            max_labels=4,
            flip_prob=0.0,
            imgsz=64,
            mask_downsample_ratio=4,
        )

        img, labels, masks = transform(image, targets, (64, 64), segments)

        assert img.shape == (3, 64, 64)
        assert labels.shape == (4, 5)
        assert masks.shape == (4, 64, 64)
        assert labels[0, 0] == 0
        assert masks[0].sum() > 0

    def test_rfdetr_seg_transform_square_resizes_non_square_images(self):
        from libreyolo.models.rfdetr.seg_transforms import RFDETRSegTransform

        image = np.zeros((40, 80, 3), dtype=np.uint8)
        targets = np.array([[20, 10, 60, 30, 0]], dtype=np.float32)
        segments = [
            [np.array([[20, 10], [60, 10], [60, 30], [20, 30]], dtype=np.float32)]
        ]
        transform = RFDETRSegTransform(max_labels=4, flip_prob=0.0, imgsz=80)

        _, labels, masks = transform(image, targets, (80, 80), segments)

        assert labels[0].tolist() == pytest.approx([0, 40, 40, 40, 40])
        assert masks[0, 40, 40] == 1

    def test_rfdetr_multi_scale_matches_upstream_scale_grid(self):
        from libreyolo.models.rfdetr.seg_transforms import (
            RFDETRDetTransform,
            compute_multi_scale_scales,
        )

        scales = compute_multi_scale_scales(
            384,
            expanded_scales=True,
            patch_size=16,
            num_windows=2,
        )

        assert scales == [224, 256, 288, 320, 352, 384, 416, 448, 480, 512, 544]

        transform = RFDETRDetTransform(
            max_labels=4,
            flip_prob=0.0,
            imgsz=384,
            multi_scale=True,
            expanded_scales=True,
            patch_size=16,
            num_windows=2,
        )

        image = np.zeros((100, 100, 3), dtype=np.uint8)
        targets = np.array([[25, 25, 75, 75, 0]], dtype=np.float32)
        img, labels = transform(image, targets, (384, 384))

        assert img.shape == (3, 544, 544)
        assert labels[0].tolist() == pytest.approx([0, 272, 272, 272, 272])

    def test_rfdetr_crop_resize_branch_updates_boxes_and_masks(self, monkeypatch):
        from libreyolo.models.rfdetr.seg_transforms import RFDETRSegTransform

        monkeypatch.setattr("libreyolo.models.rfdetr.seg_transforms.random.random", lambda: 0.0)
        monkeypatch.setattr("libreyolo.models.rfdetr.seg_transforms.random.choice", lambda seq: seq[0])

        randint_values = iter([20, 10, 10])
        monkeypatch.setattr(
            "libreyolo.models.rfdetr.seg_transforms.random.randint",
            lambda _a, _b: next(randint_values),
        )

        image = np.zeros((40, 40, 3), dtype=np.uint8)
        targets = np.array([[10, 10, 30, 30, 0]], dtype=np.float32)
        segments = [
            [np.array([[10, 10], [30, 10], [30, 30], [10, 30]], dtype=np.float32)]
        ]
        transform = RFDETRSegTransform(
            max_labels=4,
            flip_prob=0.0,
            imgsz=40,
            crop_resize_prob=1.0,
            crop_intermediate_sizes=(40,),
            crop_min_size=20,
            crop_max_size=20,
        )

        _, labels, masks = transform(image, targets, (40, 40), segments)

        assert labels[0].tolist() == pytest.approx([0, 20, 20, 40, 40])
        assert masks[0, 20, 20] == 1

    def test_yolo_dataset_polygon_segments_do_not_allocate_dense_masks(self, tmp_path):
        """Issue #270 regression: opening a polygon dataset must stay cheap.

        Loading many polygon-annotated images must not allocate full-resolution
        uint8 rasters per polygon; otherwise full-COCO loaders OOM-kill before
        the first batch.
        """
        from libreyolo.data.dataset import YOLODataset

        img_dir = tmp_path / "images" / "train"
        lbl_dir = tmp_path / "labels" / "train"
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)

        for i in range(8):
            Image.new("RGB", (640, 480)).save(img_dir / f"img_{i}.jpg")
            (lbl_dir / f"img_{i}.txt").write_text(
                "0 0.2 0.2 0.8 0.2 0.8 0.8 0.2 0.8\n" * 4
            )

        dataset = YOLODataset(
            data_dir=str(tmp_path),
            split="train",
            img_size=(640, 640),
            load_segments=True,
        )

        for instance_segments in dataset.segments:
            for instance in instance_segments:
                for ring in instance:
                    assert getattr(ring, "dense_mask", None) is None

    def test_rfdetr_crop_uses_dense_polygon_mask(self, monkeypatch):
        import cv2

        from libreyolo.data.dataset import DenseMaskRing
        from libreyolo.models.rfdetr.seg_transforms import RFDETRSegTransform

        monkeypatch.setattr("libreyolo.models.rfdetr.seg_transforms.random.random", lambda: 0.0)
        monkeypatch.setattr("libreyolo.models.rfdetr.seg_transforms.random.choice", lambda seq: seq[0])

        randint_values = iter([20, 10, 10])
        monkeypatch.setattr(
            "libreyolo.models.rfdetr.seg_transforms.random.randint",
            lambda _a, _b: next(randint_values),
        )

        image = np.zeros((40, 40, 3), dtype=np.uint8)
        targets = np.array([[0, 0, 40, 40, 0]], dtype=np.float32)
        ring = np.array([[0, 0], [40, 20], [0, 40]], dtype=np.float32)
        dense = np.zeros((40, 40), dtype=np.uint8)
        cv2.fillPoly(dense, [ring.astype(np.int32)], color=1)
        segments = [[DenseMaskRing(ring, dense)]]
        transform = RFDETRSegTransform(
            max_labels=4,
            flip_prob=0.0,
            imgsz=40,
            crop_resize_prob=1.0,
            crop_intermediate_sizes=(40,),
            crop_min_size=20,
            crop_max_size=20,
        )

        _, _, masks = transform(image, targets, (40, 40), segments)

        expected = cv2.resize(
            dense[10:30, 10:30],
            (40, 40),
            interpolation=cv2.INTER_NEAREST,
        )
        assert masks[0].sum() == pytest.approx(float(expected.sum()))
        assert np.array_equal(masks[0] > 0, expected > 0)

    def test_coco_dataset_preserves_multiple_segment_rings(self, tmp_path):
        import json

        from PIL import Image
        from libreyolo.data.dataset import COCODataset

        images_dir = tmp_path / "train2017"
        ann_dir = tmp_path / "annotations"
        images_dir.mkdir()
        ann_dir.mkdir()

        Image.new("RGB", (20, 10)).save(images_dir / "img.jpg")
        (ann_dir / "instances_train2017.json").write_text(
            json.dumps(
                {
                    "images": [
                        {
                            "id": 1,
                            "file_name": "img.jpg",
                            "width": 20,
                            "height": 10,
                        }
                    ],
                    "annotations": [
                        {
                            "id": 1,
                            "image_id": 1,
                            "category_id": 1,
                            "bbox": [1, 1, 13, 4],
                            "area": 32,
                            "iscrowd": 0,
                            "segmentation": [
                                [1, 1, 5, 1, 5, 5, 1, 5],
                                [10, 1, 14, 1, 14, 5, 10, 5],
                            ],
                        }
                    ],
                    "categories": [{"id": 1, "name": "cat"}],
                }
            )
        )

        dataset = COCODataset(
            data_dir=str(tmp_path),
            json_file="instances_train2017.json",
            name="train2017",
            img_size=(10, 20),
            load_segments=True,
        )

        assert len(dataset.segments[0][0]) == 2
        assert dataset.segments[0][0][0].tolist() == [
            [1.0, 1.0],
            [5.0, 1.0],
            [5.0, 5.0],
            [1.0, 5.0],
        ]
        assert dataset.segments[0][0][1].tolist() == [
            [10.0, 1.0],
            [14.0, 1.0],
            [14.0, 5.0],
            [10.0, 5.0],
        ]
        # Polygon-sourced rings store no eager dense mask (issue #270).
        assert getattr(dataset.segments[0][0][0], "dense_mask", None) is None

    def test_coco_dataset_decodes_rle_segments_when_requested(self, tmp_path):
        import json

        from PIL import Image
        from pycocotools import mask as mask_utils

        from libreyolo.data.dataset import COCODataset

        images_dir = tmp_path / "train2017"
        ann_dir = tmp_path / "annotations"
        images_dir.mkdir()
        ann_dir.mkdir()

        Image.new("RGB", (20, 10)).save(images_dir / "img.jpg")
        mask = np.zeros((10, 20), dtype=np.uint8)
        mask[2:6, 3:9] = 1
        rle = mask_utils.encode(np.asfortranarray(mask))
        rle["counts"] = rle["counts"].decode("ascii")

        (ann_dir / "instances_train2017.json").write_text(
            json.dumps(
                {
                    "images": [
                        {
                            "id": 1,
                            "file_name": "img.jpg",
                            "width": 20,
                            "height": 10,
                        }
                    ],
                    "annotations": [
                        {
                            "id": 1,
                            "image_id": 1,
                            "category_id": 1,
                            "bbox": [3, 2, 6, 4],
                            "area": int(mask.sum()),
                            "iscrowd": 0,
                            "segmentation": rle,
                        }
                    ],
                    "categories": [{"id": 1, "name": "cat"}],
                }
            )
        )

        dataset = COCODataset(
            data_dir=str(tmp_path),
            json_file="instances_train2017.json",
            name="train2017",
            img_size=(10, 20),
            load_segments=True,
        )

        rings = dataset.segments[0][0]
        assert rings
        assert sum(len(ring) for ring in rings) >= 4

    def test_coco_rle_segments_preserve_holes_for_rfdetr(self, tmp_path):
        import json

        from PIL import Image
        from pycocotools import mask as mask_utils

        from libreyolo.data.dataset import COCODataset
        from libreyolo.models.rfdetr.seg_transforms import RFDETRSegTransform

        images_dir = tmp_path / "train2017"
        ann_dir = tmp_path / "annotations"
        images_dir.mkdir()
        ann_dir.mkdir()

        Image.new("RGB", (12, 12)).save(images_dir / "img.jpg")
        mask = np.zeros((12, 12), dtype=np.uint8)
        mask[2:10, 2:10] = 1
        mask[5:8, 5:8] = 0
        rle = mask_utils.encode(np.asfortranarray(mask))
        rle["counts"] = rle["counts"].decode("ascii")

        (ann_dir / "instances_train2017.json").write_text(
            json.dumps(
                {
                    "images": [
                        {
                            "id": 1,
                            "file_name": "img.jpg",
                            "width": 12,
                            "height": 12,
                        }
                    ],
                    "annotations": [
                        {
                            "id": 1,
                            "image_id": 1,
                            "category_id": 1,
                            "bbox": [2, 2, 8, 8],
                            "area": int(mask.sum()),
                            "iscrowd": 0,
                            "segmentation": rle,
                        }
                    ],
                    "categories": [{"id": 1, "name": "cat"}],
                }
            )
        )

        dataset = COCODataset(
            data_dir=str(tmp_path),
            json_file="instances_train2017.json",
            name="train2017",
            img_size=(12, 12),
            load_segments=True,
        )
        transform = RFDETRSegTransform(max_labels=4, flip_prob=0.0, imgsz=12)

        _, _, masks = transform(
            np.array(Image.open(images_dir / "img.jpg"))[:, :, ::-1],
            dataset.annotations[0][0],
            (12, 12),
            dataset.segments[0],
        )

        assert masks[0, 3, 3] == 1
        assert masks[0, 6, 6] == 0

    def test_coco_rle_segments_preserve_sparse_masks_for_rfdetr(self):
        from pycocotools import mask as mask_utils

        from libreyolo.data.dataset import _coco_segmentation_to_rings
        from libreyolo.models.rfdetr.seg_transforms import RFDETRSegTransform

        mask = np.zeros((16, 16), dtype=np.uint8)
        np.fill_diagonal(mask, 1)
        rle = mask_utils.encode(np.asfortranarray(mask))
        rle["counts"] = rle["counts"].decode("ascii")

        segments = [
            _coco_segmentation_to_rings(
                rle,
                height=mask.shape[0],
                width=mask.shape[1],
            )
        ]
        image = np.zeros((16, 16, 3), dtype=np.uint8)
        targets = np.array([[0, 0, 16, 16, 0]], dtype=np.float32)
        transform = RFDETRSegTransform(max_labels=1, flip_prob=0.0, imgsz=16)

        _, _, masks = transform(image, targets, (16, 16), segments)

        np.testing.assert_array_equal(masks[0].astype(np.uint8), mask)


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

        from libreyolo.models.rfdetr.model import LibreRFDETR

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "seg_model.pt")
            torch.save(
                {"model": {"segmentation_head.weight": torch.zeros(1)}},
                path,
            )
            assert LibreRFDETR._detect_segmentation(path) is True

    def test_detect_det_from_checkpoint(self):
        """_detect_segmentation returns False for detection-only checkpoints."""
        import tempfile
        from pathlib import Path

        from libreyolo.models.rfdetr.model import LibreRFDETR

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "det_model.pt")
            torch.save(
                {"model": {"class_embed.weight": torch.zeros(1)}},
                path,
            )
            assert LibreRFDETR._detect_segmentation(path) is False

    def test_detect_seg_from_filename(self):
        """Filename-based detection avoids loading weights."""
        from libreyolo.models.rfdetr.model import LibreRFDETR

        assert LibreRFDETR.detect_task_from_filename("LibreRFDETRn-seg.pt") == "segment"
        assert LibreRFDETR.detect_task_from_filename("LibreRFDETRn.pt") is None

    def test_segmentation_flag_is_derived_from_task(self):
        from libreyolo.models.rfdetr.model import LibreRFDETR

        model = LibreRFDETR.__new__(LibreRFDETR)
        model.task = "segment"
        assert model._is_segmentation is True

        model.task = "detect"
        assert model._is_segmentation is False
        assert "_is_segmentation" not in model.__dict__

    def test_detect_size_uses_segmentation_position_embedding_tokens(self):
        from libreyolo.models.rfdetr.model import LibreRFDETR

        weights = {
            "segmentation_head.blocks.0.dwconv.weight": torch.zeros(1),
            "backbone.0.encoder.encoder.embeddings.position_embeddings": torch.zeros(1, 26 * 26 + 1, 384),
        }

        assert LibreRFDETR.detect_size(weights) == "n"

    def test_default_none_uses_upstream_pretrained_weight_name(self, monkeypatch):
        from libreyolo.models.rfdetr.model import LibreRFDETR

        monkeypatch.setattr(
            LibreRFDETR,
            "_resolve_weights_path",
            staticmethod(lambda name: f"resolved/{name}"),
        )
        monkeypatch.setattr(
            LibreRFDETR,
            "_detect_segmentation",
            staticmethod(lambda _path: False),
        )
        monkeypatch.setattr(LibreRFDETR, "_load_weights", lambda self, _path: None)

        model = LibreRFDETR(model_path=None, size="n", device="cpu")

        assert model._weight_source == "resolved/rf-detr-nano.pth"

    def test_default_none_uses_segmentation_pretrained_weight_name(self, monkeypatch):
        from libreyolo.models.rfdetr.model import LibreRFDETR

        monkeypatch.setattr(
            LibreRFDETR,
            "_resolve_weights_path",
            staticmethod(lambda name: f"resolved/{name}"),
        )
        monkeypatch.setattr(LibreRFDETR, "_load_weights", lambda self, _path: None)

        model = LibreRFDETR(
            model_path=None,
            size="n",
            segmentation=True,
            device="cpu",
        )

        assert model._weight_source == "resolved/rf-detr-seg-nano.pt"

    def test_upstream_seg_xlarge_configs_are_available(self):
        from libreyolo.models.rfdetr.model import LibreRFDETR
        from libreyolo.models.rfdetr.nn import RFDETR_SEG_CONFIGS

        assert LibreRFDETR.SEG_INPUT_SIZES["x"] == 624
        assert LibreRFDETR.SEG_INPUT_SIZES["xx"] == 768
        assert RFDETR_SEG_CONFIGS["x"].pretrain_weights == "rf-detr-seg-xlarge.pt"
        assert RFDETR_SEG_CONFIGS["xx"].pretrain_weights == "rf-detr-seg-xxlarge.pt"
        assert RFDETR_SEG_CONFIGS["x"].num_select == 300
        assert RFDETR_SEG_CONFIGS["xx"].num_select == 300


class TestRFDETRQueryLoading:
    """Tests for RF-DETR Group-DETR query tensor resizing."""

    def test_query_resize_preserves_per_group_layout(self):
        from libreyolo.models.rfdetr.nn import _slice_query_param_per_group

        tensor = torch.arange(6).view(6, 1)
        resized = _slice_query_param_per_group(
            tensor,
            ckpt_num_queries=3,
            ckpt_group_detr=2,
            target_num_queries=2,
            target_group_detr=2,
        )

        assert resized.squeeze(1).tolist() == [0, 1, 3, 4]

    def test_wrapper_preserves_checkpoint_args_for_model_load(self):
        from libreyolo.models.rfdetr.model import LibreRFDETR

        wrapper = LibreRFDETR(model_path={}, size="n", device="cpu")
        checkpoint = {"model": {}, "args": {"num_queries": 3, "group_detr": 2}}
        seen = {}

        def fake_load_state_dict(state_dict, strict=False):
            seen["state_dict"] = state_dict
            seen["strict"] = strict
            return [], []

        wrapper.model.load_state_dict = fake_load_state_dict

        wrapper._load_weights(checkpoint)

        assert seen["state_dict"] is checkpoint
        assert seen["strict"] is False


class TestRFDETRSegTrainer:
    """Tests for RF-DETR segmentation trainer plumbing."""

    def test_rfdetr_training_defaults_are_windows_safe(self):
        from libreyolo.models.rfdetr.config import RFDETRConfig

        assert RFDETRConfig().workers == 0

    def test_rfdetr_training_uses_upstream_lr_and_accumulation_defaults(self):
        from libreyolo.models.rfdetr.config import RFDETRConfig
        from libreyolo.models.rfdetr.trainer import RFDETRTrainer

        trainer = RFDETRTrainer.__new__(RFDETRTrainer)
        trainer.config = RFDETRConfig(batch=2, lr0=1e-4)

        assert trainer.config.nbs == 16
        assert trainer._accum_steps == 8
        assert trainer.config.scheduler == "step"
        assert trainer.config.lr_drop == 100
        assert trainer.effective_lr == pytest.approx(1e-4)

    def test_rfdetr_step_scheduler_matches_upstream_default(self):
        from libreyolo.models.rfdetr.config import RFDETRConfig
        from libreyolo.models.rfdetr.trainer import RFDETRTrainer

        trainer = RFDETRTrainer.__new__(RFDETRTrainer)
        trainer.config = RFDETRConfig(
            epochs=100,
            batch=4,
            nbs=16,
            lr0=1e-4,
            warmup_epochs=0,
            lr_drop=100,
        )

        scheduler = trainer.create_scheduler(iters_per_epoch=7)

        assert scheduler.update_lr(1) == pytest.approx(1e-4)
        assert scheduler.update_lr(699) == pytest.approx(1e-4)
        assert scheduler.update_lr(700) == pytest.approx(1e-5)

    def test_rfdetr_step_scheduler_warmup_uses_optimizer_steps(self):
        from libreyolo.models.rfdetr.config import RFDETRConfig
        from libreyolo.models.rfdetr.trainer import RFDETRTrainer

        trainer = RFDETRTrainer.__new__(RFDETRTrainer)
        trainer.config = RFDETRConfig(
            epochs=2,
            batch=4,
            nbs=16,
            lr0=1e-4,
            warmup_epochs=1,
            lr_drop=100,
        )

        scheduler = trainer.create_scheduler(iters_per_epoch=7)

        assert scheduler.update_lr(1) == pytest.approx(1e-4 / 7)
        assert scheduler.update_lr(7) == pytest.approx(1e-4)

    def test_rfdetr_detection_transform_uses_original_images(self):
        from libreyolo.models.rfdetr.config import RFDETRConfig
        from libreyolo.models.rfdetr.trainer import RFDETRTrainer

        trainer = RFDETRTrainer.__new__(RFDETRTrainer)
        trainer.config = RFDETRConfig(imgsz=384)
        trainer.model = type("Model", (), {"patch_size": 16, "num_windows": 2})()
        trainer.wrapper_model = type("Wrapper", (), {"task": "detect"})()

        preproc, _ = trainer.create_transforms()

        assert preproc.wants_unresized_image is True
        assert preproc.target_size == 544

    def test_rfdetr_trainer_applies_batch_multi_scale_to_images_boxes_and_masks(self):
        from types import MethodType

        from libreyolo.models.rfdetr.config import RFDETRConfig
        from libreyolo.models.rfdetr.trainer import RFDETRTrainer

        trainer = RFDETRTrainer.__new__(RFDETRTrainer)
        trainer.config = RFDETRConfig(imgsz=64, multi_scale=True)
        trainer._multi_scale_scales = MethodType(lambda self: [32], trainer)

        imgs = torch.ones(2, 3, 64, 64)
        targets = torch.zeros(2, 4, 5)
        targets[:, 0] = torch.tensor([1.0, 32.0, 32.0, 16.0, 16.0])
        masks = torch.zeros(2, 4, 64, 64)
        masks[:, 0, 16:48, 16:48] = 1

        imgs_out, targets_out, masks_out = trainer._apply_multi_scale_batch(
            imgs,
            targets,
            masks,
            step=0,
        )

        assert imgs_out.shape[-2:] == (32, 32)
        assert masks_out.shape[-2:] == (32, 32)
        expected = torch.tensor(
            [[1.0, 16.0, 16.0, 8.0, 8.0], [1.0, 16.0, 16.0, 8.0, 8.0]]
        )
        assert torch.allclose(targets_out[:, 0], expected)
        assert masks_out[:, 0].sum().item() == pytest.approx(2 * 16 * 16)

    def test_rfdetr_optimizer_uses_upstream_param_lrs(self):
        from types import SimpleNamespace

        from libreyolo.models.rfdetr.config import RFDETRConfig
        from libreyolo.models.rfdetr.trainer import RFDETRTrainer

        class FakeBackboneEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder_weight = torch.nn.Parameter(torch.ones(()))

            def get_named_param_lr_pairs(self, args, prefix="backbone.0"):
                lr = args.lr_encoder * args.lr_component_decay**2
                return {
                    f"{prefix}.encoder_weight": {
                        "params": self.encoder_weight,
                        "lr": lr,
                        "weight_decay": args.weight_decay,
                    }
                }

        class FakeCore(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = torch.nn.ModuleList([FakeBackboneEncoder()])
                self.transformer = torch.nn.Module()
                self.transformer.decoder = torch.nn.Linear(1, 1)
                self.head = torch.nn.Linear(1, 1)

        class FakeWrapper(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = FakeCore()
                self.args = SimpleNamespace(
                    lr_encoder=2e-4,
                    lr_component_decay=0.5,
                    lr_vit_layer_decay=0.8,
                    out_feature_indexes=[0],
                    weight_decay=1e-4,
                )

        trainer = RFDETRTrainer.__new__(RFDETRTrainer)
        trainer.config = RFDETRConfig(lr0=1e-4, weight_decay=0.01)
        trainer.model = FakeWrapper()

        optimizer = trainer._setup_optimizer()
        groups_by_param = {
            id(group["params"][0]): group
            for group in optimizer.param_groups
        }
        core = trainer.model.model
        backbone_group = groups_by_param[id(core.backbone[0].encoder_weight)]
        decoder_group = groups_by_param[id(core.transformer.decoder.weight)]
        head_group = groups_by_param[id(core.head.weight)]

        assert backbone_group["lr"] == pytest.approx(5e-5)
        assert backbone_group["weight_decay"] == pytest.approx(0.01)
        assert backbone_group["lr_mult"] == pytest.approx(0.5)
        assert decoder_group["lr"] == pytest.approx(5e-5)
        assert decoder_group["lr_mult"] == pytest.approx(0.5)
        assert head_group["lr"] == pytest.approx(1e-4)
        assert head_group["lr_mult"] == pytest.approx(1.0)
        assert trainer._scale_lr(1e-5, backbone_group) == pytest.approx(5e-6)

    def test_rfdetr_validation_defaults_are_windows_safe(self):
        import inspect

        from libreyolo.models.rfdetr.model import LibreRFDETR

        assert inspect.signature(LibreRFDETR.val).parameters["workers"].default == 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
    def test_seg_masks_move_to_training_device_before_indexing(self):
        from libreyolo.models.rfdetr.trainer import RFDETRTrainer

        class DummyModel(torch.nn.Module):
            def forward(self, imgs, targets=None):
                return {
                    "loss": torch.ones(
                        (), device=imgs.device, requires_grad=True
                    )
                }

        class DummyCriterion:
            weight_dict = {"loss": 1.0}

            def __call__(self, outputs, targets):
                self.targets = targets
                return {"loss": outputs["loss"]}

        trainer = object.__new__(RFDETRTrainer)
        trainer.device = torch.device("cuda")
        trainer.wrapper_model = type("Wrapper", (), {"task": "segment"})()
        trainer.model = DummyModel().to(trainer.device)
        trainer.criterion = DummyCriterion()

        imgs = torch.zeros(1, 3, 8, 8, device=trainer.device)
        targets = torch.tensor(
            [[[0.0, 4.0, 4.0, 2.0, 2.0]]],
            device=trainer.device,
        )
        cpu_masks = torch.ones(1, 1, 8, 8)

        out = trainer.on_forward(imgs, targets, polygons=cpu_masks)

        assert out["total_loss"].device.type == "cuda"
        assert trainer.criterion.targets[0]["masks"].device.type == "cuda"


class TestPolygonToCxcywh:
    """Tests for the polygon_to_cxcywh shared utility."""

    def test_square(self):
        coords = [0.1, 0.1, 0.3, 0.1, 0.3, 0.3, 0.1, 0.3]
        cx, cy, w, h = polygon_to_cxcywh(coords)
        assert cx == pytest.approx(0.2)
        assert cy == pytest.approx(0.2)
        assert w == pytest.approx(0.2)
        assert h == pytest.approx(0.2)

    def test_triangle(self):
        # Triangle: (0.2,0.2), (0.8,0.2), (0.5,0.8)
        coords = [0.2, 0.2, 0.8, 0.2, 0.5, 0.8]
        cx, cy, w, h = polygon_to_cxcywh(coords)
        assert cx == pytest.approx(0.5)
        assert cy == pytest.approx(0.5)
        assert w == pytest.approx(0.6)
        assert h == pytest.approx(0.6)

    def test_single_point(self):
        # Degenerate: single point → zero-size box
        coords = [0.5, 0.5]
        cx, cy, w, h = polygon_to_cxcywh(coords)
        assert cx == pytest.approx(0.5)
        assert cy == pytest.approx(0.5)
        assert w == pytest.approx(0.0)
        assert h == pytest.approx(0.0)

    def test_horizontal_line(self):
        # Two points on horizontal line → zero height
        coords = [0.1, 0.5, 0.9, 0.5]
        cx, cy, w, h = polygon_to_cxcywh(coords)
        assert cx == pytest.approx(0.5)
        assert cy == pytest.approx(0.5)
        assert w == pytest.approx(0.8)
        assert h == pytest.approx(0.0)
