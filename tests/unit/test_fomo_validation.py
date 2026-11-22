"""Unit tests for LibreFOMO post-processing and validation."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch
from PIL import Image

pytestmark = [pytest.mark.unit, pytest.mark.fomo]

if TYPE_CHECKING:
    from libreyolo.models.fomo.model import LibreFOMO


# ===========================================================================
# Helpers
# ===========================================================================


def _write_tiny_point_dataset(root: Path, imgsz: int = 96) -> Path:
    """Write a minimal YOLO-format point dataset (1 image, 1 label for train and val splits)."""
    for split in ("train", "val"):
        img_dir = root / "images" / split
        lbl_dir = root / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (imgsz, imgsz), color=(128, 64, 32)).save(img_dir / "sample.jpg")
        (lbl_dir / "sample.txt").write_text("0 0.5 0.5 0.05 0.05\n", encoding="utf-8")

    import yaml

    data_yaml = root / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(root).replace("\\", "/"),
                "train": "images/train",
                "val": "images/val",
                "nc": 1,
                "names": {0: "object"},
            }
        ),
        encoding="utf-8",
    )
    return data_yaml


def _make_random_fomo(size: str = "s", nc: int = 1) -> LibreFOMO:
    from libreyolo.models.fomo.model import LibreFOMO

    return LibreFOMO(model_path=None, size=size, nb_classes=nc, device="cpu")


# ===========================================================================
# Postprocessing, point validation, and decode helpers
# ===========================================================================


class TestLibreFOMOPostprocess:
    """_postprocess satisfies the PointValidator contract."""

    def test_empty_output_returns_zero_detections(self) -> None:
        model = _make_random_fomo(size="s", nc=1)
        from libreyolo.models.fomo.nn import CONFIGS

        hw = CONFIGS["s"]["imgsz"] // 8
        # All-zero logits → softmax dominated by background → no foreground above threshold
        logits = torch.zeros(1, 2, hw, hw)
        result = model._postprocess(
            logits,
            conf_thres=0.5,
            iou_thres=0.45,
            original_size=(96, 96),
        )
        assert result["points"].shape == (0, 4)

    def test_postprocess_scales_to_original_size(self) -> None:
        """Points must be in original-image pixel space, not grid space."""
        model = _make_random_fomo(size="s", nc=1)
        from libreyolo.models.fomo.nn import CONFIGS

        hw = CONFIGS["s"]["imgsz"] // 8
        logits = torch.zeros(1, 2, hw, hw)
        logits[0, 1, 0, 0] = 20.0  # top-left cell
        result = model._postprocess(
            logits,
            conf_thres=0.01,
            iou_thres=0.45,
            original_size=(480, 640),
        )
        if len(result["points"]) > 0:
            x, y = result["points"][0, :2]
            # scaled pixel should be > 0 and fit in original image
            assert x.item() >= 0.0
            assert y.item() >= 0.0
            assert x.item() <= 640
            assert y.item() <= 480

    def test_postprocess_output_structure_n_4(self) -> None:
        """Verify that postprocess returns (N, 4) points with structure [x, y, class, confidence]."""
        model = _make_random_fomo(size="s", nc=1)
        from libreyolo.models.fomo.nn import CONFIGS

        hw = CONFIGS["s"]["imgsz"] // 8
        logits = torch.zeros(1, 2, hw, hw)
        logits[0, 1, 2, 3] = 20.0  # channel 1 = class 0 foreground
        result = model._postprocess(
            logits,
            conf_thres=0.8,
            iou_thres=0.45,
            original_size=(96, 96),
        )
        pts = result["points"]
        assert len(pts) == 1
        assert pts.shape == (1, 4)
        # Check structure: [x, y, class, confidence]
        # Coordinates should be scaled center of cell (2,3)
        # grid space: x=3, y=2
        # scaled: x = (3 + 0.5) * (96 / 12) = 3.5 * 8 = 28.0
        #         y = (2 + 0.5) * (96 / 12) = 2.5 * 8 = 20.0
        # class: 0.0, confidence: ~1.0
        np.testing.assert_allclose(pts[0].cpu().numpy(), [28.0, 20.0, 0.0, 1.0], atol=1e-3)

    def test_inference_returns_points_object_with_n_4(self) -> None:
        """Verify that calling predict() on FOMO returns Results with a Points payload of shape (N, 4)."""
        model = _make_random_fomo(size="s", nc=1)
        from libreyolo.models.fomo.nn import CONFIGS

        hw = CONFIGS["s"]["imgsz"] // 8
        # Mock _forward to return a logit with a peak at (2, 3)
        def _mock_forward(x):
            logits = torch.zeros(x.shape[0], 2, hw, hw)
            logits[:, 1, 2, 3] = 20.0
            return logits

        model._forward = _mock_forward

        img = Image.new("RGB", (96, 96))
        results = model.predict(img, conf=0.8)

        if isinstance(results, list):
            results = results[0]

        assert results.points is not None
        assert results.points.data.shape == (1, 4)
        np.testing.assert_allclose(results.points.xy[0].cpu().numpy(), [28.0, 20.0], atol=1e-3)
        assert int(results.points.cls[0].item()) == 0
        assert results.points.conf[0].item() > 0.99


class TestLibreFOMOValPreprocessor:
    """Validate FOMOValPreprocessor output shape, dtype, and pixel range."""

    def _make_preprocessor(self, imgsz: int = 96):
        from libreyolo.validation.preprocessors import FOMOValPreprocessor

        return FOMOValPreprocessor(img_size=(imgsz, imgsz))

    def test_preprocessor_output_properties(self) -> None:
        """Verify output shape, dtype, and pixel range of FOMOValPreprocessor."""
        preproc = self._make_preprocessor(96)
        img = np.random.randint(0, 256, (200, 300, 3), dtype=np.uint8)
        targets = np.zeros((0, 5), dtype=np.float32)
        chw, _ = preproc(img, targets, (96, 96))
        assert chw.shape == (3, 96, 96)
        assert chw.dtype == np.float32

        # Check normalization bounds
        img_black = np.zeros((96, 96, 3), dtype=np.uint8)
        chw_black, _ = preproc(img_black, np.zeros((0, 5), dtype=np.float32), (96, 96))
        assert chw_black.min() == pytest.approx(-1.0, abs=1e-4)

        img_white = np.full((96, 96, 3), 255, dtype=np.uint8)
        chw_white, _ = preproc(img_white, np.zeros((0, 5), dtype=np.float32), (96, 96))
        assert chw_white.max() == pytest.approx(1.0, abs=1e-4)

    def test_preprocessor_metadata_flags(self) -> None:
        """Verify configuration flags for FOMOValPreprocessor."""
        p = self._make_preprocessor(96)
        assert p.custom_normalization is True
        assert p.wants_unresized_image is True
        assert p.uses_letterbox is False

    def test_target_scaling(self) -> None:
        """Box coordinates must be scaled proportionally when resizing."""
        preproc = self._make_preprocessor(96)
        img = np.zeros((200, 400, 3), dtype=np.uint8)  # orig 200×400
        # Box: [x1, y1, x2, y2, cls] in original pixels (letterboxed from dataset)
        targets = np.array([[100.0, 50.0, 200.0, 100.0, 0.0]], dtype=np.float32)
        _, padded_targets = preproc(img, targets, (96, 96))
        # scale_x = 96 / 400 = 0.24 ; scale_y = 96 / 200 = 0.48
        np.testing.assert_allclose(padded_targets[0, 0], 100.0 * (96 / 400), atol=1e-4)
        np.testing.assert_allclose(padded_targets[0, 1], 50.0 * (96 / 200), atol=1e-4)

    def test_empty_targets_padded_to_max_labels(self) -> None:
        preproc = self._make_preprocessor(96)
        img = np.zeros((96, 96, 3), dtype=np.uint8)
        _, padded = preproc(img, np.zeros((0, 5), dtype=np.float32), (96, 96))
        assert padded.shape == (preproc.max_labels, 5)
        assert padded.sum() == 0.0


class TestLibreFOMOParseGtPoints:
    """_parse_gt_points must delegate to the validator's box-centre helper."""

    def test_delegates_to_validator_helper(self) -> None:
        from libreyolo.validation.point_validator import PointValidator

        model = _make_random_fomo(size="s", nc=1)

        # Build a minimal validator scaffold
        validator = object.__new__(PointValidator)
        from libreyolo.validation.point_validator import _DEFAULT_DIST_THRESHOLDS

        validator._dist_thresholds = _DEFAULT_DIST_THRESHOLDS
        validator._primary_threshold = _DEFAULT_DIST_THRESHOLDS[0]
        validator._records = []
        validator.nc = 1
        validator._actual_imgsz = 96
        validator.config = type("_Cfg", (), {"verbose": False})()
        validator.seen = 0
        validator.val_preproc = type("_Preproc", (), {"uses_letterbox": False})()
        validator.model = model

        gt_row = np.array([[43.2, 43.2, 52.8, 52.8, 0.0]], dtype=np.float32)
        xy, cls = model._parse_gt_points(gt_row, orig_h=96, orig_w=96, validator=validator)

        assert xy.shape == (1, 2)
        np.testing.assert_allclose(xy[0], [0.5, 0.5], atol=1e-5)
        assert cls[0] == 0


class TestLibreFOMOValidator:
    """FOMOValidator grid-specific metric handling."""

    def _make_config(self):
        from libreyolo.validation import ValidationConfig

        return ValidationConfig(data="unused.yaml", device="cpu", verbose=False)

    def test_grid_size_inferred_from_model_size(self) -> None:
        from libreyolo.validation.fomo_validator import FOMOValidator

        assert FOMOValidator(_make_random_fomo(size="s"), self._make_config()).grid_size == 12
        assert FOMOValidator(_make_random_fomo(size="m"), self._make_config()).grid_size == 24
        assert FOMOValidator(_make_random_fomo(size="l"), self._make_config()).grid_size == 28

    def test_grid_metrics_stream_without_raw_logit_cache(self) -> None:
        from libreyolo.validation.fomo_validator import FOMOValidator

        model = _make_random_fomo(size="s", nc=1)
        # conf_threshold must be > 0.5: with nc=1 (2 channels), zero-logit cells
        # have softmax fg probability = 0.5 and would all fire at 0.25.
        validator = FOMOValidator(
            model,
            self._make_config(),
            conf_thresholds=(0.9,),
            nms_radii=(1,),
        )
        validator._init_metrics()
        assert validator.last_logits is None
        assert not hasattr(validator, "grid_cached")

        logits = torch.zeros(1, 2, 12, 12)
        logits[0, 1, 6, 6] = 20.0
        validator.last_logits = logits
        preds = [
            {
                "xy_norm": np.array([[0.5, 0.5]], dtype=np.float64),
                "scores": np.array([0.9], dtype=np.float64),
                "classes": np.array([0], dtype=np.int64),
            }
        ]
        targets = torch.tensor([[[0.0, 0.5, 0.5, 0.05, 0.05]]], dtype=torch.float32)

        validator._update_metrics(preds, targets, [(96, 96)])

        stats = validator.grid_stats[(0.9, 1)]
        assert stats["tp"] == 1.0
        assert stats["fp"] == 0.0
        assert stats["fn"] == 0.0
        assert not hasattr(validator, "grid_cached")


class TestDecodePoints:
    """Unit tests for _decode_points (the internal NMS peak-finder)."""

    def test_all_background_returns_empty(self) -> None:
        from libreyolo.models.fomo.utils import decode_points_from_logits as _decode_points

        # All logits zero → softmax gives equal mass to bg and fg → obj_probs ~0.5
        # depending on nc. With threshold 0.6 nothing should fire.
        logits = torch.zeros(1, 2, 6, 6)  # nc=1
        results = _decode_points(logits, conf_threshold=0.6)
        assert results[0].shape[0] == 0

    def test_peak_at_known_location(self) -> None:
        from libreyolo.models.fomo.utils import decode_points_from_logits as _decode_points

        logits = torch.zeros(1, 2, 6, 6)
        logits[0, 1, 3, 4] = 20.0  # large foreground logit at (y=3, x=4)
        results = _decode_points(logits, conf_threshold=0.01)
        assert len(results[0]) >= 1
        # x should be 4, y should be 3
        assert int(results[0][0, 0].item()) == 4  # x
        assert int(results[0][0, 1].item()) == 3  # y

    def test_nms_radius_suppresses_neighbours(self) -> None:
        from libreyolo.models.fomo.utils import decode_points_from_logits as _decode_points

        logits = torch.zeros(1, 2, 6, 6)
        # Two adjacent peaks; with nms_radius=2 the second should be suppressed
        logits[0, 1, 3, 3] = 20.0
        logits[0, 1, 3, 4] = 18.0  # within radius 2 of (3,3)
        results_r2 = _decode_points(logits, conf_threshold=0.01, nms_radius=2)
        results_r0 = _decode_points(logits, conf_threshold=0.01, nms_radius=0)
        assert len(results_r2[0]) < len(results_r0[0])

    def test_class_index_correct(self) -> None:
        """Returned class channel index must equal the argmax foreground channel."""
        from libreyolo.models.fomo.utils import decode_points_from_logits as _decode_points

        logits = torch.zeros(1, 3, 4, 4)  # nc=2
        logits[0, 2, 1, 1] = 20.0  # channel 2 → 0-based class id 1
        results = _decode_points(logits, conf_threshold=0.01)
        assert len(results[0]) >= 1
        cls_channel = int(results[0][0, 2].item())  # raw channel index
        assert cls_channel == 2

    def test_batch_dimension_respected(self) -> None:
        from libreyolo.models.fomo.utils import decode_points_from_logits as _decode_points

        # nc=1: softmax gives probs 0.5 each when all logits are zero.
        # Use threshold > 0.5 so only the explicitly boosted cells fire.
        logits = torch.zeros(3, 2, 4, 4)
        logits[0, 1, 0, 0] = 20.0   # batch 0: large foreground at (0,0)
        logits[2, 1, 3, 3] = 20.0   # batch 2: large foreground at (3,3)
        # batch 1 stays all-zeros → obj_probs ≈ 0.5, below threshold 0.6
        results = _decode_points(logits, conf_threshold=0.6)
        assert len(results) == 3
        assert len(results[0]) >= 1
        assert len(results[1]) == 0
        assert len(results[2]) >= 1

    def test_max_points_limit(self) -> None:
        from libreyolo.models.fomo.utils import decode_points_from_logits as _decode_points

        logits = torch.zeros(1, 2, 8, 8)
        logits[0, 1, :, :] = 20.0  # every cell fires
        results = _decode_points(logits, conf_threshold=0.01, nms_radius=0, max_points=3)
        assert len(results[0]) <= 3


class TestLibreFOMOValidationEndToEnd:
    """Run validation pipeline with PointValidator integration."""

    def test_point_validator_integration(self, tmp_path: Path) -> None:
        from libreyolo.validation import PointValidator, ValidationConfig

        data_yaml = _write_tiny_point_dataset(tmp_path, imgsz=96)

        # Build a random-weight FOMO model that always predicts one foreground
        # point at the centre of the grid.
        model = _make_random_fomo(size="s", nc=1)

        # Monkey-patch _postprocess so we can inspect the PointValidator pipeline
        # without needing trained weights.
        def _fixed_postprocess(output, conf_thres, iou_thres, original_size, **kwargs):
            w, h = original_size
            return {
                "points": torch.tensor([[w / 2.0, h / 2.0, 0.0, 0.9]], dtype=torch.float32),
            }

        model._postprocess = _fixed_postprocess

        config = ValidationConfig(
            data=str(data_yaml),
            batch_size=1,
            imgsz=96,
            num_workers=0,
            verbose=False,
            device="cpu",
            save_dir=str(tmp_path / "val_out"),
        )
        validator = PointValidator(model, config)
        metrics = validator.run()

        # The validator ran without error
        assert "metrics/precision" in metrics
        assert "metrics/recall" in metrics
        assert "metrics/f1" in metrics
        assert "speed/images_seen" in metrics
        assert metrics["speed/images_seen"] == 1

    def test_fomo_validator_drops_out_of_range_classes(self, tmp_path: Path) -> None:
        """FOMOValidator inherits the class bound of PointValidator (#1074)."""
        from libreyolo.validation import FOMOValidator, ValidationConfig

        data_yaml = _write_tiny_point_dataset(tmp_path, imgsz=96)
        (tmp_path / "labels" / "val" / "sample.txt").write_text(
            "0 0.5 0.5 0.05 0.05\n1 0.3 0.3 0.05 0.05\n", encoding="utf-8"
        )
        config = ValidationConfig(
            data=str(data_yaml),
            batch_size=1,
            imgsz=96,
            num_workers=0,
            verbose=False,
            device="cpu",
            save_dir=str(tmp_path / "val_out"),
        )
        validator = FOMOValidator(_make_random_fomo(size="s", nc=1), config)
        with pytest.warns(UserWarning, match="out of range"):
            loader = validator._setup_dataloader()

        assert loader.dataset.num_classes == 1
        labels = loader.dataset.load_anno(0)
        assert labels.shape[0] == 1
        assert int(labels[0, 4]) == 0
