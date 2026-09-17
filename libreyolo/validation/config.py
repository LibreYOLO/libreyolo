"""Validation configuration for LibreYOLO."""

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

import yaml

from libreyolo.utils.amp import normalize_amp_dtype


@dataclass
class ValidationConfig:
    """
    Configuration for model validation.

    Attributes:
        data: Path to data.yaml file containing dataset configuration.
        data_dir: Direct path to dataset directory (alternative to data).
        split: Dataset split to validate on ("val" or "test").
        single_cls: Evaluate every detection category as class 0.
        batch_size: Batch size for validation.
        imgsz: Image size for validation. Accepts an int (square) or (height, width) tuple.
        conf_thres: Confidence threshold. Use 0.0 or a low value for mAP calculation.
        iou_thres: IoU threshold for NMS.
        max_det: Maximum predictions per image after postprocessing.
        eval_max_det: Optional maximum detections used by COCO evaluation.
            None preserves pycocotools' default AP@100 behavior.
        iou_thresholds: IoU thresholds for mAP calculation (default: 0.50 to 0.95).
        device: Device to use ("auto", "cuda", "mps", "cpu").
        save_dir: Directory to save results.
        save_json: Whether to write the detections as a standard COCO
            results JSON (list of image_id/category_id/bbox/score entries,
            loadable by pycocotools loadRes) using the original dataset
            image and category ids. Detection writes save_dir/predictions.json;
            segmentation writes predictions_bbox.json and predictions_masks.json.
            Tasks without COCO detections ignore the flag (OBB logs a warning).
        save_plots: Whether to save validation plots (metrics bar, per-class AP,
            confusion matrix, sample images). Default False.
        verbose: Whether to print detailed metrics.
        num_workers: Number of dataloader workers.
        half: Whether to use FP16 inference.
        amp_dtype: Mixed-precision dtype used when half is enabled.
        faster_coco_eval: Use the faster-coco-eval C++ backend for COCO
            metrics (bbox/segm). On by default; falls back to pycocotools
            with a warning if the faster-coco-eval package is unavailable.
            Set False (or LIBREYOLO_FASTER_COCO_EVAL=0) to force pycocotools.
    """

    # Data
    data: Optional[str] = None
    data_dir: Optional[str] = None
    split: str = "val"
    single_cls: bool = field(default=False, kw_only=True)
    # Auto-inherited from the checkpoint's saved training config the same
    # way single_cls is (see DetectionValidator.__init__); rarely set by
    # hand. Must match the classes= the model was trained with, since the
    # head size is shared.
    classes: Optional[List[int]] = field(default=None, kw_only=True)

    # Inference
    batch_size: int = 16
    imgsz: Union[int, Tuple[int, int]] = 640
    conf_thres: float = 0.001
    iou_thres: float = 0.6
    max_det: int = 300
    eval_max_det: Optional[int] = field(default=None, kw_only=True)

    # Metrics
    # NOTE: iou_thresholds is only honored on the OBB validation path. The COCO
    # (detect/segment) path evaluates through pycocotools, which is locked to
    # its own default IoU sweep (0.50:0.05:0.95) via coco_eval.params.iouThrs
    # and ignores this value.
    iou_thresholds: Tuple[float, ...] = (
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
        0.95,
    )

    # Device
    device: str = "auto"

    # Output
    save_dir: Optional[str] = None
    save_json: bool = False
    verbose: bool = True
    save_plots: bool = field(default=False, kw_only=True)

    # Workers
    num_workers: int = 4

    # Image cache: False/None (off), True/'ram', or 'disk'. Same semantics as
    # TrainConfig.cache. Validation preprocessing is deterministic, so cached
    # reads are byte-identical to fresh ones; 'disk' pays immediately, 'ram'
    # pays once the validator (and its workers) survive across epochs.
    cache: Union[bool, str] = field(default=False, kw_only=True)

    # Replay the model forward through captured CUDA graphs (True, False or
    # 'auto'), using the same per-shape cache predict() uses. Validation is
    # the ideal capture client: every batch has one shape (plus one partial
    # final batch), and small detectors spend more host time launching the
    # forward's kernels than the GPU spends running them. Replay is
    # bit-identical to the eager forward (gated by tests/unit/test_cuda_graph
    # .py); anything uncapturable falls back to eager with a warning.
    cuda_graph: Union[bool, str] = field(default=False, kw_only=True)

    # Precision
    half: bool = False
    amp_dtype: str = field(default="float16", kw_only=True)
    allow_download_scripts: bool = False

    # COCO eval backend. On by default: uses the faster-coco-eval C++
    # evaluator when installed (metrics identical to pycocotools within
    # float64 summation order), silently-audited fallback to pycocotools
    # otherwise. The backend actually used is logged and surfaced as
    # model.last_eval_backend.
    faster_coco_eval: bool = True

    # TTA
    augment: bool = False

    # Pose validation
    keypoints_json: Optional[str] = None
    images_dir: Optional[str] = None
    oks_sigmas: Optional[List[float]] = None

    # Point validation
    dist_thresholds: Optional[List[float]] = None

    # Edge validation (BSDS-style normalized correspondence tolerance)
    edge_max_dist: float = 0.0075
    edge_thresholds: Tuple[float, ...] = field(
        default_factory=lambda: tuple(index / 100.0 for index in range(1, 100))
    )

    def __post_init__(self) -> None:
        self.amp_dtype = normalize_amp_dtype(self.amp_dtype)
        self.single_cls = bool(self.single_cls)

        if self.data is None and self.data_dir is None and self.keypoints_json is None:
            raise ValueError(
                "Specify one of: 'data' (yaml), 'data_dir' (detection/segmentation), "
                "or 'data' / 'keypoints_json' + 'images_dir' (pose)"
            )

        if self.split not in ("val", "test", "train"):
            raise ValueError(
                f"Invalid split: {self.split}. Must be 'val', 'test', or 'train'"
            )

        if not 0 <= self.conf_thres < 1:
            raise ValueError(f"conf_thres must be in [0, 1), got {self.conf_thres}")

        if not 0 < self.iou_thres < 1:
            raise ValueError(f"iou_thres must be in (0, 1), got {self.iou_thres}")

        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")

        if not 0.0 <= self.edge_max_dist <= 1.0:
            raise ValueError(
                f"edge_max_dist must be in [0, 1], got {self.edge_max_dist}"
            )
        self.edge_thresholds = tuple(float(value) for value in self.edge_thresholds)
        if not self.edge_thresholds or any(
            not 0.0 <= value <= 1.0 for value in self.edge_thresholds
        ):
            raise ValueError(
                "edge_thresholds must contain one or more values in [0, 1]"
            )

        if self.max_det < 1:
            raise ValueError(f"max_det must be >= 1, got {self.max_det}")
        if self.eval_max_det is not None:
            self.eval_max_det = int(self.eval_max_det)
            if self.eval_max_det < 1:
                raise ValueError(
                    f"eval_max_det must be >= 1, got {self.eval_max_det}"
                )

    @property
    def effective_eval_max_det(self) -> int:
        """Return the explicit COCO cap or pycocotools' historical default."""
        return 100 if self.eval_max_det is None else self.eval_max_det

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "ValidationConfig":
        """Load configuration from a YAML file."""
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)

        if "iou_thresholds" in config_dict and isinstance(
            config_dict["iou_thresholds"], list
        ):
            config_dict["iou_thresholds"] = tuple(config_dict["iou_thresholds"])
        if "edge_thresholds" in config_dict and isinstance(
            config_dict["edge_thresholds"], list
        ):
            config_dict["edge_thresholds"] = tuple(config_dict["edge_thresholds"])

        return cls(**config_dict)

    def to_yaml(self, path: Union[str, Path]) -> None:
        """Save configuration to a YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        config_dict = self.to_dict()
        config_dict["iou_thresholds"] = list(config_dict["iou_thresholds"])
        config_dict["edge_thresholds"] = list(config_dict["edge_thresholds"])

        with open(path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return asdict(self)

    def update(self, **kwargs) -> "ValidationConfig":
        """Create new configuration with updated values."""
        current = self.to_dict()
        current.update(kwargs)

        if isinstance(current.get("iou_thresholds"), list):
            current["iou_thresholds"] = tuple(current["iou_thresholds"])
        if isinstance(current.get("edge_thresholds"), list):
            current["edge_thresholds"] = tuple(current["edge_thresholds"])

        return ValidationConfig(**current)
