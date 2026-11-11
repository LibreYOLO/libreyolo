"""Base validator class for LibreYOLO."""

import copy
import logging
import sys
import time
from abc import ABC, abstractmethod
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from libreyolo.utils.amp import torch_amp_dtype

from .config import VISUALIZE_TASKS, ValidationConfig

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from libreyolo.models.base import BaseModel

    from .confusion_matrix import ConfusionMatrix


def _upcast_bfloat16(value: Any) -> Any:
    """Return ``value`` with every bfloat16 tensor cast to float32.

    NumPy has no bfloat16, so postprocessors that call ``.numpy()`` on raw
    outputs fail when validation runs under bf16 autocast (#940).

    Containers without bfloat16 tensors are returned as the same object, and
    dict subclasses keep their type, so non-bf16 validation is untouched.
    """
    if isinstance(value, torch.Tensor):
        return value.float() if value.dtype == torch.bfloat16 else value
    if isinstance(value, dict):
        items = {key: _upcast_bfloat16(item) for key, item in value.items()}
        if all(items[key] is item for key, item in value.items()):
            return value
        upcast = copy.copy(value)
        upcast.update(items)
        return upcast
    if isinstance(value, (list, tuple)):
        items = [_upcast_bfloat16(item) for item in value]
        if all(new is old for new, old in zip(items, value)):
            return value
        if isinstance(value, tuple) and hasattr(value, "_fields"):
            return type(value)(*items)
        return type(value)(items)
    return value


def _empty_curve() -> np.ndarray:
    return np.zeros((0, 0), dtype=np.float64)


class _MapMetrics:
    """``map``, ``map50``, ``map75`` and ``maps`` for one kind.

    Read from the flat ``val()`` metrics: the ``(B)`` (box) or ``(M)`` (mask)
    keys, falling back to the unsuffixed key for detection, where it is the
    box value. ``maps`` is AP50-95 per class index, ``nc`` long; classes
    without ground truth in the validation set get the overall ``map``.
    """

    _suffix = "(B)"

    def _init_map_metrics(
        self,
        metrics: Optional[Dict[str, Any]],
        per_class_ap: Optional[Dict[int, float]] = None,
        nc: Optional[int] = None,
    ) -> None:
        self._metrics = metrics if metrics is not None else {}
        self._per_class_ap = dict(per_class_ap or {})
        self._nc = nc

    def _value(self, name: str) -> float:
        metrics = self._metrics
        key = f"metrics/{name}"
        value = metrics.get(key + self._suffix)
        if value is None and not any(k.endswith("(M)") for k in metrics):
            # Detection: unsuffixed keys are the box metrics. Segmentation's
            # unsuffixed keys are mask metrics, so box never falls back there.
            value = metrics.get(key)
        return float("nan") if value is None else float(value)

    @property
    def map(self) -> float:
        """mAP at IoU 0.50:0.95 (``metrics/mAP50-95``)."""
        return self._value("mAP50-95")

    @property
    def map50(self) -> float:
        """mAP at IoU 0.50 (``metrics/mAP50``)."""
        return self._value("mAP50")

    @property
    def map75(self) -> float:
        """mAP at IoU 0.75 (``metrics/mAP75``)."""
        return self._value("mAP75")

    @property
    def maps(self) -> np.ndarray:
        """AP50-95 per class index; the overall ``map`` where a class has no AP."""
        nc = self._nc
        if nc is None:
            nc = max(self._per_class_ap, default=-1) + 1
        out = np.full(int(nc), self.map, dtype=np.float64)
        for label, ap in self._per_class_ap.items():
            if 0 <= label < len(out):
                out[label] = ap
        return out


class BoxImageMetrics(_MapMetrics):
    """``results.box``: per-image and per-class results for detect and segment.

    ``image_metrics`` (#887) maps each image filename to its box metrics.
    ``best_conf_per_class`` maps each class name to its F1-optimal confidence
    threshold (IoU 0.50 matching), 0.0 for classes where no threshold reaches
    F1 > 0; empty when the validator does not run the sweep (segmentation).

    The curve attributes (#928) describe each class across confidence
    thresholds, at IoU 0.50 matching. ``px`` holds the confidences (1000
    values over [0, 1]); ``p_curve``, ``r_curve`` and ``f1_curve`` are
    ``(classes, len(px))`` arrays of precision, recall and F1, one row per
    entry of ``ap_class_index`` (the class indices with ground truth, in
    ascending order). ``p``, ``r`` and ``f1`` are the per-class values at the
    confidence that maximizes the mean F1 over those classes. All are empty
    when the validator produced no curves.
    """

    def __init__(
        self,
        image_metrics: Dict[str, Dict[str, float]],
        best_conf_per_class: Optional[Dict[str, float]] = None,
        curves: Optional[Dict[str, np.ndarray]] = None,
        *,
        metrics: Optional[Dict[str, Any]] = None,
        per_class_ap: Optional[Dict[int, float]] = None,
        nc: Optional[int] = None,
    ) -> None:
        self._init_map_metrics(metrics, per_class_ap, nc)
        self.image_metrics = image_metrics
        self.best_conf_per_class = dict(best_conf_per_class or {})
        curves = curves or {}
        self.px = np.asarray(curves.get("px", np.zeros(0)), dtype=np.float64)
        self.ap_class_index = np.asarray(
            curves.get("labels", np.zeros(0)), dtype=np.int64
        )
        self.p_curve = np.asarray(curves.get("p", _empty_curve()), dtype=np.float64)
        self.r_curve = np.asarray(curves.get("r", _empty_curve()), dtype=np.float64)
        self.f1_curve = np.asarray(curves.get("f1", _empty_curve()), dtype=np.float64)
        if self.f1_curve.size:
            best = int(self.f1_curve.mean(axis=0).argmax())
            self.p = self.p_curve[:, best].copy()
            self.r = self.r_curve[:, best].copy()
            self.f1 = self.f1_curve[:, best].copy()
        else:
            self.p = np.zeros(0, dtype=np.float64)
            self.r = np.zeros(0, dtype=np.float64)
            self.f1 = np.zeros(0, dtype=np.float64)

    @property
    def mp(self) -> float:
        """Mean over classes of ``p``: precision at the best mean-F1 confidence.

        Not ``metrics/precision``, which holds a COCO AP summary.
        """
        return float(self.p.mean()) if self.p.size else float("nan")

    @property
    def mr(self) -> float:
        """Mean over classes of ``r``: recall at the best mean-F1 confidence."""
        return float(self.r.mean()) if self.r.size else float("nan")

    def __repr__(self) -> str:
        return f"BoxImageMetrics({len(self.image_metrics)} images)"


class MaskMetrics(_MapMetrics):
    """``results.seg`` for segmentation: the mask ``(M)`` metrics by name."""

    _suffix = "(M)"

    def __init__(
        self,
        metrics: Dict[str, Any],
        per_class_ap: Optional[Dict[int, float]] = None,
        nc: Optional[int] = None,
    ) -> None:
        self._init_map_metrics(metrics, per_class_ap, nc)

    def __repr__(self) -> str:
        return f"MaskMetrics(map={self.map:.4f})"


class ValidationMetrics(dict):
    """The metrics dict ``val()`` returns, plus per-image and per-class results.

    It is a plain, flat ``dict`` of metric keys to finite numbers. Per-image
    and per-class results live on ``box`` instead. ``box.image_metrics`` maps
    each image filename to its ``precision``, ``recall``, ``f1``, ``tp``,
    ``fp`` and ``fn``, as in the ecosystem's validation results, so the images
    a model gets wrong are ``[k for k, m in r.box.image_metrics.items() if
    m["fp"] or m["fn"]]``. ``box.best_conf_per_class`` maps class names to
    F1-optimal confidence thresholds, and ``box.p_curve`` / ``box.r_curve`` /
    ``box.f1_curve`` hold each class's precision, recall and F1 across the
    confidences in ``box.px``. Detect and segment only; segmentation counts
    boxes.

    ``confusion_matrix`` is the run's :class:`ConfusionMatrix`
    (``matrix[predicted, true]``, ``summary()``, ``to_df()``), or None for
    tasks without one.
    """

    def __init__(
        self,
        metrics: Dict[str, Any],
        image_metrics: Optional[Dict[str, Dict[str, float]]] = None,
        best_conf_per_class: Optional[Dict[str, float]] = None,
        *,
        curves: Optional[Dict[str, np.ndarray]] = None,
        confusion_matrix: Optional["ConfusionMatrix"] = None,
        per_class_ap: Optional[Dict[int, float]] = None,
        mask_per_class_ap: Optional[Dict[int, float]] = None,
        nc: Optional[int] = None,
    ) -> None:
        super().__init__(metrics)
        if image_metrics is not None:
            self.box = BoxImageMetrics(
                image_metrics,
                best_conf_per_class,
                curves,
                metrics=self,
                per_class_ap=per_class_ap,
                nc=nc,
            )
        if any(key.endswith("(M)") for key in self):
            self.seg = MaskMetrics(self, mask_per_class_ap, nc)
        self.confusion_matrix = confusion_matrix


class ClassifyMetrics(ValidationMetrics):
    """Classification ``val()`` results: the metrics dict plus ecosystem names.

    ``top1`` and ``top5`` are the top-1 and top-5 accuracies (the
    ``metrics/accuracy_top1`` and ``metrics/accuracy_top5`` entries), and
    ``confusion_matrix`` counts top-1 predictions against labels.
    """

    @property
    def top1(self) -> float:
        return float(self.get("metrics/accuracy_top1", 0.0))

    @property
    def top5(self) -> float:
        return float(self.get("metrics/accuracy_top5", 0.0))


def _class_count(validator: Any) -> Optional[int]:
    nc = getattr(validator, "nc", None)
    try:
        return int(nc) if nc is not None else None
    except (TypeError, ValueError):
        return None


def with_image_metrics(metrics: Any, validator: Any) -> Any:
    """Attach a validator's per-image, per-class and confusion results.

    Detect and segment validators yield a :class:`ValidationMetrics` with
    ``box`` and ``confusion_matrix``; the classify validator a
    :class:`ClassifyMetrics`. Other validators' metrics pass through.
    """
    if not isinstance(metrics, dict):
        return metrics
    confusion_matrix = getattr(validator, "confusion_matrix", None)
    image_metrics = getattr(validator, "image_metrics", None)
    if image_metrics is not None:
        return ValidationMetrics(
            metrics,
            image_metrics,
            getattr(validator, "best_conf_per_class", None),
            curves=getattr(validator, "confidence_curves", None),
            confusion_matrix=confusion_matrix,
            per_class_ap=getattr(validator, "per_class_ap", None),
            mask_per_class_ap=getattr(validator, "mask_per_class_ap", None),
            nc=_class_count(validator),
        )
    if getattr(validator, "task", None) == "classify":
        return ClassifyMetrics(metrics, confusion_matrix=confusion_matrix)
    return metrics


class BaseValidator(ABC):
    """Abstract base class for model validators (Template Method pattern)."""

    task: str = "base"
    #: Whether this validator draws ``visualize=True`` images (#887). Others
    #: reject the flag instead of accepting and ignoring it.
    supports_visualize: bool = False
    #: Whether this validator applies ``agnostic_nms=True`` (#928). Others
    #: reject the flag instead of accepting and ignoring it.
    supports_agnostic_nms: bool = False

    def __init__(
        self,
        model: "BaseModel",
        config: Optional[ValidationConfig] = None,
        **kwargs,
    ) -> None:
        self.model = model
        self.config = config or ValidationConfig(**kwargs)
        if kwargs and config is not None:
            self.config = self.config.update(**kwargs)
        if getattr(self.config, "visualize", False) and not self.supports_visualize:
            raise ValueError(
                f"visualize=True is not supported by {type(self).__name__}; "
                f"it covers {', '.join(VISUALIZE_TASKS)}"
            )

        if (
            getattr(self.config, "agnostic_nms", False)
            and not self.supports_agnostic_nms
        ):
            raise ValueError(
                f"agnostic_nms=True is not supported by {type(self).__name__}; "
                "it covers detect and segment validation"
            )

        self.device = self._setup_device()
        self.dataloader: Optional[DataLoader] = None
        self.seen = 0
        self.speed = {
            "preprocess": 0.0,
            "inference": 0.0,
            "postprocess": 0.0,
            "total": 0.0,
        }
        self.save_dir: Optional[Path] = None

    # =========================================================================
    # Concrete methods
    # =========================================================================

    def __call__(self, **kwargs) -> Dict[str, float]:
        """Run validation and return metrics."""
        return self.run(**kwargs)

    def run(self, **kwargs) -> Dict[str, float]:
        """Main validation entry point (template method)."""
        # Local import: models.base imports validation.preprocessors, so a
        # module-level import here would be circular.
        from libreyolo.models.base.cuda_graph import normalize_cuda_graph_mode

        from libreyolo.utils.input_channels import (
            dataset_input_channels,
            model_channels,
        )

        with dataset_input_channels(model_channels(self.model)):
            self._setup(**kwargs)
        requested = getattr(self.config, "cuda_graph", False)
        # The loss scope wraps graph capture too, so a captured graph carries
        # whatever extra outputs the family criterion needs.
        with self._validation_loss_scope():
            if normalize_cuda_graph_mode(requested) is None:
                self._run_validation()
            else:
                # Same contract as predict(..., cuda_graph=...): unsupported
                # families fail loudly up front. Uncapturable situations (CPU
                # device, capture failure) run the whole pass eagerly.
                self.model._require_cuda_graph_support()
                if self._capture_val_graph():
                    runner = self.model._get_graph_runner()
                    previous = runner.capture_on_miss
                    # Replay-only inside the loop. The loader's pin-memory
                    # threads call cudaHostAlloc while batches are in flight,
                    # and any synchronous CUDA allocation invalidates a capture
                    # in progress, so capture happens only at the controlled
                    # point in _capture_val_graph. A shape miss inside the loop
                    # (the final partial batch) runs eager.
                    runner.capture_on_miss = False
                    try:
                        with self.model.cuda_graph_scope(requested):
                            self._run_validation()
                    finally:
                        runner.capture_on_miss = previous
                else:
                    self._run_validation()
        return self._finalize()

    def _validation_loss_scope(self):
        """Enter the active loss adapter's forward scope, if it has one.

        Families whose eval forward already carries everything the criterion
        needs (YOLO9, RF-DETR) do not define one, so this is a no-op for them.
        """
        adapter = getattr(self, "_active_loss_adapter", None)
        scope = getattr(adapter, "forward_scope", None)
        return nullcontext() if scope is None else scope()

    def _setup_device(self) -> torch.device:
        if self.config.device == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        else:
            device_str = str(self.config.device)
            if device_str.isdigit():
                device_str = f"cuda:{device_str}"
            device = torch.device(device_str)
        return device

    def _setup(self, **kwargs) -> None:
        if self.config.save_dir:
            self.save_dir = Path(self.config.save_dir)
        else:
            model_tag = f"{self.model._get_model_name()}_{self.model.size}"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            self.save_dir = Path("runs/val") / f"{model_tag}_{timestamp}"

        self.save_dir.mkdir(parents=True, exist_ok=True)

        # The dataset, the dataloader (worker spawn, pinned-buffer allocation)
        # and the model warmup are per-instance costs, not per-run costs:
        # nothing they depend on can change between two runs of one validator.
        # Reusing them is what makes calling a single validator every training
        # epoch cheap — a measured 4.6 s of cudaHostAlloc alone per rebuild.
        # A fresh instance still builds everything, so standalone .val() calls
        # behave exactly as before.
        if self.dataloader is None:
            self.dataloader = self._setup_dataloader()
            warmup_needed = True
        else:
            warmup_needed = False

        self.seen = 0
        self.speed = {
            "preprocess": 0.0,
            "inference": 0.0,
            "postprocess": 0.0,
            "total": 0.0,
        }

        self._init_metrics()
        if warmup_needed:
            self._warmup_model()

        if self.config.verbose:
            logger.info("Validating on %d images...", len(self.dataloader.dataset))
            logger.info("Device: %s", self.device)
            logger.info("Batch size: %d", self.config.batch_size)

    def _run_validation(self) -> None:
        self.model.model.eval()

        if self.config.augment:
            self._run_validation_augmented()
            return

        pbar = tqdm(
            self.dataloader,
            desc="Validating",
            total=len(self.dataloader),
            disable=not self.config.verbose or not sys.stderr.isatty(),
            file=sys.stderr,
        )

        total_start = time.time()

        with torch.no_grad():
            for batch_idx, batch in enumerate(pbar):
                t1 = time.time()
                images, targets, img_info, img_ids = self._preprocess_batch(batch)
                self.speed["preprocess"] += time.time() - t1

                t2 = time.time()
                preds = self._inference(images)
                self.speed["inference"] += time.time() - t2

                # Optional family-specific metrics (for example validation
                # loss) consume the raw forward output before postprocessing
                # mutates or discards model-specific tensors.
                self._update_batch_metrics(preds, images, targets)

                t3 = time.time()
                detections = self._postprocess_predictions(preds, batch)
                self.speed["postprocess"] += time.time() - t3

                self._update_metrics(detections, targets, img_info, img_ids)
                self.seen += len(images)

        self.speed["total"] = time.time() - total_start

    def _run_validation_augmented(self) -> None:
        """TTA validation — subclasses override to call model._predict_augment per image."""
        raise NotImplementedError

    def _finalize(self) -> Dict[str, float]:
        metrics = self._compute_metrics()

        if self.config.verbose:
            self._print_results(metrics)

        self.config.to_yaml(self.save_dir / "config.yaml")

        if getattr(self.config, "visualize", False):
            logger.info("visualize images saved to %s", self.save_dir / "visualize")

        if self.config.save_plots:
            try:
                self._save_plots(metrics)
            except Exception as exc:
                logger.warning("Failed to save validation plots: %s", exc)

        # Include timing info so callers (e.g. benchmarks) can use it
        if self.seen > 0:
            ms_per_image = self.speed["total"] / self.seen * 1000
            metrics["speed/preprocess_ms"] = self.speed["preprocess"] / self.seen * 1000
            metrics["speed/inference_ms"] = self.speed["inference"] / self.seen * 1000
            metrics["speed/postprocess_ms"] = (
                self.speed["postprocess"] / self.seen * 1000
            )
            metrics["speed/total_ms"] = ms_per_image
            metrics["speed/total_s"] = self.speed["total"]
            metrics["speed/images_seen"] = self.seen

        return metrics

    def _inference(self, images: torch.Tensor) -> Any:
        """Run model inference on a batch of images (B, C, H, W)."""
        from libreyolo.models.base.cuda_graph import forward_maybe_graphed

        use_non_blocking = self.device.type == "cuda"
        images = images.to(self.device, non_blocking=use_non_blocking)
        with self._autocast_context():
            # Replays a captured CUDA graph when run() opened a graph scope;
            # exactly model._forward(images) otherwise, including for the
            # duck-typed model doubles in the test suite.
            preds = forward_maybe_graphed(self.model, images)
        return _upcast_bfloat16(preds)

    def _autocast_context(self):
        if self.config.half and self.device.type == "cuda":
            return torch.amp.autocast(
                "cuda", dtype=torch_amp_dtype(self.config.amp_dtype)
            )
        return nullcontext()

    def _capture_val_graph(self) -> bool:
        """Capture the full-batch forward before the loop; True on success.

        Capture must not happen inside the batch loop: the DataLoader's
        pin-memory threads call cudaHostAlloc while batches are in flight, and
        a synchronous CUDA allocation invalidates a capture in progress
        (cudaErrorStreamCaptureInvalidated) -- observed to then poison every
        later capture attempt in the process, while the epoch whose capture
        failed lost its validation entirely. Here, before iteration starts,
        the loader threads are quiet and warmup just synchronized the device.
        No-op when the shape is already captured, so the validator the trainer
        reuses across epochs pays this once per run.
        """
        from libreyolo.models.base.cuda_graph import CudaGraphUnavailable

        imgsz = self.config.imgsz
        if isinstance(imgsz, (list, tuple)):
            imgsz_h, imgsz_w = int(imgsz[0]), int(imgsz[1])
        else:
            imgsz_h = imgsz_w = int(imgsz)
        dummy = torch.zeros(
            (self.config.batch_size, 3, imgsz_h, imgsz_w),
            dtype=torch.float32,
            device=self.device,
        )
        model_module = getattr(self.model, "model", None)
        if hasattr(model_module, "eval"):
            model_module.eval()
        try:
            with torch.no_grad(), self._autocast_context():
                self.model._get_graph_runner().capture(dummy)
        except CudaGraphUnavailable as exc:
            if not getattr(self, "_warned_eager_val", False):
                self._warned_eager_val = True
                logger.warning("cuda_graph: validation runs eager (%s)", exc)
            return False
        return True

    def _warmup_model(self, n_warmup: int = 3) -> None:
        """Run dummy inference passes to trigger JIT compilation and CUDA kernel caching."""
        if self.config.verbose:
            logger.info("Warming up model (%d iterations)...", n_warmup)

        imgsz = self.config.imgsz
        batch_size = min(self.config.batch_size, 4)

        if isinstance(imgsz, (list, tuple)):
            imgsz_h, imgsz_w = int(imgsz[0]), int(imgsz[1])
        else:
            imgsz_h = imgsz_w = int(imgsz)

        dummy_input = torch.zeros(
            (batch_size, 3, imgsz_h, imgsz_w),
            dtype=torch.float32,
            device=self.device,
        )

        # Prevent NoneType errors during warmup forward pass
        if hasattr(self.model, "_original_size"):
            self.model._original_size = (imgsz_h, imgsz_w)

        self.model.model.eval()
        with torch.no_grad():
            for _ in range(n_warmup):
                try:
                    with self._autocast_context():
                        _ = self.model._forward(dummy_input)
                except Exception as e:
                    if self.config.verbose:
                        logger.warning("Warmup failed (non-fatal): %s", e)
                    break

        if hasattr(self.model, "_original_size"):
            self.model._original_size = None

        if self.device.type == "cuda":
            torch.cuda.synchronize()

    def _save_plots(self, metrics: Dict[str, float]) -> None:
        """Override in subclasses to save validation plots."""

    def _update_batch_metrics(
        self,
        predictions: Any,
        images: torch.Tensor,
        targets: Any,
    ) -> None:
        """Update optional metrics that consume raw per-batch predictions."""

    def _print_results(self, metrics: Dict[str, float]) -> None:
        logger.info("=" * 50)
        logger.info("Validation Results")
        logger.info("=" * 50)

        for key, value in metrics.items():
            logger.info("  %s: %.4f", key, value)

        logger.info("-" * 50)
        logger.info("  Images processed: %d", self.seen)
        logger.info("  Total time: %.2fs", self.speed["total"])
        if self.seen > 0:
            logger.info("  Speed: %.1fms/image", self.speed["total"] / self.seen * 1000)
        logger.info("=" * 50)

    # =========================================================================
    # Abstract methods
    # =========================================================================

    @abstractmethod
    def _setup_dataloader(self) -> DataLoader:
        """Create validation dataloader from config."""
        pass

    @abstractmethod
    def _init_metrics(self) -> None:
        """Initialize metrics containers."""
        pass

    @abstractmethod
    def _preprocess_batch(self, batch: Any) -> tuple:
        """Preprocess a batch → (images, targets, img_info, img_ids)."""
        pass

    @abstractmethod
    def _postprocess_predictions(self, preds: Any, batch: Any) -> Any:
        """Postprocess model predictions into detection format."""
        pass

    @abstractmethod
    def _update_metrics(
        self, preds: Any, targets: Any, img_info: Any, img_ids: Any = None
    ) -> None:
        """Update metrics with batch predictions."""
        pass

    @abstractmethod
    def _compute_metrics(self) -> Dict[str, float]:
        """Compute final metrics from accumulated stats."""
        pass
