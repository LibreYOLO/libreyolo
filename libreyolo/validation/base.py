"""Base validator class for LibreYOLO."""

import logging
import sys
import time
from abc import ABC, abstractmethod
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from libreyolo.utils.amp import torch_amp_dtype

from .config import VISUALIZE_TASKS, ValidationConfig

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from libreyolo.models.base import BaseModel


class BaseValidator(ABC):
    """Abstract base class for model validators (Template Method pattern)."""

    task: str = "base"
    #: Whether this validator draws ``visualize=True`` images (#887). Others
    #: reject the flag instead of accepting and ignoring it.
    supports_visualize: bool = False

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
            return forward_maybe_graphed(self.model, images)

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
