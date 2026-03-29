"""Base validator class for LibreYOLO."""

import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import ValidationConfig

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from libreyolo.models.base import BaseModel


class BaseValidator(ABC):
    """Abstract base class for model validators (Template Method pattern)."""

    task: str = "base"

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
        self._setup(**kwargs)
        self._run_validation()
        return self._finalize()

    def _setup_device(self) -> torch.device:
        if self.config.device == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        else:
            device = torch.device(self.config.device)
        return device

    def _setup(self, **kwargs) -> None:
        if self.config.save_dir:
            self.save_dir = Path(self.config.save_dir)
        else:
            model_tag = f"{self.model._get_model_name()}_{self.model.size}"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            self.save_dir = Path("runs/val") / f"{model_tag}_{timestamp}"

        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.dataloader = self._setup_dataloader()

        self.seen = 0
        self.speed = {
            "preprocess": 0.0,
            "inference": 0.0,
            "postprocess": 0.0,
            "total": 0.0,
        }

        self._init_metrics()
        self._warmup_model()

        if self.config.verbose:
            logger.info("Validating on %d images...", len(self.dataloader.dataset))
            logger.info("Device: %s", self.device)
            logger.info("Batch size: %d", self.config.batch_size)

    def _run_validation(self) -> None:
        self.model.model.eval()

        pbar = tqdm(
            self.dataloader,
            desc="Validating",
            total=len(self.dataloader),
            disable=not self.config.verbose,
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

                t3 = time.time()
                detections = self._postprocess_predictions(preds, batch)
                self.speed["postprocess"] += time.time() - t3

                self._update_metrics(detections, targets, img_info, img_ids)
                self.seen += len(images)

        self.speed["total"] = time.time() - total_start

    def _finalize(self) -> Dict[str, float]:
        metrics = self._compute_metrics()

        if self.config.verbose:
            self._print_results(metrics)

        self.config.to_yaml(self.save_dir / "config.yaml")

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
        use_non_blocking = self.device.type == "cuda"
        return self.model._forward(
            images.to(self.device, non_blocking=use_non_blocking)
        )

    def _warmup_model(self, n_warmup: int = 3) -> None:
        """Run dummy inference passes to trigger JIT compilation and CUDA kernel caching."""
        if self.config.verbose:
            logger.info("Warming up model (%d iterations)...", n_warmup)

        imgsz = self.config.imgsz
        batch_size = min(self.config.batch_size, 4)

        dummy_input = torch.zeros(
            (batch_size, 3, imgsz, imgsz),
            dtype=torch.float32,
            device=self.device,
        )

        # Prevent NoneType errors during warmup forward pass
        if hasattr(self.model, "_original_size"):
            self.model._original_size = (imgsz, imgsz)

        self.model.model.eval()
        with torch.no_grad():
            for _ in range(n_warmup):
                try:
                    _ = self.model._forward(dummy_input)
                except Exception as e:
                    if self.config.verbose:
                        logger.warning("Warmup failed (non-fatal): %s", e)
                    break

        if hasattr(self.model, "_original_size"):
            self.model._original_size = None

        if self.device.type == "cuda":
            torch.cuda.synchronize()

    def _print_results(self, metrics: Dict[str, float]) -> None:
        logger.info("\n" + "=" * 50)
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
