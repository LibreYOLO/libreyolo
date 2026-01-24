"""
Base validator class for LibreYOLO.

Provides an abstract base class implementing the Template Method pattern
for model validation.
"""

import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import ValidationConfig

if TYPE_CHECKING:
    from libreyolo.common.base_model import LibreYOLOBase


class BaseValidator(ABC):
    """
    Abstract base class for model validators.

    Implements the Template Method pattern where subclasses provide
    task-specific implementations for preprocessing, inference, and postprocessing.

    Attributes:
        model: LibreYOLO model instance.
        config: Validation configuration.
        device: Torch device for inference.
        dataloader: Validation data loader.
        seen: Number of images processed.
        speed: Timing information per phase.
    """

    task: str = "base"

    def __init__(
        self,
        model: "LibreYOLOBase",
        config: Optional[ValidationConfig] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the validator.

        Args:
            model: LibreYOLO model instance.
            config: ValidationConfig instance.
            **kwargs: Override config parameters.
        """
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

        # Save directory
        self.save_dir: Optional[Path] = None

    def __call__(self, **kwargs) -> Dict[str, float]:
        """Run validation and return metrics."""
        return self.run(**kwargs)

    def run(self, **kwargs) -> Dict[str, float]:
        """
        Main validation loop following template method pattern.

        Args:
            **kwargs: Additional arguments.

        Returns:
            Dictionary with validation metrics.
        """
        # Setup
        self._setup(**kwargs)

        # Validation loop
        self._run_validation()

        # Finalize and return metrics
        return self._finalize()

    def _setup_device(self) -> torch.device:
        """Setup and return the device for validation."""
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
        """Setup dataloader, metrics, and save directory."""
        # Setup save directory
        if self.config.save_dir:
            self.save_dir = Path(self.config.save_dir)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.save_dir = Path("runs/val") / f"exp_{timestamp}"

        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Setup dataloader
        self.dataloader = self._setup_dataloader()

        # Reset counters
        self.seen = 0
        self.speed = {
            "preprocess": 0.0,
            "inference": 0.0,
            "postprocess": 0.0,
            "total": 0.0,
        }

        # Initialize metrics (implemented by subclass)
        self._init_metrics()

        # Warmup model (JIT compilation, CUDA kernel caching)
        self._warmup_model()

        if self.config.verbose:
            print(f"Validating on {len(self.dataloader.dataset)} images...")
            print(f"Device: {self.device}")
            print(f"Batch size: {self.config.batch_size}")

    def _run_validation(self) -> None:
        """Run the validation loop over all batches."""
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
                # Preprocess
                t1 = time.time()
                images, targets, img_info, img_ids = self._preprocess_batch(batch)
                self.speed["preprocess"] += time.time() - t1

                # Inference
                t2 = time.time()
                preds = self._inference(images)
                self.speed["inference"] += time.time() - t2

                # Postprocess
                t3 = time.time()
                detections = self._postprocess_predictions(preds, batch)
                self.speed["postprocess"] += time.time() - t3

                # Update metrics
                self._update_metrics(detections, targets, img_info)

                # Update seen count
                self.seen += len(images)

        self.speed["total"] = time.time() - total_start

    def _finalize(self) -> Dict[str, float]:
        """Finalize validation and compute final metrics."""
        # Compute metrics (implemented by subclass)
        metrics = self._compute_metrics()

        # Print results
        if self.config.verbose:
            self._print_results(metrics)

        # Generate plots
        if self.config.plots:
            self._generate_plots()

        # Save config
        self.config.to_yaml(self.save_dir / "config.yaml")

        return metrics

    def _inference(self, images: torch.Tensor) -> Any:
        """
        Run model inference on batch of images.

        Args:
            images: (B, C, H, W) batch of images.

        Returns:
            Model predictions.
        """
        # Use non_blocking for CUDA to overlap transfer with computation
        use_non_blocking = self.device.type == "cuda"
        return self.model._forward(images.to(self.device, non_blocking=use_non_blocking))

    def _warmup_model(self, n_warmup: int = 3) -> None:
        """
        Warmup model with dummy inference passes.

        This triggers JIT compilation, CUDA kernel caching, and memory allocation
        before the timed validation loop begins.

        Args:
            n_warmup: Number of warmup iterations.
        """
        if self.config.verbose:
            print(f"Warming up model ({n_warmup} iterations)...")

        # Get input size from model or use config
        imgsz = self.config.imgsz
        batch_size = min(self.config.batch_size, 4)  # Small batch for warmup

        # Create dummy input
        dummy_input = torch.zeros(
            (batch_size, 3, imgsz, imgsz),
            dtype=torch.float32,
            device=self.device,
        )

        # Set dummy original_size for models that need it (e.g., RT-DETR)
        # This prevents NoneType errors during warmup forward pass
        if hasattr(self.model, '_original_size'):
            self.model._original_size = (imgsz, imgsz)

        # Run warmup passes
        self.model.model.eval()
        with torch.no_grad():
            for _ in range(n_warmup):
                try:
                    _ = self.model._forward(dummy_input)
                except Exception as e:
                    if self.config.verbose:
                        print(f"Warmup failed (non-fatal): {e}")
                    break

        # Reset original_size after warmup
        if hasattr(self.model, '_original_size'):
            self.model._original_size = None

        # Sync CUDA if available
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    def _print_results(self, metrics: Dict[str, float]) -> None:
        """Print validation results."""
        print("\n" + "=" * 50)
        print("Validation Results")
        print("=" * 50)

        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")

        print("-" * 50)
        print(f"  Images processed: {self.seen}")
        print(f"  Total time: {self.speed['total']:.2f}s")
        if self.seen > 0:
            print(f"  Speed: {self.speed['total'] / self.seen * 1000:.1f}ms/image")
        print("=" * 50)

    # =========================================================================
    # ABSTRACT METHODS - Must be implemented by subclasses
    # =========================================================================

    @abstractmethod
    def _setup_dataloader(self) -> DataLoader:
        """
        Create validation dataloader from config.

        Returns:
            DataLoader for validation data.
        """
        pass

    @abstractmethod
    def _init_metrics(self) -> None:
        """Initialize metrics containers."""
        pass

    @abstractmethod
    def _preprocess_batch(self, batch: Any) -> tuple:
        """
        Preprocess a batch of data.

        Args:
            batch: Raw batch from dataloader.

        Returns:
            Tuple of (images, targets, img_info, img_ids).
        """
        pass

    @abstractmethod
    def _postprocess_predictions(self, preds: Any, batch: Any) -> Any:
        """
        Postprocess model predictions.

        Args:
            preds: Raw model output.
            batch: Original batch data.

        Returns:
            Processed predictions.
        """
        pass

    @abstractmethod
    def _update_metrics(self, preds: Any, targets: Any, img_info: Any) -> None:
        """
        Update metrics with batch predictions.

        Args:
            preds: Processed predictions.
            targets: Ground truth targets.
            img_info: Image information (shapes, etc.).
        """
        pass

    @abstractmethod
    def _compute_metrics(self) -> Dict[str, float]:
        """
        Compute final metrics from accumulated stats.

        Returns:
            Dictionary with metric names and values.
        """
        pass

    @abstractmethod
    def _generate_plots(self) -> None:
        """Generate and save visualization plots."""
        pass
