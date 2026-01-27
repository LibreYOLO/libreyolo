"""
LIBREYOLOX implementation for LibreYOLO.

Supports both inference and training.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from PIL import Image

from ..common.base_model import LibreYOLOBase
from ..common.image_loader import ImageInput
from .nn import YOLOXModel
from .utils import preprocess_image as _yolox_preprocess, postprocess


class LIBREYOLOX(LibreYOLOBase):
    """
    YOLOX model for object detection.

    Args:
        model_path: Model weights source. Can be:
            - str: Path to a .pt/.pth weights file
            - dict: Pre-loaded state_dict (e.g., from torch.load())
            - None: Random initialization for training from scratch
        size: Model size variant. Must be one of: "nano", "tiny", "s", "m", "l", "x"
        nb_classes: Number of classes (default: 80 for COCO)
        device: Device for inference.

    Examples:
        Load weights for inference (prefer LIBREYOLO factory for auto-detection)::

            >>> from libreyolo import LIBREYOLO
            >>> model = LIBREYOLO("libreyoloXs.pt")
            >>> detections = model(image="image.jpg", save=True)

        Create a fresh model for training (nb_classes read from YAML)::

            >>> model = LIBREYOLOX(size="s")
            >>> results = model.train(data="coco128.yaml", epochs=100)
    """

    # Default input sizes for different model variants
    DEFAULT_INPUT_SIZES = {
        "nano": 416,
        "tiny": 416,
        "s": 640,
        "m": 640,
        "l": 640,
        "x": 640,
    }

    def __init__(
        self,
        model_path=None,
        size: str = "s",
        nb_classes: int = 80,
        device: str = "auto",
        **kwargs,
    ):
        self.input_size = self.DEFAULT_INPUT_SIZES.get(size, 640)
        super().__init__(
            model_path=model_path,
            size=size,
            nb_classes=nb_classes,
            device=device,
            **kwargs,
        )

        # Apply nano-specific BatchNorm settings (matching official YOLOX).
        # Must run after super().__init__() which loads weights.
        if self.size == 'nano':
            for m in self.model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

    def _get_valid_sizes(self) -> List[str]:
        return ["nano", "tiny", "s", "m", "l", "x"]

    def _get_model_name(self) -> str:
        return "LIBREYOLOX"

    def _get_input_size(self) -> int:
        return self.input_size

    def _init_model(self) -> nn.Module:
        return YOLOXModel(config=self.size, nb_classes=self.nb_classes)

    def _get_available_layers(self) -> Dict[str, nn.Module]:
        return {
            "backbone_stem": self.model.backbone.stem,
            "backbone_dark2": self.model.backbone.dark2,
            "backbone_dark3": self.model.backbone.dark3,
            "backbone_dark4": self.model.backbone.dark4,
            "backbone_dark5": self.model.backbone.dark5,
        }

    def _get_val_preprocessor(self, img_size: int = None):
        """YOLOX uses letterbox + no normalization (0-255 range)."""
        from libreyolo.validation.preprocessors import YOLOXValPreprocessor
        if img_size is None:
            img_size = self.input_size
        return YOLOXValPreprocessor(img_size=(img_size, img_size))

    def _load_weights(self, model_path: str):
        """Override to handle different checkpoint formats."""
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model weights file not found: {model_path}")

        try:
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

            if isinstance(checkpoint, dict):
                if "model" in checkpoint:
                    state_dict = checkpoint["model"]
                elif "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint

            self.model.load_state_dict(state_dict, strict=False)

        except Exception as e:
            raise RuntimeError(
                f"Failed to load model weights from {model_path}: {e}"
            ) from e

    def _preprocess(
        self, image: ImageInput, color_format: str = "auto", input_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Image.Image, Tuple[int, int]]:
        """YOLOX preprocessing with letterbox - stores ratio for postprocessing."""
        effective_size = input_size if input_size is not None else self.input_size
        tensor, orig_img, orig_size, ratio = _yolox_preprocess(
            image, input_size=effective_size, color_format=color_format
        )
        # Store ratio for use in _postprocess
        self._current_ratio = ratio
        return tensor, orig_img, orig_size

    def _forward(self, input_tensor: torch.Tensor) -> Any:
        return self.model(input_tensor)

    def _postprocess(
        self,
        output: Any,
        conf_thres: float,
        iou_thres: float,
        original_size: Tuple[int, int],
        max_det: int = 300,
        **kwargs,
    ) -> Dict:
        # Compute ratio from original_size if not set during _preprocess
        # This handles batch validation where _preprocess is not called per-image
        ratio = getattr(self, "_current_ratio", None)

        # Use passed input_size if available (from validator), otherwise use model's default
        # This is important when validation uses a different size than model's native size
        actual_input_size = kwargs.get('input_size', self.input_size)

        if ratio is None and original_size is not None:
            orig_w, orig_h = original_size
            # Use actual input size for ratio calculation
            ratio = min(actual_input_size / orig_h, actual_input_size / orig_w)
        elif ratio is None:
            ratio = 1.0
        return postprocess(
            output,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            input_size=actual_input_size,
            original_size=original_size,
            ratio=ratio,
            max_det=max_det,
        )

    def export(
        self,
        format: str = "onnx",
        output_path: Optional[str] = None,
        opset: int = 11,
        simplify: bool = True,
        dynamic: bool = False,
    ) -> str:
        """
        Export the model to a different format.

        Args:
            format: Export format ("onnx", "torchscript")
            output_path: Output file path (auto-generated if None)
            opset: ONNX opset version (default: 11)
            simplify: Simplify ONNX model (default: True)
            dynamic: Use dynamic input shapes (default: False)

        Returns:
            Path to the exported model file
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            output_path = f"yolox_{self.size}_{timestamp}.{format}"

        self.model.eval()

        if format == "onnx":
            return self._export_onnx(output_path, opset, simplify, dynamic)
        elif format == "torchscript":
            return self._export_torchscript(output_path)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def _export_onnx(
        self,
        output_path: str,
        opset: int = 11,
        simplify: bool = True,
        dynamic: bool = False,
    ) -> str:
        """Export to ONNX format."""
        dummy_input = torch.randn(1, 3, self.input_size, self.input_size).to(
            self.device
        )

        dynamic_axes = None
        if dynamic:
            dynamic_axes = {
                "images": {0: "batch", 2: "height", 3: "width"},
                "outputs": {0: "batch"},
            }

        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            input_names=["images"],
            output_names=["outputs"],
            opset_version=opset,
            dynamic_axes=dynamic_axes,
        )

        if simplify:
            try:
                import onnx
                from onnxsim import simplify as onnx_simplify

                model = onnx.load(output_path)
                model_simplified, check = onnx_simplify(model)
                if check:
                    onnx.save(model_simplified, output_path)
            except ImportError:
                pass

        return output_path

    def _export_torchscript(self, output_path: str) -> str:
        """Export to TorchScript format."""
        dummy_input = torch.randn(1, 3, self.input_size, self.input_size).to(
            self.device
        )
        traced_model = torch.jit.trace(self.model, dummy_input)
        traced_model.save(output_path)
        return output_path

    def train(
        self,
        data: str,
        *,
        # Training parameters
        epochs: int = 100,
        batch: int = 16,
        imgsz: int = 640,

        # Optimizer parameters
        lr0: float = 0.01,
        optimizer: str = "SGD",

        # System parameters
        device: str = "",
        workers: int = 8,
        seed: int = 0,

        # Output parameters
        project: str = "runs/train",
        name: str = "exp",
        exist_ok: bool = False,

        # Model parameters
        pretrained: bool = True,

        # Training features
        resume: bool = False,
        amp: bool = True,
        patience: int = 50,

        **kwargs
    ) -> dict:
        """
        Train the YOLOX model on a dataset.

        Args:
            data: Path to data.yaml file (required)

            epochs: Number of epochs to train
            batch: Batch size (total, will be divided across devices if multi-GPU)
            imgsz: Input image size (square: imgsz x imgsz)

            lr0: Initial learning rate
            optimizer: Optimizer name ('SGD', 'Adam', 'AdamW')

            device: Device to train on ('', 'cpu', 'cuda', '0', '0,1,2,3')
                    Empty string '' = auto-detect
            workers: Number of dataloader worker processes
            seed: Random seed for reproducibility

            project: Root directory for training runs
            name: Experiment name (auto-increments: exp, exp2, exp3...)
            exist_ok: If True, overwrite existing experiment directory

            pretrained: Use pretrained weights if available (not implemented yet)

            resume: If True, resume training from the loaded checkpoint.
                    Load the checkpoint first, then call train(resume=True).
            amp: Enable automatic mixed precision training
            patience: Early stopping patience (epochs without improvement)

            **kwargs: Additional training parameters

        Returns:
            dict: Training results containing:
                - 'final_loss': Final training loss
                - 'best_mAP50': Best mAP@0.5 achieved
                - 'best_mAP50_95': Best mAP@0.5:0.95 achieved
                - 'best_epoch': Epoch with best validation performance
                - 'save_dir': Path to training output directory
                - 'best_checkpoint': Path to best model checkpoint
                - 'last_checkpoint': Path to last model checkpoint

        Raises:
            FileNotFoundError: If data.yaml not found
            ValueError: If invalid parameters provided
            RuntimeError: If training fails (OOM, NaN loss, etc.)

        Example:
            >>> from libreyolo import LIBREYOLOX
            >>> model = LIBREYOLOX(size="s")
            >>> results = model.train(
            ...     data="coco128.yaml",
            ...     epochs=100,
            ...     batch=16,
            ...     imgsz=640,
            ...     device="0"
            ... )
            >>> print(f"Best mAP: {results['best_mAP50_95']:.3f}")
        """
        from libreyolo.training import YOLOXTrainer, YOLOXTrainConfig
        from libreyolo.data import load_data_config

        # Load and validate data config (handles built-in datasets and auto-download)
        try:
            data_config = load_data_config(data, autodownload=True)
            # Get the resolved yaml path from config
            data = data_config.get('yaml_file', data)
        except Exception as e:
            raise FileNotFoundError(f"Failed to load dataset config '{data}': {e}")

        # Reconcile nb_classes with dataset
        yaml_nc = data_config.get('nc')
        if yaml_nc is not None and yaml_nc != self.nb_classes:
            self._rebuild_for_new_classes(yaml_nc)

        # Create training config
        config = YOLOXTrainConfig(
            size=self.size,
            num_classes=self.nb_classes,
            data=data,
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            lr0=lr0,
            optimizer=optimizer.lower(),
            device=device if device else "auto",
            workers=workers,
            seed=seed,
            project=project,
            name=name,
            exist_ok=exist_ok,
            resume=resume,
            amp=amp,
            patience=patience,
            **kwargs
        )

        # Set random seed for reproducibility
        if seed > 0:
            import random
            import numpy as np
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        # Create trainer (pass wrapper model for validation)
        trainer = YOLOXTrainer(model=self.model, config=config, wrapper_model=self)

        # Resume if requested — uses the model_path that was loaded into this instance
        if resume:
            if not self.model_path:
                raise ValueError(
                    "resume=True requires a checkpoint. Load one first: "
                    "model = LIBREYOLOX('path/to/last.pt'); model.train(data=..., resume=True)"
                )
            trainer.resume(str(self.model_path))

        # Run training
        results = trainer.train()

        # Load best model weights into current instance
        if Path(results['best_checkpoint']).exists():
            self._load_weights(results['best_checkpoint'])

        return results
