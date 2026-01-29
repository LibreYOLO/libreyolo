"""
LibreYOLO9 inference and training wrapper.

Provides a high-level API for YOLOv9 object detection.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from PIL import Image

from ..common.base_model import LibreYOLOBase
from ..common.image_loader import ImageInput
from ..common.utils import preprocess_image
from .nn import LibreYOLO9Model, DDetect, V9_CONFIGS
from .utils import postprocess


class LIBREYOLO9(LibreYOLOBase):
    """
    LibreYOLO9 model for object detection.

    Args:
        model_path: Model weights source. Can be:
            - str: Path to a .pt/.pth weights file
            - dict: Pre-loaded state_dict (e.g., from torch.load())
        size: Model size variant (required). Must be one of: "t", "s", "m", "c"
        reg_max: Regression max value for DFL (default: 16)
        nb_classes: Number of classes (default: 80 for COCO)
        device: Device for inference. "auto" uses CUDA if available, else MPS, else CPU.

    Example:
        >>> model = LIBREYOLO9(model_path="path/to/weights.pt", size="s")
        >>> detections = model(image=image_path, save=True)
        >>> # Use tiling for large images
        >>> detections = model(image=large_image_path, save=True, tiling=True)
    """

    def __init__(
        self,
        model_path,
        size: str,
        reg_max: int = 16,
        nb_classes: int = 80,
        device: str = "auto",
        **kwargs,
    ):
        self.reg_max = reg_max
        super().__init__(
            model_path=model_path,
            size=size,
            nb_classes=nb_classes,
            device=device,
            **kwargs,
        )

    def _get_valid_sizes(self) -> List[str]:
        return ["t", "s", "m", "c"]

    def _get_model_name(self) -> str:
        return "LIBREYOLO9"

    def _get_input_size(self) -> int:
        return 640

    def _init_model(self) -> nn.Module:
        return LibreYOLO9Model(
            config=self.size, reg_max=self.reg_max, nb_classes=self.nb_classes
        )

    def _get_available_layers(self) -> Dict[str, nn.Module]:
        return {
            # Backbone layers
            "backbone_conv0": self.model.backbone.conv0,
            "backbone_conv1": self.model.backbone.conv1,
            "backbone_elan1": self.model.backbone.elan1,
            "backbone_down2": self.model.backbone.down2,
            "backbone_elan2": self.model.backbone.elan2,
            "backbone_down3": self.model.backbone.down3,
            "backbone_elan3": self.model.backbone.elan3,
            "backbone_down4": self.model.backbone.down4,
            "backbone_elan4": self.model.backbone.elan4,
            "backbone_spp": self.model.backbone.spp,
            # Neck layers
            "neck_elan_up1": self.model.neck.elan_up1,
            "neck_elan_up2": self.model.neck.elan_up2,
            "neck_elan_down1": self.model.neck.elan_down1,
            "neck_elan_down2": self.model.neck.elan_down2,
        }

    def _preprocess(
        self, image: ImageInput, color_format: str = "auto", input_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Image.Image, Tuple[int, int]]:
        effective_size = input_size if input_size is not None else 640
        return preprocess_image(image, input_size=effective_size, color_format=color_format)

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
        actual_input_size = kwargs.get('input_size', 640)
        return postprocess(
            output,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            input_size=actual_input_size,
            original_size=original_size,
            max_det=max_det,
        )

    def _strict_loading(self) -> bool:
        """Use non-strict loading for YOLOv9 to handle profiling artifacts in weights."""
        return False

    def _get_val_preprocessor(self, img_size: int = None):
        """YOLOv9 uses letterbox + normalization (0-1 range)."""
        from libreyolo.validation.preprocessors import V9ValPreprocessor
        if img_size is None:
            img_size = 640
        return V9ValPreprocessor(img_size=(img_size, img_size))

    def _rebuild_for_new_classes(self, new_nc: int):
        """Rebuild detection head for different number of classes."""
        self.nb_classes = new_nc
        self.model.nc = new_nc
        cfg = V9_CONFIGS[self.size]
        self.model.detect = DDetect(
            nc=new_nc,
            ch=cfg['detect_channels'],
            reg_max=self.reg_max,
            stride=(8, 16, 32),
        )
        self.model.detect.to(next(self.model.parameters()).device)

    def train(
        self,
        data: str,
        *,
        # Training parameters
        epochs: int = 300,
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
        name: str = "v9_exp",
        exist_ok: bool = False,

        # Training features
        resume: bool = False,
        amp: bool = True,
        patience: int = 50,

        **kwargs
    ) -> dict:
        """
        Train the YOLOv9 model on a dataset.

        Args:
            data: Path to data.yaml file (required)

            epochs: Number of epochs to train (default: 300)
            batch: Batch size (default: 16)
            imgsz: Input image size (default: 640)

            lr0: Initial learning rate (default: 0.01)
            optimizer: Optimizer name ('SGD', 'Adam', 'AdamW')

            device: Device to train on ('', 'cpu', 'cuda', '0', '0,1,2,3')
            workers: Number of dataloader workers (default: 8)
            seed: Random seed for reproducibility

            project: Root directory for training runs
            name: Experiment name (auto-increments)
            exist_ok: If True, overwrite existing experiment directory

            resume: If True, resume training from checkpoint
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

        Example:
            >>> from libreyolo import LIBREYOLO9
            >>> model = LIBREYOLO9(size='t')
            >>> results = model.train(
            ...     data='coco128.yaml',
            ...     epochs=100,
            ...     batch=16,
            ...     imgsz=640,
            ...     device='0'
            ... )
            >>> print(f"Best mAP: {results['best_mAP50_95']:.3f}")
        """
        from .trainer import V9Trainer
        from .config import V9TrainConfig
        from libreyolo.data import load_data_config

        # Load and validate data config
        try:
            data_config = load_data_config(data, autodownload=True)
            data = data_config.get('yaml_file', data)
        except Exception as e:
            raise FileNotFoundError(f"Failed to load dataset config '{data}': {e}")

        # Reconcile nb_classes with dataset
        yaml_nc = data_config.get('nc')
        if yaml_nc is not None and yaml_nc != self.nb_classes:
            self._rebuild_for_new_classes(yaml_nc)

        # Create training config
        config = V9TrainConfig(
            size=self.size,
            num_classes=self.nb_classes,
            reg_max=self.reg_max,
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
        trainer = V9Trainer(model=self.model, config=config, wrapper_model=self)

        # Resume if requested
        if resume:
            if not self.model_path:
                raise ValueError(
                    "resume=True requires a checkpoint. Load one first: "
                    "model = LIBREYOLO9('path/to/last.pt', size='t'); model.train(data=..., resume=True)"
                )
            trainer.resume(str(self.model_path))

        # Run training
        results = trainer.train()

        # Load best model weights into current instance
        if Path(results['best_checkpoint']).exists():
            self._load_weights(results['best_checkpoint'])

        return results
