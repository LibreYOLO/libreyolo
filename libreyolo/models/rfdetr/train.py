"""
Training module for RF-DETR.
Wraps the original rfdetr training API with Ultralytics-compatible interface.
"""

from pathlib import Path
from typing import Dict, Optional

# Use the original rfdetr training which is already well-tuned
from rfdetr import RFDETRLarge, RFDETRNano, RFDETRSmall, RFDETRMedium

# Map size codes to rfdetr classes
RFDETR_TRAINERS = {
    'n': RFDETRNano,
    's': RFDETRSmall,
    'm': RFDETRMedium,
    'l': RFDETRLarge,
}


def train_rfdetr(
    data: str,
    size: str = "s",
    epochs: int = 100,
    batch_size: int = 4,
    lr: float = 1e-4,
    output_dir: str = "runs/train",
    resume: str = None,
    **kwargs
) -> Dict:
    """
    Train RF-DETR using the original training implementation.

    This wraps the official rfdetr training API which includes:
    - EMA (Exponential Moving Average)
    - Proper warmup and cosine LR schedule
    - Hungarian matching loss
    - Distributed training support

    Args:
        data: Path to dataset (Roboflow format with COCO annotations)
        size: Model size ('n', 's', 'm', 'l')
        epochs: Number of training epochs
        batch_size: Batch size per GPU
        lr: Learning rate
        output_dir: Directory to save outputs
        resume: Path to checkpoint to resume from
        **kwargs: Additional args passed to rfdetr train()

    Returns:
        Dictionary with training results

    Example:
        >>> from libreyolo.rfdetr import train_rfdetr
        >>> train_rfdetr(data="path/to/dataset", size="s", epochs=50)
    """
    if size not in RFDETR_TRAINERS:
        raise ValueError(f"Invalid size: {size}. Must be one of {list(RFDETR_TRAINERS.keys())}")

    # Create the appropriate trainer
    trainer_cls = RFDETR_TRAINERS[size]
    model = trainer_cls()

    # Train with the original API
    model.train(
        dataset_dir=data,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        output_dir=output_dir,
        resume=resume,
        **kwargs
    )

    return {
        "output_dir": output_dir,
        "model": model,
    }
