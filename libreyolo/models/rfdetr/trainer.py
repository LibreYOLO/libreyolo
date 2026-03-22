"""
Training module for RF-DETR.
Wraps the original rfdetr training API with Ultralytics-compatible interface.
"""

from typing import Dict

from rfdetr import RFDETRLarge, RFDETRNano, RFDETRSmall, RFDETRMedium
from rfdetr import RFDETRSegLarge, RFDETRSegNano, RFDETRSegSmall, RFDETRSegMedium

RFDETR_TRAINERS = {
    "n": RFDETRNano,
    "s": RFDETRSmall,
    "m": RFDETRMedium,
    "l": RFDETRLarge,
}

RFDETR_SEG_TRAINERS = {
    "n": RFDETRSegNano,
    "s": RFDETRSegSmall,
    "m": RFDETRSegMedium,
    "l": RFDETRSegLarge,
}


def train_rfdetr(
    data: str,
    size: str = "s",
    epochs: int = 100,
    batch_size: int = 4,
    lr: float = 1e-4,
    output_dir: str = "runs/train",
    resume: str | None = None,
    pretrain_weights: str | None = None,
    segmentation: bool = False,
    **kwargs,
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
        pretrain_weights: Path to pretrained weights. When provided the
            upstream rfdetr trainer reuses these instead of downloading
            its own copy from Google Cloud Storage.
        **kwargs: Additional args passed to rfdetr train()

    Returns:
        Dictionary with training results

    Example:
        >>> from libreyolo.rfdetr import train_rfdetr
        >>> train_rfdetr(data="path/to/dataset", size="s", epochs=50)
    """
    trainers = RFDETR_SEG_TRAINERS if segmentation else RFDETR_TRAINERS
    if size not in trainers:
        raise ValueError(
            f"Invalid size: {size}. Must be one of {list(trainers.keys())}"
        )

    trainer_cls = trainers[size]
    init_kwargs = {}
    if pretrain_weights is not None:
        init_kwargs["pretrain_weights"] = pretrain_weights
    model = trainer_cls(**init_kwargs)

    model.train(
        dataset_dir=data,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        output_dir=output_dir,
        resume=resume,
        **kwargs,
    )

    return {
        "output_dir": output_dir,
        "model": model,
    }
