# YOLOv8/v11 Fine-Tuning

```{warning}
**EXPERIMENTAL**: The training pipeline for YOLOv8 and YOLOv11 models is under active development. The API may change in future versions. For production training workflows, consider using the YOLOX training API which is more stable.
```

This guide explains how to fine-tune YOLOv8 and YOLOv11 models on custom datasets using LibreYOLO's training module.

## Overview

Fine-tuning allows you to adapt pre-trained COCO weights to your specific object detection task with fewer training epochs and less data than training from scratch.

## Dataset Preparation

LibreYOLO expects data in the standard YOLO format.

### Directory Structure

```
dataset/
├── images/
│   ├── train/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   └── val/
│       └── image3.jpg
└── labels/
    ├── train/
    │   ├── image1.txt
    │   └── image2.txt
    └── val/
        └── image3.txt
```

### Label Format

Each image needs a corresponding `.txt` file with one line per object:

```
<class_id> <x_center> <y_center> <width> <height>
```

All coordinates are normalized to `[0, 1]`:

```
0 0.5 0.5 0.2 0.3
1 0.3 0.7 0.15 0.25
```

Where:
- `class_id`: Integer class index (0-indexed)
- `x_center`, `y_center`: Center of bounding box relative to image width/height
- `width`, `height`: Box dimensions relative to image width/height

## Configuration

Create a YAML configuration file for training:

```yaml
# train_config.yaml

# Data
data_path: "path/to/dataset/images/train"

# Model
size: "n"                                    # n, s, m, l, x
version: "8"                                 # 8 or 11
pretrained_weights: "weights/libreyolo8n.pt"

# Training
epochs: 50
batch_size: 16
lr: 0.001
weight_decay: 0.0005
num_classes: 80
workers: 4

# Output
output_dir: "runs/training/"
save_interval: 5  # Save every N epochs
```

### Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data_path` | str | required | Path to training images directory |
| `size` | str | required | Model size: `n`, `s`, `m`, `l`, `x` |
| `version` | str | `"8"` | Model version: `"8"` or `"11"` |
| `pretrained_weights` | str | optional | Path to pretrained weights |
| `epochs` | int | 50 | Number of training epochs |
| `batch_size` | int | 16 | Batch size |
| `lr` | float | 0.001 | Initial learning rate |
| `weight_decay` | float | 0.0005 | Weight decay for regularization |
| `num_classes` | int | 80 | Number of object classes |
| `workers` | int | 4 | Number of data loader workers |
| `output_dir` | str | `"runs/"` | Output directory for checkpoints |
| `save_interval` | int | 5 | Save checkpoint every N epochs |

## Running Training

Use the training module to start fine-tuning:

```bash
uv run python -m libreyolo.training.train --config train_config.yaml
```

Or with pip:

```bash
python -m libreyolo.training.train --config train_config.yaml
```

The training script will:
1. Load the model architecture based on `size` and `version`
2. Load pretrained weights (if specified)
3. Prepare the dataset from `data_path`
4. Train for the specified number of epochs
5. Save checkpoints to `output_dir`

## Using Trained Weights

After training, load your fine-tuned model:

```python
from libreyolo import LIBREYOLO

# Load your trained model
model = LIBREYOLO(
    model_path="runs/training/libreyolo8n_epoch_50.pt",
    size="n"
)

# Run inference
results = model(image="test_image.jpg", save=True)
```

## Data Augmentation

The training pipeline applies the following augmentations automatically:

| Augmentation | Description |
|--------------|-------------|
| Mosaic | Combines 4 images into one training sample |
| Mixup | Blends two images with their labels |
| HSV Jitter | Random hue, saturation, and value adjustments |
| Horizontal Flip | Random left-right flipping |
| Random Affine | Rotation, scaling, translation, and shear |

## Training Tips

### Learning Rate

- **Fine-tuning**: Start with `lr=0.001` (default)
- **Training from scratch**: Use higher `lr=0.01`
- Learning rate is automatically reduced using cosine annealing

### Batch Size

- Larger batch sizes provide more stable gradients
- Reduce if running out of GPU memory
- Effective batch size should be at least 16 for stable training

### Number of Epochs

- Fine-tuning typically converges in 50-100 epochs
- Training from scratch may require 300+ epochs
- Monitor validation loss to detect overfitting

### Class Imbalance

If your dataset has imbalanced classes:
- Use focal loss (configured automatically)
- Consider oversampling minority classes
- Adjust class weights in the loss function

## Output Structure

Training outputs are organized as follows:

```
runs/training/
├── exp_20240101_120000/
│   ├── config.yaml           # Training configuration
│   ├── libreyolo8n_epoch_5.pt
│   ├── libreyolo8n_epoch_10.pt
│   ├── libreyolo8n_epoch_50.pt
│   └── training_log.txt      # Training logs
```

## Checkpoint Format

Checkpoints contain the model state dict and can be loaded directly:

```python
import torch

# Inspect checkpoint
checkpoint = torch.load("libreyolo8n_epoch_50.pt")
print(checkpoint.keys())  # Shows state dict keys
```

## Comparison: YOLOv8/v11 vs YOLOX Training

| Feature | YOLOv8/v11 | YOLOX |
|---------|------------|-------|
| Training API | CLI config-based | `.train()` method |
| Status | Experimental | Stable |
| EMA Support | Limited | Full |
| Mixed Precision | Basic | Full AMP |
| Resume Training | Via config | Via `resume` param |

For more stable training with advanced features, consider using {doc}`YOLOX <yolox>`.

## Troubleshooting

### Out of Memory

```
RuntimeError: CUDA out of memory
```

Solutions:
- Reduce `batch_size`
- Use smaller model size (`n` instead of `s`)
- Enable gradient checkpointing (if available)

### Slow Training

- Increase `workers` for faster data loading
- Ensure images are stored on SSD
- Use mixed precision training

### Loss Not Decreasing

- Check that labels are in correct format
- Verify `num_classes` matches your dataset
- Try reducing learning rate
- Ensure pretrained weights match model size

## Next Steps

- {doc}`inference` - Run inference with trained models
- {doc}`explainability` - Visualize model decisions
- {doc}`yolox` - Alternative training with YOLOX

