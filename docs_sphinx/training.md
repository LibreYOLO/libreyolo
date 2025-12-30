# Training Guide

This guide explains how to train and fine-tune LibreYOLO models on custom datasets.

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

## Configuration

Create a YAML configuration file:

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

## Running Training

```bash
uv run python -m libreyolo.training.train --config train_config.yaml
```

## Using Trained Weights

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

## Training Tips

### Learning Rate

- Start with `lr=0.001` for fine-tuning
- Use `lr=0.01` for training from scratch

### Batch Size

- Larger batch sizes = more stable training
- Reduce if running out of GPU memory

### Data Augmentation

LibreYOLO applies the following augmentations during training:

| Augmentation | Description |
|--------------|-------------|
| Mosaic | Combines 4 images into one training sample |
| Mixup | Blends two images with their labels |
| HSV Jitter | Random hue, saturation, and value adjustments |
| Horizontal Flip | Random left-right flipping |
| Random Affine | Rotation, scaling, translation, and shear |

These augmentations are automatically applied during training and can be configured via the training config.

## Monitoring Training

Training logs are saved to the output directory:

```
runs/training/
├── libreyolo8n_epoch_5.pt
├── libreyolo8n_epoch_10.pt
├── ...
└── training_log.txt
```

