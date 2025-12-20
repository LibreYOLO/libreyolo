# Fine-tuning Guide

This guide explains how to fine-tune LibreYOLO on your own dataset.

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

Each image must have a corresponding `.txt` file in the `labels` directory.
Format per line:
```
<class_id> <x_center> <y_center> <width> <height>
```
All coordinates should be normalized to [0, 1].

## Configuration

Create a YAML configuration file (e.g., `my_train_config.yaml`).

```yaml
# Data configuration
data_path: "path/to/dataset/images/train"  # Path to training images

# Model configuration
size: "n"                                    # Model size: "n", "s", "m", "l", "x"
version: "8"                                 # Model version: "8" or "11"
pretrained_weights: "weights/libreyolo8n.pt" # Path to pretrained weights

# Training parameters
epochs: 50
batch_size: 16
lr: 0.001
weight_decay: 0.0005
num_classes: 80
workers: 4
output_dir: "runs/experiment1/"
save_interval: 5 # Save checkpoint every N epochs
```

## Running Training

Use the `libreyolo.training.train` script to start training.

```bash
uv run python -m libreyolo.training.train --config my_train_config.yaml
```

The script will:
1. Load the model and pretrained weights.
2. Prepare the dataset from the path specified.
3. Train for the specified number of epochs.
4. Save checkpoints to `output_dir`.

## Output

Checkpoints will be saved as `libreyolo{version}{size}_epoch_{N}.pt` in the output directory.
You can then use these weights for inference:

```python
model = LIBREYOLO8(model_path="runs/experiment1/libreyolo8n_epoch_50.pt", size="n")
```

