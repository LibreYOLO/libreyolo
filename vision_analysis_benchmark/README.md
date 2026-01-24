# Vision Analysis Benchmark

Comprehensive benchmarking suite for LibreYOLO models following the Vision Analysis benchmark specification v1.0.

## Overview

This benchmark measures complete end-to-end performance of object detection models:
- **Accuracy**: mAP@50, mAP@50-95, size-specific metrics
- **Timing**: Detailed breakdown of preprocess, inference, postprocess stages
- **Throughput**: FPS, latency percentiles (p50, p95, p99)
- **Model Stats**: Parameters, GFLOPs

## Target Hardware

- NVIDIA A100 GPU (40GB or 80GB)
- CUDA-enabled system
- Ubuntu 22.04 LTS (recommended)

## Setup

### 1. Install LibreYOLO

```bash
pip install -e ..
```

### 2. Install Benchmark Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download COCO val2017

```bash
# Download and extract COCO val2017 dataset
# Expected structure:
# coco/
# ├── val2017/         (5000 images)
# └── annotations/
#     └── instances_val2017.json
```

### 4. Prepare GPU

```bash
# Enable persistence mode
sudo nvidia-smi -pm 1

# Lock GPU clocks to maximum (optional, improves consistency)
# sudo nvidia-smi -lgc <max_clock>

# Verify no other GPU workloads
nvidia-smi
```

## Usage

### Benchmark All LibreYOLO Models

```bash
python benchmark_coco.py \
    --coco-yaml vision_analysis_benchmark/coco_benchmark.yaml \
    --output-dir ./results
```

### Benchmark Specific Models

```bash
# Single model
python benchmark_coco.py \
    --coco-yaml vision_analysis_benchmark/coco_benchmark.yaml \
    --models yolov8n \
    --output-dir ./results

# Multiple models
python benchmark_coco.py \
    --coco-yaml vision_analysis_benchmark/coco_benchmark.yaml \
    --models yolov8n yolov8s yolov11m \
    --output-dir ./results
```

### Options

- `--coco-yaml`: Path to COCO dataset YAML configuration file
- `--output-dir`: Output directory for results (default: `./results`)
- `--models`: Specific models to benchmark (default: all LibreYOLO models)
- `--batch-size`: Batch size for inference (default: 1 for accurate latency)
- `--device`: Device to use (default: auto-detect)

## Output

### Per-Model JSON

Each model generates a detailed JSON file with:
- Model metadata (name, family, variant, input size)
- Hardware/software environment
- Accuracy metrics (mAP@50, mAP@50-95, size-specific)
- Detailed timing breakdown (mean, std, p50, p95, p99)
- Throughput metrics (FPS, latency percentiles)
- Model statistics (parameters, GFLOPs)

Example: `results/yolov8n_benchmark.json`

### Summary CSV

All models compiled into a single CSV for easy comparison:
```
results/benchmark_summary.csv
```

## LibreYOLO Models Benchmarked

| Family   | Variants          |
|----------|-------------------|
| YOLOv8   | n, s, m, l, x     |
| YOLOv9   | t, s, m, c        |
| YOLOv11  | n, s, m, l, x     |
| YOLOX    | nano, tiny, s, m, l, x |

## Publishing Results

Results are published to [visionanalysis.org](https://visionanalysis.org) for public comparison with other YOLO implementations.

## Troubleshooting

### High Timing Variance

- Check for thermal throttling: `nvidia-smi dmon`
- Ensure no other GPU workloads: `nvidia-smi`
- Lock GPU clocks to max frequency

### Out of Memory

- Reduce batch size: `--batch-size 1`
- Use smaller model variants

### Accuracy Mismatch

- Verify COCO dataset integrity
- Check model weights are correctly loaded
- Ensure preprocessing matches training configuration

## Specification

This benchmark follows the Vision Analysis Benchmark Specification v1.0.
See `../gonzalo/yolo/23-01-2026/1_prompts/validation.md` for full specification.
