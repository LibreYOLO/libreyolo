# RF5 - Roboflow 5 Training Validation Suite

## Overview

RF5 is a minimal validation subset of the Roboflow100 (RF100) benchmark, designed
to quickly verify that object detection training code works correctly. It serves
as an end-to-end (E2E) test for the LibreYOLO library.

**Key Question RF5 Answers:** "Does our training code work?"

RF5 is NOT a benchmark for model performance comparison. It is a smoke test /
integration test that validates the entire training pipeline from data loading
to model checkpoint saving.


## Motivation

### The Problem

The full Roboflow100 benchmark contains 100 datasets with ~180,000 training images.
Running RF100 takes days of GPU time, making it impractical for:

- Validating code changes during development
- CI/CD pipelines
- Quick sanity checks before merging PRs

### The Solution

RF5 selects 5 carefully chosen datasets (~3,600 training images) that cover the
key edge cases and failure modes a training pipeline might encounter. This allows
developers to validate their changes in ~30-60 minutes instead of days.

```
RF100: ~180,000 images → Days of GPU time
RF5:   ~3,600 images   → 30-60 minutes (50x faster)
```


## Design Principles

### 1. Cover Edge Cases, Not Average Cases

RF5 prioritizes datasets that might break training code:
- Extremely small datasets (batch size edge cases)
- Many classes (loss function stress test)
- Extremely dense annotations (NMS, memory issues)
- Tiny objects (FPN, anchor configuration)
- Large objects (scale handling)

### 2. Minimize Redundancy

Each dataset should test something unique. No two datasets should cover the
same failure mode.

### 3. Fast Execution

Total training time should be under 1 hour for the minimal model set
(one model per family: YOLOX, YOLOv9, RF-DETR).

### 4. Deterministic Results

Same code should produce same results (within floating point tolerance).
Training uses fixed seeds where possible.


## Dataset Selection

### Selection Methodology

We analyzed all 91 available RF100 datasets across these dimensions:

| Dimension | Range in RF100 | Why It Matters |
|-----------|----------------|----------------|
| Dataset size | 16 - 9,306 train images | Batch handling, memory |
| Number of classes | 1 - 59 classes | Loss functions, class weighting |
| Objects per image | 1 - 255 avg | NMS, dense prediction heads |
| Object size | 0.0001 - 0.44 (normalized area) | FPN levels, anchor sizes |
| Domain | 7 categories | Generalization |

### Selected Datasets

We selected 5 datasets that span the extremes of these dimensions:

#### 1. bacteria-ptywi (Microscopic)

```
Purpose:      Minimum viable dataset + tiny objects + high density
Train images: 30
Classes:      1 (bacteria)
Obj/image:    61 (dense)
Object size:  0.0001 (0.01% of image - extremely small)
Domain:       Scientific microscopy
```

**What it tests:**
- Can training handle extremely small datasets?
- Does the model learn to detect tiny objects?
- Does dense annotation cause memory issues?

**Failure modes caught:**
- Batch size > dataset size errors
- Small object detection failures
- NMS breaking on dense predictions

#### 2. circuit-elements (Documents)

```
Purpose:      Stress test for many classes + extreme density
Train images: 672
Classes:      45 (resistors, capacitors, diodes, transistors, etc.)
Obj/image:    255 (extremely dense - highest in RF100)
Object size:  0.0019 (0.2% of image - small)
Domain:       Industrial electronics / circuit diagrams
```

**What it tests:**
- Can the model handle 45+ classes?
- Does training work with 255 objects per image?
- Are class weights computed correctly?

**Failure modes caught:**
- Multi-class loss function bugs
- Memory overflow on dense predictions
- Class imbalance handling issues

#### 3. aquarium-qlnqy (Underwater)

```
Purpose:      Balanced baseline - "typical" dataset
Train images: 448
Classes:      7 (fish, jellyfish, penguin, puffin, shark, starfish, stingray)
Obj/image:    7 (moderate)
Object size:  0.06 (6% of image - medium)
Domain:       Nature / underwater photography
```

**What it tests:**
- Does training work on a "normal" dataset?
- Baseline for comparison with edge cases

**Failure modes caught:**
- General training pipeline bugs
- Basic data loading issues

#### 4. aerial-cows (Aerial)

```
Purpose:      Small objects in aerial/satellite imagery
Train images: 1,084
Classes:      1 (cow)
Obj/image:    14 (moderate density)
Object size:  0.0004 (0.04% of image - very small)
Domain:       Agricultural remote sensing / aerial photography
```

**What it tests:**
- Small object detection (common real-world use case)
- Single-class training
- Aerial imagery domain

**Failure modes caught:**
- FPN not detecting small objects
- Anchor configuration issues
- Single-class loss computation

#### 5. road-signs-6ih4y (Real World)

```
Purpose:      Large objects with sparse annotations
Train images: 1,376
Classes:      21 (various traffic signs)
Obj/image:    1 (sparse - lowest in selection)
Object size:  0.14 (14% of image - large)
Domain:       Traffic / autonomous driving
```

**What it tests:**
- Large object detection
- Sparse annotation handling
- Multi-class with moderate count

**Failure modes caught:**
- Large object scale handling
- Empty/near-empty label file handling
- Class distribution with sparse data


## Coverage Matrix

| Dimension | bacteria | circuit | aquarium | aerial | road-signs |
|-----------|----------|---------|----------|--------|------------|
| Tiny dataset | ✓ | | | | |
| Medium dataset | | ✓ | ✓ | | |
| Large dataset | | | | ✓ | ✓ |
| 1 class | ✓ | | | ✓ | |
| Few classes | | | ✓ | | |
| Many classes | | ✓ | | | ✓ |
| Tiny objects | ✓ | | | ✓ | |
| Medium objects | | ✓ | ✓ | | |
| Large objects | | | | | ✓ |
| Dense (>50/img) | ✓ | ✓ | | | |
| Moderate | | | ✓ | ✓ | |
| Sparse (<5/img) | | | | | ✓ |


## Summary Statistics

```
Total datasets:        5
Total train images:    3,610
Total classes:         75 (unique across datasets)
Estimated train time:  30-60 minutes (minimal model set)
                       2-4 hours (full model matrix)
```

### Comparison with RF100

| Metric | RF5 | RF100 | Ratio |
|--------|-----|-------|-------|
| Datasets | 5 | 100 | 5% |
| Train images | 3,610 | ~180,000 | 2% |
| GPU time | ~1 hour | ~days | ~50x faster |
| Edge case coverage | High | Complete | Targeted |


## Models Tested

RF5 tests all model families supported by LibreYOLO:

### Minimal Set (Default)

One small model per family for fast validation:

| Model | Family | Parameters | Input Size |
|-------|--------|------------|------------|
| yolox-nano | YOLOX | 0.9M | 416×416 |
| yolo9-t | YOLOv9 | 2.0M | 640×640 |
| rfdetr-base | RF-DETR | 29M | 560×560 |

### Full Set (--all-models)

All 15 supported models:

**YOLOX Family (6 models):**
- yolox-nano, yolox-tiny, yolox-s, yolox-m, yolox-l, yolox-x

**YOLOv9 Family (4 models):**
- yolo9-t, yolo9-s, yolo9-m, yolo9-c

**RF-DETR Family (5 models):**
- rfdetr-base, rfdetr-small, rfdetr-medium, rfdetr-large


## Test Configuration

### Training Parameters

RF5 uses minimal epochs to verify training works, not to achieve good accuracy:

```yaml
epochs: 5          # Just enough to verify training loop works
batch_size: 8      # Conservative for memory
imgsz: 640         # Standard (416 for YOLOX nano/tiny)
device: auto       # Use available GPU
patience: 50       # Effectively disabled for short runs
```

### Success Criteria

A training run is considered successful if:

1. Training completes without errors
2. Loss decreases (not NaN or increasing)
3. Checkpoints are saved correctly
4. mAP can be computed (even if low due to few epochs)


## Usage

### As Pytest Tests

```bash
# Quick smoke test (1 model, 1 dataset)
pytest tests/e2e/test_rf5_training.py::test_rf5_quick -v -s

# Test specific model family
pytest tests/e2e/test_rf5_training.py -v -s -k "yolox"

# Test specific dataset
pytest tests/e2e/test_rf5_training.py -v -s -k "aquarium"
```

### As Standalone Script

```bash
# Minimal test (3 models × 5 datasets = 15 runs)
python -m tests.e2e.test_rf5_training

# Specific model and dataset
python -m tests.e2e.test_rf5_training --model yolox-nano --dataset aquarium-qlnqy

# Custom epochs
python -m tests.e2e.test_rf5_training --model yolox-nano --epochs 10

# Full model matrix (15 models × 5 datasets = 75 runs)
python -m tests.e2e.test_rf5_training --all-models

# List available options
python -m tests.e2e.test_rf5_training --list-models
python -m tests.e2e.test_rf5_training --list-datasets
```


## When to Run RF5

### Must Run

- Before merging changes to `libreyolo/training/`
- Before merging changes to loss functions
- Before merging new model architectures
- Before releases

### Should Run

- After modifying data loading code
- After changing augmentation pipelines
- After optimizer/scheduler changes

### Optional

- For routine bug fixes unrelated to training
- For documentation changes


## Data Source

RF5 datasets are sourced from the Roboflow100 collection:

- **Original source:** https://universe.roboflow.com/roboflow-100/
- **Format:** YOLOv8 (ultralytics-compatible)
- **License:** CC BY 4.0 (Creative Commons Attribution)
- **Cache location:** `~/.cache/rf100/`

Datasets are automatically downloaded on first use if not present in cache.


## Alternatives Considered

### RF7 (Rejected)

Adding 2 more datasets for domain coverage:
- bccd-ouzjz (medical/blood cells)
- thermal-dogs-and-people (thermal imaging)

**Decision:** Marginal benefit (+13% time) for domain coverage that doesn't
test additional code paths. RF5 already covers the key failure modes.

### RF10 (Rejected)

Larger subset for better statistical significance.

**Decision:** RF5 is for "does it work?" not "how well does it work?".
For performance measurement, use full RF100.

### Single Dataset (Rejected)

Using just aquarium-qlnqy as a quick test.

**Decision:** Misses edge cases. Many bugs only manifest with tiny datasets,
many classes, or extreme object densities.


## Relationship to RF100

```
┌─────────────────────────────────────────────────────────────┐
│                        RF100                                 │
│   Full benchmark for publication & model comparison          │
│   100 datasets, ~180k images, days of GPU time               │
│                                                              │
│   ┌─────────────────────────────────────────────────────┐   │
│   │                      RF5                             │   │
│   │   E2E test for training code validation              │   │
│   │   5 datasets, ~3.6k images, ~1 hour                  │   │
│   │                                                      │   │
│   │   bacteria | circuit | aquarium | aerial | road-signs│   │
│   └─────────────────────────────────────────────────────┘   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

- **RF5:** "Does the training code work?" (validation)
- **RF100:** "How good is the model?" (benchmarking)


## Output Format

### Console Output

```
======================================================================
RF5 End-to-End Training Test
======================================================================
Models: ['yolox-nano', 'yolo9-t', 'rfdetr-base']
Datasets: ['bacteria-ptywi', 'circuit-elements', 'aquarium-qlnqy',
           'aerial-cows', 'road-signs-6ih4y']
Epochs per dataset: 5
Device: cuda:0
======================================================================

============================================================
Training yolox-nano on bacteria-ptywi
============================================================
...
Training completed in 45.2s
  Best mAP@50: 0.1234
  Best mAP@50-95: 0.0567

[... more training runs ...]

======================================================================
RF5 TEST SUMMARY
======================================================================
Total runs: 15
Successful: 15
Failed: 0
Total time: 32.5 minutes
======================================================================
```

### JSON Results

Results are saved to `tests/e2e/rf5_results/rf5_results_YYYYMMDD_HHMMSS.json`:

```json
{
  "total_runs": 15,
  "successful": 15,
  "failed": 0,
  "total_time_seconds": 1950.5,
  "timestamp": "2025-01-28T15:30:00",
  "config": {
    "epochs": 5,
    "batch_size": 8,
    "device": "cuda:0"
  },
  "results": [
    {
      "model": "yolox-nano",
      "dataset": "bacteria-ptywi",
      "success": true,
      "training_time_seconds": 45.2,
      "best_mAP50": 0.1234,
      "best_mAP50_95": 0.0567,
      "save_dir": "runs/rf5_test/yolox-nano_bacteria-ptywi"
    },
    ...
  ]
}
```


## Future Considerations

### Potential Additions

1. **Validation-only mode:** Skip training, just validate data loading
2. **Checkpoint resume test:** Verify training can resume from checkpoint
3. **Multi-GPU test:** Validate distributed training works
4. **Export test:** Verify trained models can be exported to ONNX

### Not Planned

1. **Accuracy thresholds:** RF5 tests functionality, not performance
2. **Automated CI:** Requires GPU infrastructure we don't have
3. **More datasets:** 5 is sufficient for code validation


## References

- Roboflow 100 Paper: https://arxiv.org/abs/2211.13523
- Roboflow 100 GitHub: https://github.com/roboflow/roboflow-100-benchmark
- Roboflow Universe: https://universe.roboflow.com/roboflow-100


## Appendix: Full Dataset Analysis

Data collected from RF100 datasets on 2025-01-28:

```
Dataset                    Classes  Train   Obj/img  Bbox Area  Category
---------------------------------------------------------------------------
bacteria-ptywi                  1      30     61.0    0.0001    microscopic
cells-uyemf                     1      16    105.4    0.0019    microscopic
circuit-elements               45     672    254.7    0.0019    documents
team-fight-tactics             59    1162      1.9    0.0056    videogames
poker-cards-cxcvz              53     964      4.3    0.1117    real_world
paper-parts                    46    8472      4.6    0.1636    documents
aerial-cows                     1    1084     13.6    0.0004    aerial
aquarium-qlnqy                  7     448      7.1    0.0600    underwater
bccd-ouzjz                      3     255     14.0    0.0402    microscopic
road-signs-6ih4y               21    1376      1.1    0.1422    real_world
flir-camera-objects             4    9306      6.6    0.0119    electromagnetic
brain-tumor-m2pbp               3    6930      2.3    0.0353    medical
...
---------------------------------------------------------------------------
```

Selection criteria applied:
1. bacteria-ptywi: Smallest viable + tiny objects + dense
2. circuit-elements: Most classes in small dataset + densest
3. aquarium-qlnqy: Median characteristics (baseline)
4. aerial-cows: Very small objects + real-world aerial use case
5. road-signs-6ih4y: Large objects + sparse + multi-class
