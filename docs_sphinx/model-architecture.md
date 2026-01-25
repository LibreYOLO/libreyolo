# Model Architecture

This page provides technical details about the model architectures supported by LibreYOLO.

## Supported Models

LibreYOLO supports five model architectures:

| Model | Class | Sizes | Key Features |
|-------|-------|-------|--------------|
| YOLOv8 | `LIBREYOLO8` | n, s, m, l, x | C2F blocks, anchor-free |
| YOLOv9 | `LIBREYOLO9` | t, s, m, c | ELAN blocks, anchor-free, GELAN architecture |
| YOLOv11 | `LIBREYOLO11` | n, s, m, l, x | C3k2 blocks, C2PSA attention |
| YOLOX | `LIBREYOLOX` | nano, tiny, s, m, l, x | CSPDarknet backbone, decoupled head |
| RF-DETR | `LIBREYOLORFDETR` | n, s, b, m, l | Detection Transformer, DINOv2, deformable attention |

## Architecture Overview

YOLO models (v8, v9, v11, YOLOX) follow a similar high-level structure:

```
Input Image → Backbone → Neck (FPN/PAN) → Detection Heads → Output
```

RF-DETR uses a different paradigm based on Detection Transformers:

```
Input Image → DINOv2 Backbone → Transformer Encoder → Transformer Decoder → Output
```

### Backbone

**YOLO models**: Extracts hierarchical features at multiple scales (P3, P4, P5 corresponding to strides 8, 16, 32).

**RF-DETR**: Uses DINOv2 Vision Transformer for feature extraction.

### Neck

**YOLO models**: Feature Pyramid Network (FPN) with Path Aggregation Network (PAN) for multi-scale feature fusion.

**RF-DETR**: Uses deformable attention in transformer encoder/decoder instead of traditional FPN/PAN.

### Detection Heads

**YOLO models**: Produce bounding box coordinates, objectness scores, and class probabilities at each scale.

**RF-DETR**: Set-based prediction using transformer decoder queries (no anchors or explicit scales).

---

## YOLOv8 Architecture

YOLOv8 uses **C2F** (Cross Stage Partial with 2 convolutions and Fusion) blocks throughout.

### Model Sizes

| Size | Width Multiple | Depth Multiple | Parameters (approx) |
|------|----------------|----------------|---------------------|
| `n` | 0.25 | 0.33 | 3.2M |
| `s` | 0.50 | 0.33 | 11.2M |
| `m` | 0.75 | 0.67 | 25.9M |
| `l` | 1.00 | 1.00 | 43.7M |
| `x` | 1.25 | 1.00 | 68.2M |

### Layer Reference

| Layer Name | Module | Description | Stride |
|------------|--------|-------------|--------|
| `backbone_p1` | Conv | First convolution (stem) | 2 |
| `backbone_p2` | Conv | Second convolution | 4 |
| `backbone_c2f1` | C2F | First C2F block | 4 |
| `backbone_p3` | Conv | Downsample to P3 | 8 |
| `backbone_c2f2_P3` | C2F | C2F at P3 scale | 8 |
| `backbone_p4` | Conv | Downsample to P4 | 16 |
| `backbone_c2f3_P4` | C2F | C2F at P4 scale | 16 |
| `backbone_p5` | Conv | Downsample to P5 | 32 |
| `backbone_c2f4` | C2F | Fourth C2F block | 32 |
| `backbone_sppf_P5` | SPPF | Spatial Pyramid Pooling - Fast | 32 |
| `neck_c2f21` | C2F | Neck upsampling path | - |
| `neck_c2f11` | C2F | Neck P3 fusion | 8 |
| `neck_c2f12` | C2F | Neck P4 fusion | 16 |
| `neck_c2f22` | C2F | Neck P5 fusion | 32 |
| `head8_conv11` | Conv | P3 head - box branch | 8 |
| `head8_conv21` | Conv | P3 head - class branch | 8 |
| `head16_conv11` | Conv | P4 head - box branch | 16 |
| `head16_conv21` | Conv | P4 head - class branch | 16 |
| `head32_conv11` | Conv | P5 head - box branch | 32 |
| `head32_conv21` | Conv | P5 head - class branch | 32 |

### Key Components

#### C2F Block

```
Input
  │
  ├─────────────┐
  │             │
  Conv 1x1      │
  │             │
  Split         │
  │     │       │
  │   Bottleneck(s)
  │     │       │
  └──Concat─────┘
        │
    Conv 1x1
        │
     Output
```

#### SPPF (Spatial Pyramid Pooling - Fast)

Applies multiple max pooling operations sequentially and concatenates the results for multi-scale context.

---

## YOLOv11 Architecture

YOLOv11 builds upon YOLOv8 with **C3k2** blocks and adds the **C2PSA** (Partial Self-Attention) module for enhanced feature extraction.

### Model Sizes

Same width/depth multiples as YOLOv8.

### Layer Reference

YOLOv11 includes all YOLOv8 layers plus:

| Layer Name | Module | Description | Stride |
|------------|--------|-------------|--------|
| `backbone_c2psa_P5` | C2PSA | Partial Self-Attention at P5 | 32 |

### Complete Layer List

| Layer Name | Module | Description | Stride |
|------------|--------|-------------|--------|
| `backbone_p1` | Conv | First convolution (stem) | 2 |
| `backbone_p2` | Conv | Second convolution | 4 |
| `backbone_c2f1` | C3k2 | First C3k2 block | 4 |
| `backbone_p3` | Conv | Downsample to P3 | 8 |
| `backbone_c2f2_P3` | C3k2 | C3k2 at P3 scale | 8 |
| `backbone_p4` | Conv | Downsample to P4 | 16 |
| `backbone_c2f3_P4` | C3k2 | C3k2 at P4 scale | 16 |
| `backbone_p5` | Conv | Downsample to P5 | 32 |
| `backbone_c2f4` | C3k2 | Fourth C3k2 block | 32 |
| `backbone_sppf_P5` | SPPF | Spatial Pyramid Pooling - Fast | 32 |
| `backbone_c2psa_P5` | C2PSA | Partial Self-Attention | 32 |
| `neck_c2f21` | C3k2 | Neck upsampling path | - |
| `neck_c2f11` | C3k2 | Neck P3 fusion | 8 |
| `neck_c2f12` | C3k2 | Neck P4 fusion | 16 |
| `neck_c2f22` | C3k2 | Neck P5 fusion | 32 |
| `head8_conv11` | Conv | P3 head - box branch | 8 |
| `head8_conv21` | Conv | P3 head - class branch | 8 |
| `head16_conv11` | Conv | P4 head - box branch | 16 |
| `head16_conv21` | Conv | P4 head - class branch | 16 |
| `head32_conv11` | Conv | P5 head - box branch | 32 |
| `head32_conv21` | Conv | P5 head - class branch | 32 |

### Key Components

#### C3k2 Block

Similar to C2F but with a different internal structure using CSP bottlenecks with kernel size k.

#### C2PSA (Partial Self-Attention)

Adds self-attention mechanism at the P5 scale to capture long-range dependencies:

```
Input
  │
  Split (channels)
  │         │
  │    Attention
  │         │
  └──Concat─┘
       │
    Conv 1x1
       │
    Output
```

---

## YOLOX Architecture

YOLOX uses a **CSPDarknet** backbone with a **PAFPN** neck and a **decoupled detection head**.

### Model Sizes

| Size | Width | Depth | Input Size | Parameters (approx) |
|------|-------|-------|------------|---------------------|
| `nano` | 0.25 | 0.33 | 416 | 0.9M |
| `tiny` | 0.375 | 0.33 | 416 | 5.1M |
| `s` | 0.50 | 0.33 | 640 | 9.0M |
| `m` | 0.75 | 0.67 | 640 | 25.3M |
| `l` | 1.00 | 1.00 | 640 | 54.2M |
| `x` | 1.25 | 1.33 | 640 | 99.1M |

### Architecture Components

#### CSPDarknet Backbone

Uses CSP (Cross Stage Partial) bottleneck blocks with Darknet-style convolutions:

- **Focus/Stem**: Initial feature extraction
- **Dark2-5**: Progressive downsampling stages with CSP bottlenecks
- **SPP**: Spatial Pyramid Pooling at the end

#### PAFPN Neck

Path Aggregation Feature Pyramid Network:
- Top-down pathway (FPN): High-level features flow to lower levels
- Bottom-up pathway (PAN): Low-level features flow to higher levels

#### Decoupled Head

Unlike YOLOv8/v11, YOLOX uses separate branches for:
1. **Classification**: Predicts object classes
2. **Regression**: Predicts bounding box coordinates
3. **Objectness**: Predicts object presence

```
FPN Feature
     │
  ┌──┴──┐
  │     │
 Cls   Reg+Obj
  │     │
  └──┬──┘
     │
  Output
```

---

## RF-DETR Architecture

RF-DETR (Real-time Detection Transformer) uses a fundamentally different approach based on **DETR** (Detection Transformer) architecture with optimizations for real-time inference.

### Model Sizes

| Size | Backbone | Encoder Layers | Decoder Layers | Queries | Parameters (approx) |
|------|----------|----------------|----------------|---------|---------------------|
| `n` | DINOv2-S | 6 | 6 | 300 | ~20M |
| `s` | DINOv2-S | 6 | 6 | 300 | ~25M |
| `b` | DINOv2-B | 6 | 6 | 300 | ~85M |
| `m` | DINOv2-L | 6 | 6 | 300 | ~150M |
| `l` | DINOv2-G | 6 | 6 | 300 | ~300M |

### Architecture Components

#### DINOv2 Backbone

RF-DETR uses **DINOv2** (self-supervised Vision Transformer) as its backbone:

- Pre-trained on large-scale unlabeled data
- Provides rich semantic features
- Multi-scale feature extraction via patch embeddings
- Requires **ImageNet normalization** (handled automatically)

```python
# ImageNet normalization (automatic in LibreYOLO)
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
```

#### Transformer Encoder

Multi-scale deformable attention mechanism:
- Processes multi-scale features from backbone
- Deformable attention focuses on relevant spatial locations
- More efficient than standard self-attention

```
Multi-scale Features (from DINOv2)
         │
    Deformable
     Attention
         │
    Feed Forward
         │
   Encoded Features
```

#### Transformer Decoder

Set-based object detection using learned queries:
- **300 object queries** (learnable embeddings)
- Each query predicts one potential object
- Cross-attention with encoded features
- Self-attention between queries

```
Object Queries (300)
         │
   Self-Attention
         │
  Cross-Attention ← Encoded Features
         │
   Feed Forward
         │
    Detection Outputs
         │
    ┌────┴────┐
    │         │
  Boxes    Classes
```

#### Detection Head

Unlike YOLO's anchor-based or anchor-free heads, RF-DETR uses:
- Direct box regression (normalized coordinates)
- Classification logits for each query
- No NMS needed (set-based prediction eliminates duplicates)
- Hungarian matching during training

### Key Differences from YOLO

| Aspect | YOLO (v8/v9/v11/X) | RF-DETR |
|--------|-------------------|---------|
| **Paradigm** | Dense prediction (per-pixel) | Set prediction (queries) |
| **Backbone** | CNN-based (CSP, C2F, ELAN) | Vision Transformer (DINOv2) |
| **Feature Fusion** | FPN/PAN | Deformable attention |
| **Detection** | Multi-scale anchors/anchor-free | Object queries |
| **Post-processing** | NMS required | No NMS (one-to-one matching) |
| **Normalization** | 0-1 scaling | ImageNet normalization |
| **Training** | Dense labels | Bipartite matching loss |

### Advantages

- **No NMS**: Set prediction eliminates duplicate detections
- **Global context**: Transformer attention captures long-range dependencies
- **Pre-trained backbone**: DINOv2 provides strong semantic features
- **End-to-end**: Differentiable throughout (no hand-crafted components)

### Trade-offs

- **Slower inference**: Transformer operations are more computationally expensive than CNNs
- **Memory intensive**: Self-attention scales quadratically with sequence length
- **Different preprocessing**: Requires ImageNet normalization (not compatible with YOLO preprocessing)

### Usage Example

```python
from libreyolo import LIBREYOLO

# RF-DETR auto-detected and uses ImageNet preprocessing
model = LIBREYOLO("weights/libreyolorfdetrn.pt")
results = model(image="image.jpg", save=True)

# Or use explicit class
from libreyolo import LIBREYOLORFDETR
model = LIBREYOLORFDETR("weights/libreyolorfdetrn.pt", size="n")
```

---

## Programmatic Layer Access

### Getting Available Layers

```python
from libreyolo import LIBREYOLO

# Auto-detect model version and size
model = LIBREYOLO("weights/libreyolo8n.pt")

# List all available layers
layers = model.get_available_layer_names()
print(layers)
```

### Feature Map Extraction

Available for YOLOv8 models only:

```python
# Save all feature maps
model = LIBREYOLO(
    "weights/libreyolo8n.pt",
    save_feature_maps=True
)

# Or save specific layers only
model = LIBREYOLO(
    "weights/libreyolo8n.pt",
    save_feature_maps=["backbone_c2f2_P3", "neck_c2f22"]
)

results = model(image="image.jpg")
print(results.get("feature_maps_path"))
```

### Understanding Layer Hierarchy

Different layers capture different levels of abstraction:

| Layer Type | Example | Features Captured |
|------------|---------|-------------------|
| Early backbone | `backbone_p1`, `backbone_p2` | Low-level (edges, textures, colors) |
| Mid backbone | `backbone_c2f2_P3`, `backbone_c2f3_P4` | Mid-level (parts, patterns, shapes) |
| Late backbone | `backbone_sppf_P5`, `backbone_c2psa_P5` | High-level (object parts, context) |
| Neck | `neck_c2f22` | Semantic features (full objects, relationships) |

---

## Detection Output Format

All models output detections in a unified format:

```python
{
    "boxes": [[x1, y1, x2, y2], ...],  # xyxy format, pixel coordinates
    "scores": [0.95, 0.87, ...],        # Confidence scores [0, 1]
    "classes": [0, 2, 15, ...],         # Class IDs (COCO: 0-79)
    "num_detections": 5,
    "source": "image.jpg"
}
```

### Post-Processing Pipeline

1. **Box Decoding**: Convert raw predictions to bounding boxes
2. **Confidence Filtering**: Remove predictions below `conf_thres`
3. **NMS (Non-Maximum Suppression)**: Remove overlapping boxes above `iou_thres`
4. **Coordinate Scaling**: Scale boxes back to original image dimensions

