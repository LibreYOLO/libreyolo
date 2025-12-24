# Explainability (XAI)

LibreYOLO includes built-in explainability methods to visualize what the model focuses on when making predictions.

## Available CAM Methods

| Method | Description | Gradient-Free |
|--------|-------------|---------------|
| `eigencam` | SVD-based, fast and reliable | ✅ |
| `gradcam` | Gradient-weighted class activation | ❌ |
| `gradcam++` | Improved GradCAM with second-order gradients | ❌ |
| `xgradcam` | Axiom-based GradCAM | ❌ |
| `hirescam` | High-resolution CAM | ❌ |
| `layercam` | Layer-wise CAM | ❌ |
| `eigengradcam` | Eigen-based gradient CAM | ❌ |

## Basic Usage

```python
from libreyolo import LIBREYOLO

model = LIBREYOLO(model_path="weights/libreyolo8n.pt", size="n")

# Generate explanation
result = model.explain(
    image="image.jpg",
    method="gradcam",
    save=True
)

# Access the heatmap
heatmap = result["heatmap"]       # Grayscale [0, 1]
overlay = result["overlay"]       # RGB overlay image
```

## Explain Method Parameters

```python
result = model.explain(
    image="image.jpg",
    method="eigencam",           # CAM method
    target_layer="neck_c2f22",   # Layer to visualize
    eigen_smooth=False,          # SVD smoothing
    save=True,                   # Save to disk
    output_path="results/",      # Custom save location
    alpha=0.5,                   # Overlay blending factor
)
```

## Target Layers

Use `get_available_layer_names()` to see available layers:

```python
model = LIBREYOLO("weights/libreyolo8n.pt", size="n")
print(model.get_available_layer_names())
```

### YOLO8 Layers

| Layer | Description |
|-------|-------------|
| `backbone_p1` | First convolution |
| `backbone_c2f2_P3` | C2F at P3 (Stride 8) |
| `backbone_c2f3_P4` | C2F at P4 (Stride 16) |
| `backbone_sppf_P5` | SPPF at P5 (Stride 32) |
| `neck_c2f21` | Neck C2F block 1 |
| `neck_c2f22` | Neck C2F block 4 (default) |

### Choosing a Layer

- **Earlier layers** (backbone_p1, p2): Low-level features (edges, textures)
- **Middle layers** (c2f2, c2f3): Mid-level features (parts, patterns)
- **Later layers** (neck_c2f22): High-level features (objects, semantics)

## Auto-Save During Inference

Enable automatic CAM visualization during every inference:

```python
model = LIBREYOLO(
    model_path="weights/libreyolo8n.pt",
    size="n",
    save_eigen_cam=True,      # Auto-save EigenCAM
    cam_method="eigencam",    # Default method
    cam_layer="neck_c2f22"    # Target layer
)

# CAM is automatically saved alongside detections
results = model(image="image.jpg", save=True)
print(results.get("eigen_cam_path"))
```

## Feature Map Visualization

Save raw feature maps from model layers:

```python
# Save all layers
model = LIBREYOLO(
    model_path="weights/libreyolo8n.pt",
    size="n",
    save_feature_maps=True
)

# Save specific layers only
model = LIBREYOLO(
    model_path="weights/libreyolo8n.pt",
    size="n",
    save_feature_maps=["backbone_p1", "neck_c2f22"]
)

results = model(image="image.jpg")
print(results.get("feature_maps_path"))
```

## Output Format

The `explain()` method returns:

```python
{
    "heatmap": np.ndarray,        # Shape (H, W), values [0, 1]
    "overlay": np.ndarray,        # Shape (H, W, 3), RGB uint8
    "original_image": PIL.Image,  # Original input image
    "method": "gradcam",          # Method used
    "target_layer": "neck_c2f22", # Layer used
    "saved_path": "runs/gradcam/..." # If save=True
}
```

## Comparing Methods

```python
methods = ["eigencam", "gradcam", "gradcam++", "xgradcam"]

for method in methods:
    result = model.explain(
        image="image.jpg",
        method=method,
        save=True
    )
    print(f"{method}: saved to {result['saved_path']}")
```

