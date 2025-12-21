# Libre YOLO

Libre YOLO is an open-source, MIT-licensed implementation of YOLO object detection models. It provides a clean, independent codebase for training and inference, designed to be free from restrictive licensing for the software itself.

> **Note:** While this codebase is MIT licensed, pre-trained weights converted from other repositories (like Ultralytics) may inherit their original licenses (often AGPL-3.0). Please check the license of the specific weights you use.

## Features

- 🚀 **Supported Models:** Full support for **YOLOv8** and **YOLOv11** architectures.
- 📦 **Unified API:** Simple, consistent interface for loading and using different YOLO versions.
- 🛠️ **Training Engine:** Built-in support for training models on custom datasets.
- ⚖️ **MIT License:** Permissive licensing for the codebase, making it suitable for commercial and research integration.
- 🔄 **Weight Conversion:** Tools to convert weights from Ultralytics format to LibreYOLO.

## Installation

You can install Libre YOLO directly from the source:

```bash
git clone https://github.com/Libre-YOLO/libreyolo.git
cd libreyolo
pip install -e .
```

To include dependencies for weight conversion:

```bash
uv sync --extra convert
```

## Quick Start

### Inference

Libre YOLO provides a unified factory to load models. It automatically detects the model version (v8 or v11) from the weights.

```python
from libreyolo import LIBREYOLO

# Load a model (automatically detects v8 vs v11)
model = LIBREYOLO(model_path="weights/libreyolo8n.pt", size="n")

# Run inference
detections = model(image="media/test_image_1_creative_commons.jpg", save=True)

# Access results
for detection in detections:
    print(f"Detected {detection.class_name} with confidence {detection.confidence:.2f}")
```

### Training

You can train models using the training module.

```python
from libreyolo import LIBREYOLO8

model = LIBREYOLO8(model_path="libreyolo8n.pt", size="n")
detections = model(image="image.jpg", save=True)
```

## Feature Map Visualization

Save feature maps from different layers of the model for visualization and analysis.

### Usage

```python
# Save all layers
model = LIBREYOLO8(model_path="weights/libreyolo8n.pt", size="n", save_feature_maps=True)

# Save specific layers only
model = LIBREYOLO8(model_path="weights/libreyolo8n.pt", size="n", 
                   save_feature_maps=["backbone_p1", "backbone_c2f2_P3", "neck_c2f21"])

# Get list of available layers
print(model.get_available_layer_names())
```

Feature maps are saved to `runs/feature_maps/` directory.

## Documentation

For more detailed information, check out our documentation:

- [User Guide](docs/user_guide.md): Comprehensive guide on how to use Libre YOLO.
- [Fine-Tuning Guide](docs/fine_tuning.md): Instructions on how to fine-tune models on your custom datasets.
- [Model Layers Reference](docs/model_layers.md): detailed list of available layers for feature map extraction.

## License

- **Code:** [MIT License](LICENSE)
- **Weights:** Use of converted weights must comply with their original licenses (typically AGPL-3.0 for Ultralytics models). See `weights/LICENSE_NOTICE.txt` for details.


## PossibleRoadmap
- [ ] onnx inference
- [ ] Add tiled inference
- [x] YOLOv8 Support
- [x] YOLOv11 Support
- [x] Training Engine
- [x] Weight Conversion Tools
- [ ] Fine-tuning with custom datasets
- [ ] Publish to PyPI
- [ ] Export formats (ONNX, TensorRT)
- [ ] Other models support (YOLOv10, YOLOv12, etc.)
- [ ] Feature Maps Visualization
- [ ] Automated testing (when a commit is pushed)
- [ ] Add explainability techniques
- [ ] Cocoeval to measure how good the model is
- [ ] CLI Tool
- [ ] Batch Processing
- [ ] Export Formats
- [ ] Video Processing
- [ ] Model Validation/Testing
- [ ] Training Support
- [ ] Model Quantization
- [ ] ONNX/TensorRT Export
- [ ] Web API / Server
- [ ] Documentation & Examples
- [ ] GPU memory optimization (gradient checkpointing)
- [ ] Multi-threaded batch processing
- [ ] Async inference support
- [ ] Model pruning (remove unnecessary weights)
- [ ] Knowledge distillation (smaller models)
- [ ] Mixed precision inference (FP16/BF16)
- [ ] TensorRT optimization pipeline
- [ ] CoreML export for Apple devices
- [ ] OpenVINO export for Intel hardware
- [ ] Model caching for faster repeated loads
- [ ] Real-time webcam detection
- [ ] Stream processing (RTSP/HTTP streams)
- [ ] Object tracking (track IDs across frames)
- [ ] Custom class names support
- [ ] Region of Interest (ROI) detection
- [ ] Confidence score calibration
- [ ] Multi-model ensemble inference
- [ ] Temporal smoothing for video
- [ ] Object counting per class
- [ ] Detection filtering by area/size
- [ ] YOLO format dataset export
- [ ] COCO format dataset export
- [ ] Pascal VOC format export
- [ ] LabelImg format compatibility
- [ ] CSV export with metadata
- [ ] JSON-LD export for structured data
- [ ] Database integration (SQLite/PostgreSQL)
- [ ] Image augmentation pipeline
- [ ] Dataset validation tools
- [ ] Annotation format converters
- [ ] Type hints throughout codebase
- [ ] Configuration file support (YAML/TOML)
- [ ] Progress bars for batch processing
- [ ] Logging system with levels
- [ ] Error handling improvements
- [ ] Model versioning system
- [ ] Checkpoint management
- [ ] Pre-commit hooks setup
- [ ] Code formatting (black/isort)
- [ ] Linting setup (pylint/flake8)
- [ ] Docker container image
- [ ] PyPI package publication
- [ ] Conda package support
- [ ] GitHub Actions CI/CD
- [ ] Pre-built Docker images
- [ ] Kubernetes deployment examples
- [ ] AWS Lambda function template
- [ ] Google Cloud Function template
- [ ] Azure Function template
- [ ] Edge device deployment guide



### Available Layers

#### LIBREYOLO8

| Layer | Description |
|-------|-------------|
| `backbone_p1` | First convolution |
| `backbone_p2` | Second convolution |
| `backbone_c2f1` | First C2F block |
| `backbone_p3` | Third convolution |
| `backbone_c2f2_P3` | C2F at P3 (Stride 8) |
| `backbone_p4` | Fourth convolution |
| `backbone_c2f3_P4` | C2F at P4 (Stride 16) |
| `backbone_p5` | Fifth convolution |
| `backbone_c2f4` | Fourth C2F block |
| `backbone_sppf_P5` | SPPF at P5 (Stride 32) |
| `neck_c2f21` | Neck C2F block 1 |
| `neck_c2f11` | Neck C2F block 2 |
| `neck_c2f12` | Neck C2F block 3 |
| `neck_c2f22` | Neck C2F block 4 |
| `head8_conv11` | Head8 box conv |
| `head8_conv21` | Head8 class conv |
| `head16_conv11` | Head16 box conv |
| `head16_conv21` | Head16 class conv |
| `head32_conv11` | Head32 box conv |
| `head32_conv21` | Head32 class conv |

#### LIBREYOLO11

| Layer | Description |
|-------|-------------|
| `backbone_p1` | First convolution |
| `backbone_p2` | Second convolution |
| `backbone_c2f1` | First C3k2 block |
| `backbone_p3` | Third convolution |
| `backbone_c2f2_P3` | C3k2 at P3 (Stride 8) |
| `backbone_p4` | Fourth convolution |
| `backbone_c2f3_P4` | C3k2 at P4 (Stride 16) |
| `backbone_p5` | Fifth convolution |
| `backbone_c2f4` | Fourth C3k2 block |
| `backbone_sppf` | SPPF block |
| `backbone_c2psa_P5` | C2PSA at P5 (Stride 32) |
| `neck_c2f21` | Neck C3k2 block 1 |
| `neck_c2f11` | Neck C3k2 block 2 |
| `neck_c2f12` | Neck C3k2 block 3 |
| `neck_c2f22` | Neck C3k2 block 4 |
| `head8_conv11` | Head8 box conv |
| `head8_conv21` | Head8 class conv |
| `head16_conv11` | Head16 box conv |
| `head16_conv21` | Head16 class conv |
| `head32_conv11` | Head32 box conv |
| `head32_conv21` | Head32 class conv |
