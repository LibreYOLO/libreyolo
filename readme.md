DeepWiki automatically generated documentation --> https://deepwiki.com/Libre-YOLO/libreyolo
# Libre YOLO
```bash
pip install -e .
```

## Convert weights

```bash
```

(If error: `pip install <legacy-converter-dependency>`)

## Use

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

## Possible Roadmap
These are some possible ideas to implement, although at this early stage the future direction is extremely unclear:
- [ ] Publish the package in pypi. Automated with a Github pipeline
- [ ] Automated testing (when a commit is pushed)
- [ ] Add explainability techniques
- [ ] Make the activation maps available
- [ ] Cocoeval to measure how good the model is
- [ ] YOLOv4 support (LIBREYOLO4 class)
- [ ] YOLOv5 support (LIBREYOLO5 class)
- [ ] YOLOv6 support (LIBREYOLO6 class)
- [ ] YOLOv7 support (LIBREYOLO7 class)
- [ ] YOLOv8 support (LIBREYOLO8 class)
- [x] YOLOv8 support (LIBREYOLO8 class)
- [ ] YOLOv9 support (LIBREYOLO9 class)
- [ ] YOLOv10 support (LIBREYOLO10 class)
- [ ] YOLOv11 support (LIBREYOLO11 class)
- [ ] YOLOv12 support (LIBREYOLO12 class)
- [ ] YOLOX support (LIBREYOLOX class)
- [ ] YOLO-NAS support (LIBREYOLONAS class)
- [ ] YOLO-World support (LIBREYOLOWORLD class)
- [ ] YOLOR support (LIBREYOLOR class)
- [ ] YOLOF support (LIBREYOLOF class)
- [ ] PP-YOLO/PP-YOLOE support (LIBREPPYOLO class)
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

## License

MIT for code. AGPL-3.0 for weights (see `weights/LICENSE_NOTICE.txt`).
