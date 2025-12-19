# Libre YOLO

```bash
pip install -e .
```

## Convert weights

```bash
python weights/convert_yolo8_weights.py --source yolov8n.pt --output libreyolo8n.pt
```

(If error: `pip install <legacy-converter-dependency>`)

## Use

```python
from libreyolo import LIBREYOLO8

model = LIBREYOLO8(model_path="libreyolo8n.pt", size="n")
detections = model(image="image.jpg", save=True)
```

## Possible Roadmap
These are some possible ideas to implement, although at this early stage the future direction is extremely unclear:
- [ ] Cocoeval to measure how good the model is
- [ ] YOLOv4 support (LIBREYOLO4 class)
- [ ] YOLOv5 support (LIBREYOLO5 class)
- [ ] YOLOv6 support (LIBREYOLO6 class)
- [ ] YOLOv7 support (LIBREYOLO7 class)
- [ ] YOLOv8 support (LIBREYOLO8 class)
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
