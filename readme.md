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

## License

MIT for code. AGPL-3.0 for weights (see `weights/LICENSE_NOTICE.txt`).
