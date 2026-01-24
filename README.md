# LibreYOLO

[![Documentation](https://img.shields.io/badge/docs-docs.libreyolo.com-blue)](https://docs.libreyolo.com)
[![PyPI](https://img.shields.io/pypi/v/libreyolo)](https://pypi.org/project/libreyolo/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

MIT-licensed YOLO implementation supporting inference for YOLOv7, YOLOv8 (n, s, m, l, x), YOLOv9 (t, s, m, c), YOLOv11 (n, s, m, l, x), YOLOX (nano, tiny, s, m, l, x), YOLO-RD, and RT-DETR (s, ms, m, l, x).

Training support is under development

![LibreYOLO Detection Example](media/parkour_result.jpg)

## Installation

```bash
pip install libreyolo
```

## Quick Start

```python
from libreyolo import LIBREYOLO

# Weights auto-download from HuggingFace
model = LIBREYOLO(model_path="libreyolo8n.pt", size="n")
results = model(image="https://raw.githubusercontent.com/Libre-YOLO/libreyolo/main/media/parkour.jpg", save=True)

print(f"Detected {results['num_detections']} objects")
```

## Documentation

Full documentation at [docs.libreyolo.com](https://docs.libreyolo.com):

## License

- **Code:** MIT License
- **Weights:** Pre-trained weights may inherit licensing from the original source
