# LibreYOLO

[![Documentation](https://img.shields.io/badge/docs-libreyolo.com-blue)](https://www.libreyolo.com/docs)
[![PyPI](https://img.shields.io/pypi/v/libreyolo)](https://pypi.org/project/libreyolo/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

MIT-licensed YOLO implementation supporting inference for YOLOv9 (t, s, m, c), YOLOX (nano, tiny, s, m, l, x), and RF-DETR (nano, small, base, medium, large).

Training support is under development

![LibreYOLO Detection Example](media/parkour_result.jpg)

## Installation

```bash
pip install libreyolo
```

## Quick Start

```python
from libreyolo import LIBREYOLO

# Auto-detect model version and size
model = LIBREYOLO("libreyoloXs.pt")
results = model(image="https://raw.githubusercontent.com/Libre-YOLO/libreyolo/main/media/parkour.jpg", save=True)

print(f"Detected {results['num_detections']} objects")
```

## Documentation

Full documentation at [libreyolo.com/docs](https://www.libreyolo.com/docs):

## License

- **Code:** MIT License
- **Weights:** Pre-trained weights may inherit licensing from the original source
