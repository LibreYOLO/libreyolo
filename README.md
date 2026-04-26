# LibreYOLO

[![Documentation](https://img.shields.io/badge/docs-libreyolo.com-blue)](https://www.libreyolo.com/docs)
[![PyPI](https://img.shields.io/pypi/v/libreyolo)](https://pypi.org/project/libreyolo/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

MIT-licensed object detection library with training and inference support across YOLOv9 (`t`, `s`, `m`, `c`), YOLOX (`n`, `t`, `s`, `m`, `l`, `x`), YOLO-NAS (`s`, `m`, `l`), RF-DETR (`n`, `s`, `m`, `l`), and D-FINE (`n`, `s`, `m`, `l`, `x`).

![LibreYOLO Detection Example](libreyolo/assets/parkour_result.jpg)

## Installation

```bash
pip install libreyolo
```

For optional runtime and export dependencies such as ONNX Runtime, OpenVINO, TensorRT, NCNN, and RF-DETR, see the full docs.

## Quick Start

```python
from libreyolo import LibreYOLO, SAMPLE_IMAGE

# Auto-detect family and size from the checkpoint name
model = LibreYOLO("LibreYOLOXs.pt")
result = model(SAMPLE_IMAGE, save=True)

print(f"Detected {len(result)} objects")
print(result.boxes.xyxy)
print(result.saved_path)
```

## Documentation

Full documentation at [libreyolo.com/docs](https://www.libreyolo.com/docs).

## License

- **Code:** MIT License
- **Weights:** Pre-trained weights may inherit licensing from the original source
