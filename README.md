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

## Export Backend Support

| Backend | Status | Notes |
|---------|--------|-------|
| ONNX | Supported | Release-blocking export path. Keep it in routine validation and release e2e runs. |
| TorchScript | Experimental | Useful compatibility target, but not a release gate. |
| TensorRT (`tensorrt` / `trt`) | Experimental | CUDA-specific path. Valuable coverage, but not required for every release run. |
| OpenVINO | Experimental | Runtime-specific path. Keep coverage available, but not release-blocking. |
| NCNN | Experimental | Highest maintenance overhead today. Safe to exclude from routine release runs. |

The e2e suite mirrors this policy with pytest markers: `supported_backend` for ONNX and `experimental_backend` for the other export backends.

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
