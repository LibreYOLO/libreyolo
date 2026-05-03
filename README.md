# LibreYOLO

> ⭐ **Support LibreYOLO.** The best way to help is to **[star the repo](https://github.com/LibreYOLO/libreyolo)**. Feel free to [open an issue](https://github.com/LibreYOLO/libreyolo/issues/new) if you encounter problems or have suggestions, and code contributions are very welcome (see [CONTRIBUTING.md](CONTRIBUTING.md)). We are also looking for sponsors to donate GPU resources to the project. If you or your company can help, please [reach out on LinkedIn](https://www.linkedin.com/in/xuban-ceccon).

> **v1.1 released!** New model families (YOLO-NAS, D-FINE, RT-DETR), instance segmentation, ByteTrack tracking, video inference, and a brand-new CLI. [See the release notes](https://github.com/LibreYOLO/libreyolo/releases/tag/v1.1.0).

[![Documentation](https://img.shields.io/badge/docs-libreyolo.com-blue)](https://www.libreyolo.com/docs)
[![PyPI](https://img.shields.io/pypi/v/libreyolo)](https://pypi.org/project/libreyolo/)
[![PyPI Downloads](https://static.pepy.tech/badge/libreyolo)](https://pepy.tech/projects/libreyolo)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

MIT-licensed object detection library with training and inference support across YOLOv9, YOLOX, YOLO-NAS, RF-DETR, D-FINE, and RT-DETR.

![LibreYOLO Detection Example](libreyolo/assets/parkour_result.jpg)

## Installation

```bash
pip install libreyolo
```

For optional runtime and export dependencies such as ONNX Runtime, OpenVINO, TensorRT, NCNN, and RF-DETR, see the full docs.

## Inference Backend and Export Support

### Format Status

| Format | Export | Runtime backend | Status | Precision | Notes |
|--------|--------|-----------------|--------|-----------|-------|
| ONNX | Yes | ONNX Runtime | Supported | FP32, FP16 | Release-blocking export path and the broadest tested runtime target. |
| TorchScript | Yes | PyTorch JIT | Experimental | FP32, FP16 | Useful compatibility target, but not a release gate. |
| TensorRT (`tensorrt` / `trt`) | Yes | TensorRT | Experimental | FP32, FP16, INT8 | CUDA-only path. INT8 requires calibration data. |
| OpenVINO | Yes | OpenVINO | Experimental | FP32, FP16, INT8 | Runtime-specific path with CPU-oriented deployment coverage. |
| NCNN | Yes | NCNN | Experimental | FP32, FP16 | Highest maintenance overhead today. No INT8 path, and some DETR-family models are not supported. |

The e2e suite mirrors this policy with pytest markers: `supported_backend` for ONNX and `experimental_backend` for the other export backends.

### Model Family Matrix

`✓` supported, `~` supported with caveats, `—` intentionally unsupported

| Model family | ONNX | TorchScript | TensorRT | OpenVINO | NCNN |
|--------------|------|-------------|----------|----------|------|
| YOLOX | ✓ | ✓ | ✓ | ✓ | ✓ |
| YOLOv9 | ✓ | ✓ | ✓ | ✓ | ✓ |
| YOLO-NAS | ✓ | ✓ | ✓ | ✓ | ✓ |
| RF-DETR | ✓ | ~ | ✓ | ✓ | ~ |
| D-FINE | ✓ | ✓ | ✓ | ✓ | — |
| RT-DETR | ✓ | ✓ | ✓ | ✓ | — |

Notes:
- RF-DETR TorchScript export exists, but tracing can still be brittle on some checkpoints and shapes.
- NCNN is intentionally blocked for D-FINE and RT-DETR because the runtime lacks required DETR query-selection ops.
- RF-DETR on NCNN is not blocked at export time, but current e2e coverage still tracks known runtime limitations.

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
