# LibreYOLO

[English](README.md) | [简体中文](README.zh-CN.md)

> ⭐ **Support LibreYOLO.** The best way to help is to **star the repo**. Feel free to [open an issue](https://github.com/LibreYOLO/libreyolo/issues/new) if you encounter problems or have suggestions, and code contributions are very welcome (see [CONTRIBUTING.md](CONTRIBUTING.md)). We are also looking for sponsors to donate GPU resources to the project. If you or your company can help, please [reach out on LinkedIn](https://www.linkedin.com/in/xuban-ceccon).

[![Documentation](https://img.shields.io/badge/docs-libreyolo.com-blue)](https://www.libreyolo.com/docs)
[![PyPI](https://img.shields.io/pypi/v/libreyolo)](https://pypi.org/project/libreyolo/)
[![PyPI Downloads](https://static.pepy.tech/badge/libreyolo)](https://pepy.tech/projects/libreyolo)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-LibreYOLO-yellow)](https://huggingface.co/LibreYOLO)
[![Benchmarks](https://img.shields.io/badge/benchmarks-visionanalysis.org-purple)](https://www.visionanalysis.org/)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-LibreYOLO-blue?logo=linkedin)](https://www.linkedin.com/company/libreyolo/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

MIT-licensed computer vision library with inference and training support for a variety of models. It provides a familiar high-level Python and CLI interface and reads common YOLO-format datasets, so existing workflows port over with minimal changes.

![LibreYOLO Detection Example](libreyolo/assets/parkour_result.jpg)

## Installation & Quick start

```bash
pip install libreyolo
```

To install the latest `main` in editable mode (for development or to track unreleased changes):

```bash
git clone https://github.com/LibreYOLO/libreyolo.git
cd libreyolo
pip install -e .
```

For optional runtime and export dependencies such as ONNX Runtime, OpenVINO, TensorRT, NCNN, and RF-DETR, see the [full docs](https://www.libreyolo.com/docs).

```python
from libreyolo import LibreYOLO, SAMPLE_IMAGE

model = LibreYOLO("LibreYOLO9t.pt")
result = model(SAMPLE_IMAGE, save=True)
```

## Flagship models

LibreYOLO recommends these model families because they offer the best balance
and receive the heaviest testing:

- **YOLOv9** for CNN-based YOLO models.
- **RF-DETR** for transformer-based detection and segmentation.

## Compatibility

`✓` supported, `exp` experimental. Empty cells are not currently supported.

<table>
  <thead>
    <tr>
      <th rowspan="2">Model family</th>
      <th colspan="3">Inference</th>
      <th rowspan="2">Training</th>
      <th colspan="5">Export formats</th>
    </tr>
    <tr>
      <th>Detection</th>
      <th>Segmentation</th>
      <th>Pose</th>
      <th>ONNX</th>
      <th>TorchScript</th>
      <th>TensorRT</th>
      <th>OpenVINO</th>
      <th>NCNN</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>YOLOv9</td><td>✓</td><td></td><td></td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td></tr>
    <tr><td>RF-DETR</td><td>✓</td><td>✓</td><td></td><td>exp</td><td>✓</td><td></td><td>✓</td><td>✓</td><td></td></tr>
    <tr><td>YOLOX</td><td>✓</td><td></td><td></td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td></tr>
    <tr><td>YOLOv9-E2E</td><td>✓</td><td></td><td></td><td>✓</td><td></td><td></td><td></td><td></td><td></td></tr>
    <tr><td>YOLO-NAS</td><td>✓</td><td></td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td></tr>
    <tr><td>D-FINE</td><td>✓</td><td></td><td></td><td>exp</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td></td></tr>
    <tr><td>DEIM</td><td>✓</td><td></td><td></td><td>exp</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td></td></tr>
    <tr><td>DEIMv2</td><td>✓</td><td></td><td></td><td>exp</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td></td></tr>
    <tr><td>RT-DETR</td><td>✓</td><td></td><td></td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td></td></tr>
    <tr><td>RT-DETRv2</td><td>✓</td><td></td><td></td><td>exp</td><td></td><td></td><td></td><td></td><td></td></tr>
    <tr><td>RT-DETRv4</td><td>✓</td><td></td><td></td><td>exp</td><td></td><td></td><td></td><td></td><td></td></tr>
    <tr><td>PicoDet</td><td>✓</td><td></td><td></td><td>exp</td><td></td><td></td><td></td><td></td><td></td></tr>
    <tr><td>EC</td><td>✓</td><td>✓</td><td>✓</td><td>exp</td><td></td><td></td><td></td><td></td><td></td></tr>
  </tbody>
</table>

## License

- **Code:** MIT License
- **Weights:** Pre-trained weights may inherit licensing from the original source. Check the license in the specific HF repo of weights that you are interested in. LibreYOLO HF models always have a license.

## Releases

- **v1.1.0** (2026-04-27): New model families (YOLO-NAS, D-FINE, RT-DETR), instance segmentation, ByteTrack tracking, video inference, and a brand-new CLI. [See the release notes](https://github.com/LibreYOLO/libreyolo/releases/tag/v1.1.0).
