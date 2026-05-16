# LibreYOLO

[English](README.md) | [简体中文](README.zh-CN.md)

> **注意：** 本中文 README 由 AI 翻译，可能包含不准确或不自然的表述。请以英文 README 为准。

> ⭐ **支持 LibreYOLO。** 帮助项目最好的方式是给仓库 **star**。如果你遇到问题或有建议，欢迎[打开 issue](https://github.com/LibreYOLO/libreyolo/issues/new)；也欢迎代码贡献（见 [CONTRIBUTING.md](CONTRIBUTING.md)）。我们也在寻找赞助方为项目捐赠 GPU 资源。如果你或你的公司可以提供帮助，请通过 [LinkedIn 联系我们](https://www.linkedin.com/in/xuban-ceccon)。

[![Documentation](https://img.shields.io/badge/docs-libreyolo.com-blue)](https://www.libreyolo.com/docs)
[![PyPI](https://img.shields.io/pypi/v/libreyolo)](https://pypi.org/project/libreyolo/)
[![PyPI Downloads](https://static.pepy.tech/badge/libreyolo)](https://pepy.tech/projects/libreyolo)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-LibreYOLO-yellow)](https://huggingface.co/LibreYOLO)
[![Benchmarks](https://img.shields.io/badge/benchmarks-visionanalysis.org-purple)](https://www.visionanalysis.org/)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-LibreYOLO-blue?logo=linkedin)](https://www.linkedin.com/company/libreyolo/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

LibreYOLO 是一个采用 MIT 许可证的计算机视觉库，支持多种模型的推理和训练。它使用与 Ultralytics 相同的 API：如果你使用过 Ultralytics，就已经知道如何使用 LibreYOLO，并且现有脚本可以直接运行。

![LibreYOLO 检测示例](libreyolo/assets/parkour_result.jpg)

## 安装与快速开始

```bash
pip install libreyolo
```

如需以可编辑模式安装最新的 `main` 分支（用于开发或跟踪尚未发布的改动）：

```bash
git clone https://github.com/LibreYOLO/libreyolo.git
cd libreyolo
pip install -e .
```

ONNX Runtime、OpenVINO、TensorRT、NCNN 和 RF-DETR 等可选运行时与导出依赖，请见[完整文档](https://www.libreyolo.com/docs)。

```python
from libreyolo import LibreYOLO, SAMPLE_IMAGE

model = LibreYOLO("LibreYOLO9t.pt")
result = model(SAMPLE_IMAGE, save=True)
```

## 兼容性

`✓` 表示支持，`exp` 表示实验性支持。空单元格表示当前不支持。

<table>
  <thead>
    <tr>
      <th rowspan="2">模型系列</th>
      <th colspan="3">推理</th>
      <th rowspan="2">训练</th>
      <th colspan="5">导出格式</th>
    </tr>
    <tr>
      <th>检测</th>
      <th>分割</th>
      <th>姿态</th>
      <th>ONNX</th>
      <th>TorchScript</th>
      <th>TensorRT</th>
      <th>OpenVINO</th>
      <th>NCNN</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>YOLOX</td><td>✓</td><td></td><td></td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td></tr>
    <tr><td>YOLOv9</td><td>✓</td><td></td><td></td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td></tr>
    <tr><td>YOLOv9-E2E</td><td>✓</td><td></td><td></td><td>✓</td><td></td><td></td><td></td><td></td><td></td></tr>
    <tr><td>YOLO-NAS</td><td>✓</td><td></td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td></tr>
    <tr><td>RF-DETR</td><td>✓</td><td>✓</td><td></td><td>exp</td><td>✓</td><td></td><td>✓</td><td>✓</td><td></td></tr>
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

## 许可证

- **代码：** MIT License
- **权重：** 预训练权重可能继承原始来源的许可证。请检查你感兴趣的具体 HF 权重仓库中的许可证。LibreYOLO HF 模型始终包含许可证。

## 发布

- **v1.1.0**（2026-04-27）：新增模型系列（YOLO-NAS、D-FINE、RT-DETR）、实例分割、ByteTrack 跟踪、视频推理和全新的 CLI。[查看发布说明](https://github.com/LibreYOLO/libreyolo/releases/tag/v1.1.0)。
