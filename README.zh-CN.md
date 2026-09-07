# LibreYOLO

[English](README.md) | [简体中文](README.zh-CN.md)

> **注意：** 本中文 README 由 AI 翻译，可能包含不准确或不自然的表述。请以英文 README 为准。

> ⭐ **支持 LibreYOLO。** 帮助项目最好的方式是给仓库 **star**。如果你遇到问题或有建议，欢迎[打开 issue](https://github.com/LibreYOLO/libreyolo/issues/new)；也欢迎代码贡献（见 [CONTRIBUTING.md](CONTRIBUTING.md)）。

[![Documentation](https://img.shields.io/badge/docs-libreyolo.com-blue)](https://www.libreyolo.com/docs)
[![PyPI](https://img.shields.io/pypi/v/libreyolo)](https://pypi.org/project/libreyolo/)
[![PyPI Downloads](https://static.pepy.tech/badge/libreyolo)](https://pepy.tech/projects/libreyolo)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-LibreYOLO-yellow)](https://huggingface.co/LibreYOLO)
[![Benchmarks](https://img.shields.io/badge/benchmarks-visionanalysis.org-purple)](https://www.visionanalysis.org/)
[![Greptile: The War on Bugs](https://www.greptile.com/badge.svg)](https://www.greptile.com/?utm_source=oss_badge&utm_medium=readme&utm_campaign=greptile_for_open_source)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-LibreYOLO-blue?logo=linkedin)](https://www.linkedin.com/company/libreyolo/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**一个采用 MIT 许可证的计算机视觉库。** 检测、分割、姿态、深度、OCR
等十几种任务共用一套精简 API，训练和导出内置提供，而非单独收费。
支持读取常见的 YOLO 格式数据集，因此现有工作流只需少量改动即可迁移。

![LibreYOLO 检测示例](libreyolo/assets/parkour_result.jpg)

## 安装

```bash
pip install libreyolo
```

```python
from libreyolo import LibreYOLO, SAMPLE_IMAGE

model = LibreYOLO("LibreYOLO9t.pt")
result = model(SAMPLE_IMAGE, save=True)
```

<details>
<summary><b>可选扩展依赖</b></summary>

<br>

基础安装已覆盖 YOLOv9 与其他核心检测器、训练和推理。当你需要更重的模型系列
或某个导出后端时，再安装对应的扩展依赖即可。多个扩展用逗号组合，例如
`pip install "libreyolo[rfdetr,hub-kernels]"`。

| 分组 | 扩展依赖 |
| --- | --- |
| 导出 | `onnx`、`tensorrt`、`openvino`、`coreml`、`coreai`、`tflite`（别名 `litert`）、`ncnn`、`mnn`、`paddle`、`executorch` |
| 服务 | `triton` |
| 模型 | `rfdetr`、`vlm`、`sam`、`openvocab`、`clip`、`siglip2`、`eomt`、`midas`、`modus`、`sensenova`、`gaze` |
| 训练 | `lora`、`plots`、`tensorboard`、`mlflow`、`wandb`、`comet`、`clearml`、`neptune`、`dvclive` |
| 提速 | `fast-eval`、`hub-kernels` |
| 输入源 | `stream` |
| 全部 | `pip install "libreyolo[all]"` |

`executorch`、`coreai` 和 `neptune` 被有意排除在 `all` 之外：它们对 torch 或
protobuf 的版本锁定会拖累环境中的其他依赖。完整列表和各后端说明见
[安装指南](https://www.libreyolo.com/docs/install)。

</details>

<details>
<summary><b>从源码安装</b></summary>

<br>

```bash
git clone https://github.com/LibreYOLO/libreyolo.git
cd libreyolo
pip install -e .
```

普通克隆会检出 `release` 分支，即与已发布包一致的稳定分支。如需获取尚未发布的
内容，请使用 `git checkout dev`。

</details>

## 一套 API，十七种任务

每种任务都是同样的三行代码，只有权重文件不同。

```python
from libreyolo import LibreYOLO

LibreYOLO("LibreYOLO9t.pt")("street.jpg", save=True)             # 检测
LibreYOLO("LibreDeepLabv3mv3-sem.pt")("street.jpg", save=True)   # 语义分割
LibreYOLO("LibreHRNetw32-pose.pt")("street.jpg", save=True)      # 姿态
LibreYOLO("LibreMiDaSs-depth.pt")("street.jpg", save=True)       # 深度
LibreYOLO("LibreFeyNobgl-matte.pt")("portrait.jpg", save=True)   # 背景移除
LibreYOLO("LibreRTDETRv2n-obb.pt")("aerial.jpg", save=True)      # 旋转框
```

输入源不只是文件。它还可以是摄像头、RTSP 流、视频、目录、YouTube 链接或你的屏幕：

```bash
libreyolo predict --model yolo9-t --source 0 --show          # 摄像头
libreyolo predict --model yolo9-t --source rtsp://camera/1   # 网络摄像头
libreyolo predict --model yolo9-t --source screen            # 屏幕捕获
```

## 内置内容

| 任务 | 模型 |
| --- | --- |
| **检测** | YOLOv9、RF-DETR、YOLOX、YOLO-NAS、D-FINE、DEIM、RT-DETR v1/v2/v4、RTMDet、PicoDet、PP-YOLOE、YOLOv7、EfficientDet，以及经典模型：DETR、Deformable DETR、DINO-DETR、LW-DETR、Faster R-CNN、RetinaNet、SSD、FCOS、CenterNet |
| **微小目标** | TinyFormer、Dome-DETR（航拍、无人机、遥感） |
| **实例分割** | RF-DETR、RTMDet、D-FINE、Mask R-CNN |
| **可提示分割** | SAM、SAM 2、SAM 3、MobileSAM、EdgeTAM、PicoSAM3 |
| **语义分割** | SegFormer、PIDNet、PP-LiteSeg、U-Net、DeepLabv3、FCN、LingBot-Vision、DINOv2、EoMT |
| **全景分割** | EoMT |
| **姿态** | RF-DETR、YOLO-NAS、HRNet、DEKR、EC |
| **旋转框（OBB）** | RF-DETR、RT-DETRv2、YOLO-NAS-R |
| **分类** | MobileNetV4、ConvNeXt、EfficientNetV2、ResNet、ViT、Swin、DeiT、VGG、AlexNet、CLIP、SigLIP2、DINOv2 |
| **深度估计** | Depth Anything 3、Depth Anything V2、ZipDepth、MiDaS |
| **表面法线** | MoGe-2 |
| **边缘检测** | DexiNed、TEED |
| **嵌入** | LibreFaceEmbedder、CLIP、SigLIP2、Perception Encoder（图像、文本、整段视频；亦支持零样本分类）、DINOv2 |
| **视频嵌入** | V-JEPA 2（片段级嵌入，另有可训练注意力探针的视频分类） |
| **人体网格** | SAM 3D Body |
| **图像复原** | DDColor、HVI-CIDNet、LaMa、NAFNet、QuickSRNet、Real-ESRGAN、SwinIR |
| **背景移除** | BiRefNet、FeyNobg、ViTMatte |
| **OCR** | PP-OCR |
| **点检测** | FOMO、LocateAnything |
| **视线估计** | L2CS |
| **开放词汇与 VLM** | Grounding DINO、OWLv2、OmDet-Turbo、OV-DEIM、Florence-2、Kosmos-2、Qwen3-VL、InternVL3、LFM2-VL、North Micro Vision、SmolVLM2、MODUS |

各模型系列的尺寸、权重与一致性验证证据见[模型参考](https://www.libreyolo.com/docs/models)。

## 训练

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLO9t.pt")
model.train(data="dataset.yaml", epochs=100, imgsz=640)
```

```bash
libreyolo train --model yolo9-t --data dataset.yaml --epochs 100
```

多 GPU、LoRA、层冻结、蒸馏、从零训练，以及 TensorBoard、MLflow、
Weights & Biases、Comet、ClearML、Neptune 和 DVCLive 日志记录均受支持。
详见[训练指南](https://www.libreyolo.com/docs/train)。

## 导出与部署

十二种格式：ONNX、TorchScript、TensorRT、OpenVINO、CoreML、Core AI、
TFLite（LiteRT）、NCNN、MNN、RKNN、Paddle 和 ExecuTorch。另支持 NVIDIA
Triton 服务和 DeepStream 配置生成。

```bash
libreyolo export --model yolo9-t --format onnx
```

支持范围因模型系列和任务而异，见[导出矩阵](https://www.libreyolo.com/docs/reference/export-matrix)。

## 文档

- [文档](https://www.libreyolo.com/docs)涵盖安装、任务、模型、训练、推理、导出和 CLI
- [基准测试](https://www.visionanalysis.org/)提供独立的第三方数据
- [CHANGELOG.md](CHANGELOG.md) 记录版本变更

## 许可证

- **代码：** MIT License。
- **权重：** 预训练权重可能继承原始来源的许可证，且并非全部为宽松许可。
  商用前请先查看具体 Hugging Face 仓库上的许可证。每个 LibreYOLO
  Hugging Face 模型都会标明其许可证。
