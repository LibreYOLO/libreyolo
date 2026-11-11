<div align="center">

<img src="docs/readme/banner.png" alt="LibreYOLO，采用 MIT 许可证的计算机视觉库。pip install libreyolo" width="100%">

<br>

LibreYOLO 是一个采用 MIT 许可证的计算机视觉库，用于训练、运行和导出计算机视觉模型。

[![PyPI](https://img.shields.io/pypi/v/libreyolo)](https://pypi.org/project/libreyolo/)
[![PyPI Downloads](https://static.pepy.tech/badge/libreyolo)](https://pepy.tech/projects/libreyolo)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![中文文档](https://img.shields.io/badge/docs-%E4%B8%AD%E6%96%87%E6%96%87%E6%A1%A3-blue)](https://www.libreyolo.com/zh/docs)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-LibreYOLO-yellow)](https://huggingface.co/LibreYOLO)
[![Greptile: The War on Bugs](https://www.greptile.com/badge.svg)](https://www.greptile.com/?utm_source=oss_badge&utm_medium=readme&utm_campaign=greptile_for_open_source)

[English](README.md) | [简体中文](README.zh-CN.md)

</div>

> [!IMPORTANT]
> **致中国的 LibreYOLO 用户：我需要你的帮助。**
>
> 你好！我身在西方国家，不太了解你们使用时会遇到哪些限制。我知道 Hugging Face 在中国无法访问，也许把模型权重镜像到 hf-mirror.com、modelscope.ai 或 modelscope.cn 会对你有帮助，又或者有更好的办法。
>
> 如果你知道怎样让 LibreYOLO 在中国更好用，请[开一个 issue](https://github.com/LibreYOLO/libreyolo/issues/new?title=%5B%E4%B8%AD%E5%9B%BD%E7%94%A8%E6%88%B7%5D%20%E5%A6%82%E4%BD%95%E8%AE%A9%20LibreYOLO%20%E5%9C%A8%E4%B8%AD%E5%9B%BD%E6%9B%B4%E5%A5%BD%E7%94%A8&body=%E4%BD%A0%E6%89%80%E5%9C%A8%E7%9A%84%E5%9C%B0%E5%8C%BA%20/%20%E7%BD%91%E7%BB%9C%E7%8E%AF%E5%A2%83%EF%BC%9A%0A%0A%E4%BD%BF%E7%94%A8%20LibreYOLO%20%E6%97%B6%E9%81%87%E5%88%B0%E7%9A%84%E9%97%AE%E9%A2%98%EF%BC%9A%0A%0A%E4%BD%A0%E5%B8%8C%E6%9C%9B%E7%9A%84%E6%96%B9%E6%A1%88%EF%BC%88%E4%BE%8B%E5%A6%82%20hf-mirror.com%E3%80%81ModelScope%20%E9%95%9C%E5%83%8F%EF%BC%8C%E6%88%96%E5%85%B6%E4%BB%96%E5%8A%9E%E6%B3%95%EF%BC%89%EF%BC%9A%0A)，告诉我们怎样改进。非常感谢！
>
> Xuban，LibreYOLO 维护者

> **注意：** 本中文 README 由 AI 翻译，可能包含不准确或不自然的表述。请以英文 README 为准。

> ⭐ **支持 LibreYOLO。** 帮助项目最好的方式是给仓库 **star**。如果你遇到问题或有建议，欢迎[打开 issue](https://github.com/LibreYOLO/libreyolo/issues/new)；也欢迎代码贡献（见 [CONTRIBUTING.md](CONTRIBUTING.md)）。如果你愿意在经济上支持项目，非常欢迎：请查看[赞助计划](SPONSORS.md)。

<p align="center"><img src="libreyolo/assets/guggenheim-bilbao_result.jpg" width="520" alt="LibreYOLO 在毕尔巴鄂古根海姆图片上的检测结果"></p>

## 安装

```bash
pip install libreyolo -i https://pypi.tuna.tsinghua.edu.cn/simple
export HF_ENDPOINT=https://hf-mirror.com
```

第一行通过清华大学 PyPI 镜像安装；第二行让权重从
[hf-mirror.com](https://hf-mirror.com) 下载，这是社区维护的镜像，并非 Hugging Face
或 LibreYOLO 官方运营。Windows 下改用 `set HF_ENDPOINT=https://hf-mirror.com`（cmd）
或 `$env:HF_ENDPOINT="https://hf-mirror.com"`（PowerShell）。

所有权重下载都遵循 `HF_ENDPOINT` 需要 libreyolo 1.8 或从源码安装（`dev` 分支）；
在 1.7.4 中，`LibreYOLO("LibreYOLO9t.pt")` 这类按文件名自动下载的权重仍会直连
huggingface.co。此时可以手动下载权重，例如
`https://hf-mirror.com/LibreYOLO/LibreYOLO9t/resolve/main/LibreYOLO9t.pt`，
然后传入本地路径：`LibreYOLO("path/to/LibreYOLO9t.pt")`。

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
| 模型 | `rfdetr`、`vlm`、`sam`、`openvocab`、`clip`、`siglip2`、`eomt`、`midas`、`modus`、`sensenova`、`gaze`、`gen` |
| 训练 | `lora`、`plots`、`tensorboard`、`mlflow`、`wandb`、`comet`、`clearml`、`neptune`、`dvclive` |
| 提速 | `fast-eval`、`hub-kernels` |
| 输入源 | `stream` |
| 全部 | `pip install "libreyolo[all]"` |

`executorch`、`coreai`、`paddle` 和 `neptune` 被有意排除在 `all` 之外：它们对
torch、onnx 或 protobuf 的版本锁定会拖累环境中的其他依赖。`fast-eval` 和
`hub-kernels` 也不在 `all` 中。完整列表和各后端说明见
[安装指南](https://www.libreyolo.com/zh/docs/install)。

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

## 训练

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLO9t.pt")
model.train(data="dataset.yaml", epochs=100, imgsz=640)
```

```bash
libreyolo train --model yolo9-t --data dataset.yaml --epochs 100
```

多 GPU、LoRA、层冻结、蒸馏、从零训练、断点续训、混合精度、早停和日志记录，
详见[训练指南](https://www.libreyolo.com/zh/docs/train)。

## 推理

```python
from libreyolo import LibreYOLO, SAMPLE_IMAGE

model = LibreYOLO("LibreYOLO9t.pt")
result = model(SAMPLE_IMAGE, save=True)
```

可以直接用 Hugging Face 仓库名加载任意模型：`LibreYOLO("LibreYOLO/LibreRFDETRn")`。

输入源不只是文件。它还可以是摄像头、RTSP 流、视频、目录、YouTube 链接或你的屏幕：

```bash
libreyolo predict --model yolo9-t --source 0 --show          # 摄像头
libreyolo predict --model yolo9-t --source rtsp://camera/1   # 网络摄像头
libreyolo predict --model yolo9-t --source screen            # 屏幕捕获
```

输入源、结果、阈值和性能，详见[推理指南](https://www.libreyolo.com/zh/docs/predict)。

## 导出

十二种格式：ONNX、TorchScript、TensorRT、OpenVINO、CoreML、Core AI、
TFLite（LiteRT）、NCNN、MNN、RKNN、Paddle 和 ExecuTorch。另支持 NVIDIA
Triton 服务和 DeepStream 配置生成。

```bash
libreyolo export --model yolo9-t --format onnx
```

各格式说明见[导出指南](https://www.libreyolo.com/zh/docs/export)。支持范围因模型系列和任务而异，见[导出矩阵](https://www.libreyolo.com/zh/docs/reference/export-matrix)。

## 文档

- [文档](https://www.libreyolo.com/zh/docs)涵盖安装、任务、模型、训练、推理、导出和 CLI
- [基准测试](https://www.visionanalysis.org/)发布在 Vision Analysis 上
- [CHANGELOG.md](CHANGELOG.md) 记录版本变更

## 完整模型列表

LibreYOLO 按任务组织，例如检测、分割或分类。每个任务包含多个模型，每个模型又有
多个变体。例如 `LibreYOLO9t` 的任务是检测，模型是 YOLO9，变体是 `t`，即 tiny。
点击模型名可打开对应的文档页。

<details>
<summary><b>全部任务与模型</b></summary>

<br>

| 任务 | 模型 |
| --- | --- |
| **检测** | [YOLOv9](https://www.libreyolo.com/zh/docs/models/yolov9)、[RF-DETR](https://www.libreyolo.com/zh/docs/models/rf-detr)、[YOLOX](https://www.libreyolo.com/zh/docs/models/yolox)、[YOLO-NAS](https://www.libreyolo.com/zh/docs/models/yolo-nas)、[D-FINE](https://www.libreyolo.com/zh/docs/models/d-fine)、[DEIM](https://www.libreyolo.com/zh/docs/models/deim)、[RT-DETR v1/v2/v4](https://www.libreyolo.com/zh/docs/models/rt-detr)、[RTMDet](https://www.libreyolo.com/zh/docs/models/rtmdet)、[PicoDet](https://www.libreyolo.com/zh/docs/models/picodet)、[PP-YOLOE](https://www.libreyolo.com/zh/docs/models/ppyoloe)、[YOLOv7](https://www.libreyolo.com/zh/docs/models/yolov7)、[EfficientDet](https://www.libreyolo.com/zh/docs/models/efficientdet)，以及经典模型：[DETR](https://www.libreyolo.com/zh/docs/models/detr)、[Deformable DETR](https://www.libreyolo.com/zh/docs/models/deformable-detr)、[DINO-DETR](https://www.libreyolo.com/zh/docs/models/dino-detr)、[LW-DETR](https://www.libreyolo.com/zh/docs/models/lw-detr)、[Faster R-CNN](https://www.libreyolo.com/zh/docs/models/faster-rcnn)、[RetinaNet](https://www.libreyolo.com/zh/docs/models/retinanet)、[SSD](https://www.libreyolo.com/zh/docs/models/ssd)、[FCOS](https://www.libreyolo.com/zh/docs/models/fcos)、[CenterNet](https://www.libreyolo.com/zh/docs/models/centernet) |
| **微小目标** | [TinyFormer](https://www.libreyolo.com/zh/docs/models/tinyformer)、[Dome-DETR](https://www.libreyolo.com/zh/docs/models/dome-detr)（航拍、无人机、遥感） |
| **实例分割** | [RF-DETR](https://www.libreyolo.com/zh/docs/models/rf-detr)、[RTMDet](https://www.libreyolo.com/zh/docs/models/rtmdet)、[D-FINE](https://www.libreyolo.com/zh/docs/models/d-fine)、[Mask R-CNN](https://www.libreyolo.com/zh/docs/models/mask-rcnn) |
| **可提示分割** | [SAM](https://www.libreyolo.com/zh/docs/models/sam)、[SAM 2](https://www.libreyolo.com/zh/docs/models/sam-2)、[SAM 3](https://www.libreyolo.com/zh/docs/models/sam-3)、[MobileSAM](https://www.libreyolo.com/zh/docs/models/mobilesam)、[EdgeTAM](https://www.libreyolo.com/zh/docs/models/edgetam)、[PicoSAM3](https://www.libreyolo.com/zh/docs/models/picosam3) |
| **语义分割** | [SegFormer](https://www.libreyolo.com/zh/docs/models/segformer)、[PIDNet](https://www.libreyolo.com/zh/docs/models/pidnet)、[PP-LiteSeg](https://www.libreyolo.com/zh/docs/models/ppliteseg)、[U-Net](https://www.libreyolo.com/zh/docs/models/unet)、[DeepLabv3](https://www.libreyolo.com/zh/docs/models/deeplabv3)、[FCN](https://www.libreyolo.com/zh/docs/models/fcn)、[LingBot-Vision](https://www.libreyolo.com/zh/docs/models/lingbot-vision)、[DINOv2](https://www.libreyolo.com/zh/docs/models/dinov2)、[EoMT](https://www.libreyolo.com/zh/docs/models/eomt) |
| **全景分割** | [EoMT](https://www.libreyolo.com/zh/docs/models/eomt) |
| **姿态** | [RF-DETR](https://www.libreyolo.com/zh/docs/models/rf-detr)、[YOLO-NAS](https://www.libreyolo.com/zh/docs/models/yolo-nas)、[HRNet](https://www.libreyolo.com/zh/docs/models/hrnet)、[DEKR](https://www.libreyolo.com/zh/docs/models/dekr)、[EC](https://www.libreyolo.com/zh/docs/models/edgecrafter) |
| **旋转框（OBB）** | [RF-DETR](https://www.libreyolo.com/zh/docs/models/rf-detr)、[RT-DETRv2](https://www.libreyolo.com/zh/docs/models/rt-detr)、RiO-DETRv2、[YOLO-NAS-R](https://www.libreyolo.com/zh/docs/models/yolo-nas) |
| **三维检测** | [WildDet3D](https://www.libreyolo.com/zh/docs/models/wilddet3d)、[3D-MOOD](https://www.libreyolo.com/zh/docs/models/3d-mood)、[FCOS3D](https://www.libreyolo.com/zh/docs/models/fcos3d)、[DetAny3D](https://www.libreyolo.com/zh/docs/models/detany3d)（独立运行环境） |
| **分类** | [MobileNetV4](https://www.libreyolo.com/zh/docs/models/mobilenetv4)、[ConvNeXt](https://www.libreyolo.com/zh/docs/models/convnext)、[EfficientNetV2](https://www.libreyolo.com/zh/docs/models/efficientnetv2)、[ResNet](https://www.libreyolo.com/zh/docs/models/resnet)、[ViT](https://www.libreyolo.com/zh/docs/models/vit)、[Swin](https://www.libreyolo.com/zh/docs/models/swin)、[DeiT](https://www.libreyolo.com/zh/docs/models/deit)、[VGG](https://www.libreyolo.com/zh/docs/models/vgg)、[AlexNet](https://www.libreyolo.com/zh/docs/models/alexnet)、[CLIP](https://www.libreyolo.com/zh/docs/models/clip)、[SigLIP2](https://www.libreyolo.com/zh/docs/models/siglip2)、[DINOv2](https://www.libreyolo.com/zh/docs/models/dinov2) |
| **深度估计** | [Depth Anything 3](https://www.libreyolo.com/zh/docs/models/depth-anything-3)、[Depth Anything V2](https://www.libreyolo.com/zh/docs/models/depth-anything-v2)、Video Depth Anything、[ZipDepth](https://www.libreyolo.com/zh/docs/models/zipdepth)、[MiDaS](https://www.libreyolo.com/zh/docs/models/midas)、MoGe-3 |
| **表面法线** | MoGe-3、[MoGe-2](https://www.libreyolo.com/zh/docs/models/moge-2) |
| **深度、法线与反照率** | [Marigold V2](https://www.libreyolo.com/zh/docs/models/marigold-v2) |
| **边缘检测** | [DexiNed](https://www.libreyolo.com/zh/docs/models/dexined)、[TEED](https://www.libreyolo.com/zh/docs/models/teed) |
| **光流** | SEA-RAFT、NeuFlow v2、RAFT、WAFT、FlowSeek、DPFlow、UniMatch、MEMFOF |
| **嵌入** | [LibreFaceEmbedder](https://www.libreyolo.com/zh/docs/models/librefacerec)、[CLIP](https://www.libreyolo.com/zh/docs/models/clip)、[SigLIP2](https://www.libreyolo.com/zh/docs/models/siglip2)、[Perception Encoder](https://www.libreyolo.com/zh/docs/models/pe)（图像、文本、整段视频；亦支持零样本分类）、[DINOv2](https://www.libreyolo.com/zh/docs/models/dinov2) |
| **视频嵌入** | [V-JEPA 2](https://www.libreyolo.com/zh/docs/models/vjepa2)（片段嵌入及可训练分类探针）、[LeVJEPA](https://www.libreyolo.com/zh/docs/models/levjepa)（片段及分块嵌入；权重采用 CC BY-NC 4.0） |
| **人体网格** | [SAM 3D Body](https://www.libreyolo.com/zh/docs/models/sam-3d-body)、InstantHMR |
| **图像复原** | [DDColor](https://www.libreyolo.com/zh/docs/models/ddcolor)、[HVI-CIDNet](https://www.libreyolo.com/zh/docs/models/hvi-cidnet)、[LaMa](https://www.libreyolo.com/zh/docs/models/lama)、[NAFNet](https://www.libreyolo.com/zh/docs/models/nafnet)、[QuickSRNet](https://www.libreyolo.com/zh/docs/models/quicksrnet)、[Real-ESRGAN](https://www.libreyolo.com/zh/docs/models/real-esrgan)、[SwinIR](https://www.libreyolo.com/zh/docs/models/swinir) |
| **背景移除** | [BiRefNet](https://www.libreyolo.com/zh/docs/models/birefnet)、[FeyNobg](https://www.libreyolo.com/zh/docs/models/feynobg)、[BEN2](https://www.libreyolo.com/zh/docs/models/ben2)、[ViTMatte](https://www.libreyolo.com/zh/docs/models/vitmatte) |
| **OCR** | [PP-OCR](https://www.libreyolo.com/zh/docs/models/pp-ocrv5)、DeepSeek-OCR、DeepSeek-OCR-2、PaddleOCR-VL、GLM-OCR、Falcon OCR |
| **文档版面分析** | PP-DocLayoutV3（版面区域及阅读顺序） |
| **点检测** | [FOMO](https://www.libreyolo.com/zh/docs/models/fomo)、[LocateAnything](https://www.libreyolo.com/zh/docs/models/locate-anything) |
| **视线估计** | [L2CS](https://www.libreyolo.com/zh/docs/models/l2cs) |
| **开放词汇与 VLM** | [Grounding DINO](https://www.libreyolo.com/zh/docs/models/grounding-dino)、[OWLv2](https://www.libreyolo.com/zh/docs/models/owlv2)、[OmDet-Turbo](https://www.libreyolo.com/zh/docs/models/omdet-turbo)、[OV-DEIM](https://www.libreyolo.com/zh/docs/models/ov-deim)、[Florence-2](https://www.libreyolo.com/zh/docs/models/florence-2)、[Kosmos-2](https://www.libreyolo.com/zh/docs/models/kosmos-2)、[Qwen3-VL](https://www.libreyolo.com/zh/docs/models/qwen3-vl)、[InternVL3](https://www.libreyolo.com/zh/docs/models/internvl3)、[LFM2-VL](https://www.libreyolo.com/zh/docs/models/lfm2-vl)、[North Micro Vision](https://www.libreyolo.com/zh/docs/models/northmicrovision)、[SmolVLM2](https://www.libreyolo.com/zh/docs/models/smolvlm2)、[Gemma 4](https://www.libreyolo.com/zh/docs/models/gemma-4)、[Moondream](https://www.libreyolo.com/zh/docs/models/moondream)、[Molmo2](https://www.libreyolo.com/zh/docs/models/molmo2)、[WeVisOne-VL](https://www.libreyolo.com/zh/docs/models/wevisone-vl)、[MODUS](https://www.libreyolo.com/zh/docs/models/libremodus) |
| **机器人动作（VLA）** | [SmolVLA](https://www.libreyolo.com/zh/docs/models/librevla)、[ACT](https://www.libreyolo.com/zh/docs/models/librevla)、[Diffusion Policy](https://www.libreyolo.com/zh/docs/models/librevla) |
| **图像生成** | [FLUX.2 klein](https://www.libreyolo.com/zh/docs/models/flux2-klein)、[SANA-Sprint](https://www.libreyolo.com/zh/docs/models/sana-sprint)、PRX、PRXPixel |
| **3D 生成** | [Speridlabs Idol](https://www.libreyolo.com/zh/docs/models/speridlabs-idol)、TripoSplat、TRELLIS、TRELLIS.2、LGM、TripoSR、CRM、Direct3D-S2 |

</details>

各模型系列的尺寸、权重与一致性验证证据见[模型参考](https://www.libreyolo.com/zh/docs/models)。

## 关注我们

- [LinkedIn](https://www.linkedin.com/company/libreyolo/)
- [Reddit](https://www.reddit.com/r/LibreYOLO/)

## 贡献者

<a href="https://github.com/LibreYOLO/libreyolo/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=LibreYOLO/libreyolo" alt="LibreYOLO 贡献者">
</a>

## 赞助

LibreYOLO 在业余时间开发。为了更进一步，我们希望训练新模型，例如 LibreYOLO26，
以及为权重许可受限的模型系列重新训练 MIT 权重。这需要租用 GPU，为此我们设立了
LibreYOLO 赞助计划。详情见[这篇文章](https://www.libreyolo.com/zh/sponsors)。

[![Sponsor](https://img.shields.io/badge/sponsor-Open%20Collective-7fadf2?logo=opencollective)](https://opencollective.com/libreyolo)

**支持者。** 感谢所有捐助过的朋友：

<a href="https://github.com/picjul" title="Mattia Di Giusto"><img src="docs/readme/supporters/picjul.png" width="48" height="48" alt="Mattia Di Giusto"></a> <a href="https://github.com/Magellol" title="Thomas Lefebvre (ma.ge)"><img src="docs/readme/supporters/Magellol.png" width="48" height="48" alt="Thomas Lefebvre"></a>

## 许可证

- **代码：** MIT License。
- **权重：** 商用前请先查看具体 Hugging Face 仓库的许可证。每个 LibreYOLO Hugging Face 模型都会标明其许可证。
