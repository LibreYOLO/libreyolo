# LibreYOLO

[English](README.md) | [简体中文](README.zh-CN.md)

> ⭐ **Support LibreYOLO.** The best way to help is to **star the repo**. Feel free to [open an issue](https://github.com/LibreYOLO/libreyolo/issues/new) if you encounter problems or have suggestions, and code contributions are very welcome (see [CONTRIBUTING.md](CONTRIBUTING.md)).

[![Documentation](https://img.shields.io/badge/docs-libreyolo.com-blue)](https://www.libreyolo.com/docs)
[![PyPI](https://img.shields.io/pypi/v/libreyolo)](https://pypi.org/project/libreyolo/)
[![PyPI Downloads](https://static.pepy.tech/badge/libreyolo)](https://pepy.tech/projects/libreyolo)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-LibreYOLO-yellow)](https://huggingface.co/LibreYOLO)
[![Benchmarks](https://img.shields.io/badge/benchmarks-visionanalysis.org-purple)](https://www.visionanalysis.org/)
[![Greptile: The War on Bugs](https://www.greptile.com/badge.svg)](https://www.greptile.com/?utm_source=oss_badge&utm_medium=readme&utm_campaign=greptile_for_open_source)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-LibreYOLO-blue?logo=linkedin)](https://www.linkedin.com/company/libreyolo/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**An MIT-licensed computer vision library.** Detection, segmentation, pose,
depth, OCR and a dozen more tasks behind one small API, with training and
export included rather than sold separately. Reads common YOLO-format
datasets, so existing workflows port over with minimal changes.

![LibreYOLO Detection Example](libreyolo/assets/parkour_result.jpg)

## Install

```bash
pip install libreyolo
```

```python
from libreyolo import LibreYOLO, SAMPLE_IMAGE

model = LibreYOLO("LibreYOLO9t.pt")
result = model(SAMPLE_IMAGE, save=True)
```

<details>
<summary><b>Optional extras</b></summary>

<br>

The base install covers YOLOv9 and the other core detectors, training, and
inference. Add an extra when you need a heavier family or an export backend.
Comma-separate to combine, for example `pip install "libreyolo[rfdetr,hub-kernels]"`.

| Group | Extras |
| --- | --- |
| Export | `onnx`, `tensorrt`, `openvino`, `coreml`, `coreai`, `tflite` (alias `litert`), `ncnn`, `mnn`, `paddle`, `executorch` |
| Serving | `triton` |
| Models | `rfdetr`, `vlm`, `sam`, `openvocab`, `clip`, `siglip2`, `eomt`, `midas`, `modus`, `sensenova`, `gaze` |
| Training | `lora`, `plots`, `tensorboard`, `mlflow`, `wandb`, `comet`, `clearml`, `neptune`, `dvclive` |
| Speed | `fast-eval`, `hub-kernels` |
| Sources | `stream` |
| Everything | `pip install "libreyolo[all]"` |

`executorch`, `coreai` and `neptune` are deliberately left out of `all`: they
pin torch or protobuf in ways that would drag the rest of the environment with
them. Full list and per-backend notes in the
[install guide](https://www.libreyolo.com/docs/install).

</details>

<details>
<summary><b>Install from source</b></summary>

<br>

```bash
git clone https://github.com/LibreYOLO/libreyolo.git
cd libreyolo
pip install -e .
```

A plain clone checks out `release`, the stable branch matching the published
package. For unreleased work, `git checkout dev`.

</details>

## One API, seventeen tasks

The same three lines run every task. Only the checkpoint changes.

```python
from libreyolo import LibreYOLO

LibreYOLO("LibreYOLO9t.pt")("street.jpg", save=True)             # detection
LibreYOLO("LibreDeepLabv3mv3-sem.pt")("street.jpg", save=True)   # semantic segmentation
LibreYOLO("LibreHRNetw32-pose.pt")("street.jpg", save=True)      # pose
LibreYOLO("LibreMiDaSs-depth.pt")("street.jpg", save=True)       # depth
LibreYOLO("LibreFeyNobgl-matte.pt")("portrait.jpg", save=True)   # background removal
LibreYOLO("LibreRTDETRv2n-obb.pt")("aerial.jpg", save=True)      # oriented boxes
```

Sources are not just files. Point it at a webcam, an RTSP stream, a video, a
directory, a YouTube URL or your screen:

```bash
libreyolo predict --model yolo9-t --source 0 --show          # webcam
libreyolo predict --model yolo9-t --source rtsp://camera/1   # network camera
libreyolo predict --model yolo9-t --source screen            # screen capture
```

## What ships

| Task | Models |
| --- | --- |
| **Detection** | YOLOv9, RF-DETR, YOLOX, YOLO-NAS, D-FINE, DEIM, RT-DETR v1/v2/v4, RTMDet, PicoDet, PP-YOLOE, YOLOv7, EfficientDet, and the classics: DETR, Deformable DETR, DINO-DETR, LW-DETR, Faster R-CNN, RetinaNet, SSD, FCOS, CenterNet |
| **Tiny objects** | TinyFormer, Dome-DETR (aerial, drone, remote sensing) |
| **Instance segmentation** | RF-DETR, RTMDet, D-FINE, Mask R-CNN |
| **Promptable segmentation** | SAM, SAM 2, SAM 3, MobileSAM, EdgeTAM, PicoSAM3 |
| **Semantic segmentation** | SegFormer, PIDNet, PP-LiteSeg, U-Net, DeepLabv3, FCN, LingBot-Vision, DINOv2, EoMT |
| **Panoptic segmentation** | EoMT |
| **Pose** | RF-DETR, YOLO-NAS, HRNet, DEKR, EC |
| **Oriented boxes** | RF-DETR, RT-DETRv2, YOLO-NAS-R |
| **Classification** | MobileNetV4, ConvNeXt, EfficientNetV2, ResNet, ViT, Swin, DeiT, VGG, AlexNet, CLIP, SigLIP2, DINOv2 |
| **Depth** | Depth Anything 3, Depth Anything V2, ZipDepth, MiDaS |
| **Surface normals** | MoGe-2 |
| **Edges** | DexiNed, TEED |
| **Embeddings** | LibreFaceEmbedder, CLIP, SigLIP2, Perception Encoder (image, text, whole-video; also zero-shot classify), DINOv2 |
| **Video embeddings** | V-JEPA 2 (clip-level embedding, plus video classification with a trainable attentive probe) |
| **Body mesh** | SAM 3D Body |
| **Restoration** | DDColor, HVI-CIDNet, LaMa, NAFNet, QuickSRNet, Real-ESRGAN, SwinIR |
| **Background removal** | BiRefNet, FeyNobg, BEN2, ViTMatte |
| **OCR** | PP-OCR |
| **Point detection** | FOMO, LocateAnything |
| **Gaze** | L2CS |
| **Open vocabulary and VLMs** | Grounding DINO, OWLv2, OmDet-Turbo, OV-DEIM, Florence-2, Kosmos-2, Qwen3-VL, InternVL3, LFM2-VL, North Micro Vision, SmolVLM2, Gemma 4, Moondream, MODUS |

Per-family sizes, checkpoints and parity evidence live in the
[model reference](https://www.libreyolo.com/docs/models).

## Train

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLO9t.pt")
model.train(data="dataset.yaml", epochs=100, imgsz=640)
```

```bash
libreyolo train --model yolo9-t --data dataset.yaml --epochs 100
```

Multi-GPU, LoRA, layer freezing, distillation, from-scratch training, and
TensorBoard, MLflow, Weights & Biases, Comet, ClearML, Neptune and DVCLive
logging are all supported. See the
[training guide](https://www.libreyolo.com/docs/train).

## Export and deploy

Twelve formats: ONNX, TorchScript, TensorRT, OpenVINO, CoreML, Core AI, TFLite
(LiteRT), NCNN, MNN, RKNN, Paddle and ExecuTorch. Plus NVIDIA Triton serving
and DeepStream config generation.

```bash
libreyolo export --model yolo9-t --format onnx
```

Support varies by family and task, see the
[export matrix](https://www.libreyolo.com/docs/reference/export-matrix).

## Documentation

- [Docs](https://www.libreyolo.com/docs) covers install, tasks, models, training, prediction, export and the CLI
- [Benchmarks](https://www.visionanalysis.org/) for independent numbers
- [CHANGELOG.md](CHANGELOG.md) for what changed

## License

- **Code:** MIT License.
- **Weights:** pre-trained weights may inherit licensing from their original
  source, and not all of them are permissive. Check the license on the
  specific Hugging Face repo before you use one commercially. Every LibreYOLO
  Hugging Face model states its license.
