# VGG

- **LibreYOLO module:** `libreyolo/models/vgg/`, centralized postprocessing in
  `libreyolo/postprocess/vgg.py`, and converter at
  `weights/convert_vgg_weights.py`.
- **Upstream code:** `pytorch/vision` at commit
  `10f68dbd78b9aa5cab9328f3b2e99cfb0b608122`, file
  `torchvision/models/vgg.py`.
- **Code license:** BSD-3-Clause, Copyright (c) Soumith Chintala 2016. The full
  notice is reproduced in `libreyolo/models/vgg/NOTICE`.
- **Scope:** ImageNet-style classification inference and fixed-resolution
  export. Training and validation are intentionally excluded; `train()` raises
  `NotImplementedError`.

## Source checkpoints

The official files below were downloaded from PyTorch's model host and checked
by SHA-256. Conversion preserves every learned tensor and only adds LibreYOLO
checkpoint metadata. The files are not included in this repository.

| Size | LibreYOLO filename | Official file | Bytes | SHA-256 |
|---|---|---|---:|---|
| `16` | `LibreVGG16-cls.pt` | `vgg16-397923af.pth` | 553,433,881 | `397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0` |
| `19` | `LibreVGG19-cls.pt` | `vgg19-dcbb9e9d.pth` | 574,673,361 | `dcbb9e9dad569fff7a846263a77324fc34978fea2bfb039c012d710e1776ae44` |
| `16bn` | `LibreVGG16bn-cls.pt` | `vgg16_bn-6c64b313.pth` | 553,507,836 | `6c64b3138f2f4fcb3bcc4cafde11619c4f440eb1631787e93a682fd88305888a` |
| `19bn` | `LibreVGG19bn-cls.pt` | `vgg19_bn-c79401a0.pth` | 574,769,405 | `c79401a0cf3cb42714e4182f5868c7a6f4f4534f5df9e956e2bb2098de41cbb6` |

Official URLs are the filename above appended to
`https://download.pytorch.org/models/`. The selected variants are the
torchvision `IMAGENET1K_V1` weight enums. The feature-only VGG-16 checkpoint is
not converted because it has no released classifier head.

The metadata-wrapped artifacts prepared for the LibreYOLO mirrors are:

| LibreYOLO filename | Bytes | SHA-256 |
|---|---:|---|
| `LibreVGG16-cls.pt` | 553,463,039 | `90d84cd370681f5ef0145c34a7e0a134b48a0fc07d1e586e865da1997483f6d7` |
| `LibreVGG19-cls.pt` | 574,703,721 | `edf2f8856d791a119cd8f451d8eaaa2275ebb5999c65c147d631619a2c7d3873` |
| `LibreVGG16bn-cls.pt` | 553,546,527 | `d2ecae40cd5b73d7de8f91370086e98e2ea25d5ce38eab3f9380e0b193687c65` |
| `LibreVGG19bn-cls.pt` | 574,811,267 | `6cb1346e6f3cdf11647e32b94c7bfecec85e0f14f0bd8292289d236a4ddfea35` |

### Weight-license status

The torchvision source is BSD-3-Clause. The publisher does not attach a
separate license file to these four checkpoint objects, so the redistribution
basis is **BSD-3-Clause implied by the releasing project**, not an explicit
checkpoint-specific grant. Torchvision separately warns that pretrained models
may have terms derived from their training data and leaves use-case permission
to the user. ImageNet is a separate dataset and is not distributed by
LibreYOLO.

Every LibreYOLO weight repository therefore carries the verbatim torchvision
BSD-3-Clause license, an attribution notice, and that caveat. The model cards
must never describe the checkpoint license as publisher-confirmed.

The public mirrors are
[`16`](https://huggingface.co/LibreYOLO/LibreVGG16-cls),
[`19`](https://huggingface.co/LibreYOLO/LibreVGG19-cls),
[`16bn`](https://huggingface.co/LibreYOLO/LibreVGG16bn-cls), and
[`19bn`](https://huggingface.co/LibreYOLO/LibreVGG19bn-cls). Each repository
was verified public with exactly five files, a verbatim copy of the pinned BSD
license, and an LFS SHA-256 matching the converted-artifact table. All four are
members of the LibreYOLO Classification collection. Each bare canonical
filename was then downloaded into an empty directory, hash-checked, loaded on
CUDA, and used for a real prediction before the temporary downloads were
removed.

## Ported surface

| LibreYOLO surface | Pinned torchvision source |
|---|---|
| `models/vgg/nn.py` -- VGG graph, configurations, and initialization | `torchvision/models/vgg.py` |
| `models/vgg/model.py` -- family, task, and checkpoint contracts | LibreYOLO integration code |
| `models/base/classify_preprocess.py` (shared eval transform, `crop_pct` 0.875 bilinear at 224) -- resize, center crop, and ImageNet normalization | `VGG*_Weights.IMAGENET1K_V1.transforms()` contract |
| `postprocess/vgg.py` -- softmax probability payload | LibreYOLO integration code |

Module names retain the official `features.*`, `avgpool`, and
`classifier.{0,3,6}` layout, so all four official state dicts load with
`strict=True` and no key remapping. Variant detection uses convolution count,
batch-normalization buffers, and classifier dimensions rather than filename
guessing.

## Measured evidence

- Eager CUDA parity for `16`, `19`, `16bn`, and `19bn` against the pinned
  torchvision graph: `max_abs_diff == 0.0` for logits on every variant.
- The public preprocessing contract matches the official V1 weight preset:
  bilinear resize to short side 256, center crop 224, and ImageNet mean/std.
- A converted VGG-16 checkpoint strict-loads through `LibreYOLO`, predicts the
  bundled sample image on CUDA, returns 1,000 probabilities summing to one,
  and identifies ImageNet class 880 (`unicycle`) as top-1.
- Full trained VGG-16 FP32 export and backend reload pass probability cosine
  similarity above 0.999 with identical top-1 output for ONNX Runtime,
  TorchScript, OpenVINO, and TensorRT at fixed 224 and batch one.
