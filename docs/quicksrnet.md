# QuickSRNet real-time super-resolution

LibreQuickSRNet provides native inference and paired PSNR/SSIM validation for
QuickSRNet Medium 2x. It is a compact, fully convolutional RGB upscaler built
for low-latency deployment. Prediction runs at the source resolution and
returns `Results.restored` at twice the input height and width, with
`Results.restore_scale == 2`.

| Size | Architecture | Scale | Parameters |
|---|---|---:|---:|
| `m2` | QuickSRNet Medium, 32 channels, 5 intermediate layers | 2x | 50,604 |

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreQuickSRNetm2-restore.pt")
result = model.predict("input.jpg")
result.save("upscaled.png")
```

Inputs are converted to RGB float tensors in `[0, 1]`. There is no resize or
padding requirement, so native PyTorch and dynamic ONNX inference accept any
positive height and width. ONNX supports dynamic spatial axes; TorchScript uses
a fixed export canvas. Both paths have native-versus-runtime pixel parity tests.
Training is not implemented in the initial family, but `model.val(...)` accepts
paired low-resolution and 2x high-resolution images and reports PSNR and SSIM.

Reference model-forward latency for the converted trained checkpoint on an
NVIDIA GeForce RTX 5070 Ti, PyTorch 2.11.0 + CUDA 12.8, batch 1, cuDNN
benchmarking enabled, 10 warmups and 30 timed iterations:

| Input to output | Precision | Median | p95 | Median throughput |
|---|---|---:|---:|---:|
| 360p to 720p | FP32 | 1.746 ms | 1.770 ms | 572.7 FPS |
| 360p to 720p | FP16 | 0.937 ms | 0.966 ms | 1066.7 FPS |
| 720p to 1440p | FP32 | 10.536 ms | 10.931 ms | 94.9 FPS |
| 720p to 1440p | FP16 | 5.437 ms | 5.855 ms | 183.9 FPS |

These are synchronized model-forward timings. Image decoding, host-to-device
transfer, and result conversion are excluded, so application throughput will
be lower.

The architecture is adapted from the BSD-3-Clause `quic/aimet-model-zoo`
implementation at commit
`1bd2bf5b17cdda9251437c444009b29e1a25054b`. The current Qualcomm AI Hub Models
integration was audited at commit
`16dbeb5e2805d4ada7218026de72e36878717d46`. The official Medium 2x checkpoint
was trained on DIV2K; conversion preserves its 14 learned tensors, removes
training-only optimizer and history objects, and adds LibreYOLO checkpoint
metadata. Exact URLs and SHA-256 hashes are recorded in the family `NOTICE`,
`THIRD_PARTY_NOTICES.txt`, and `weights/LICENSE_NOTICE.txt`.

The original `.pth.tar` also contains pickled Adam optimizer state and cannot
be opened by LibreYOLO's untrusted-file safe loader. Convert the SHA-256-pinned
official archive with `weights/convert_quicksrnet_weights.py`, which is the
explicit trusted-artifact path, or use the hosted lean checkpoint above. A
plain tensor-only upstream state dict remains eligible for generic runtime
auto-conversion. LibreYOLO does not weaken safe loading for the training
archive.
