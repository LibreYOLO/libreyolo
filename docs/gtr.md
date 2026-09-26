# GTR detection

`LibreGTR` supports detection with the upstream S, M, L and X architectures.
The GTR source pin is `782e737efe2e6437ac537fbdcee089673d3376c1` (MIT).
The upstream weight repository explicitly declares MIT. Required inherited
Apache-2.0 notices are retained in `libreyolo/models/gtr/NOTICE`.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreGTRs.pt")
results = model.predict("image.jpg")
model.train(data="dataset.yaml", epochs=30, device="cuda")
model.export(format="onnx", dynamic=False)
```

Canonical names download converted checkpoints from `LibreYOLO/LibreGTR{s,m,l,x}`
at pinned revisions. Their learned tensors match the authors' published EMA
parameters exactly. Local
`gtr_s_coco.pth` files also auto-convert. An explicit converter is available as
`weights/convert_gtr_weights.py`. Learned tensor names and values are preserved.
Each hosted repository includes the upstream MIT license and checkpoint provenance.

## Execution and training

CPU and MPS use a differentiable PyTorch recurrence. CUDA uses the MIT
`flash-linear-attention` package when installed, otherwise the same portable
recurrence. Install `flash-linear-attention==0.5.0` separately on a compatible
CUDA system for that optional path. Neither Triton nor transformers is required
to import the model. The upstream custom CUDA/TensorRT plugin is not vendored.
The portable recurrence prioritizes compatibility; it does not reproduce the
published GPU latency.

Inputs are square, at least 160 pixels, and divisible by 32. Preprocessing
uses square resize, RGB 0-1 and ImageNet normalization. Default resolution is
640. Postprocessing uses DETR top-K selection without NMS.

Training uses upstream group-DETR matching and MAL/FGL/DDF losses through the
shared D-FINE trainer. FP32 is the default. Backbone LR multipliers and weight
decay follow the size-specific upstream recipe, with identical Python/CLI
defaults. Quadratic warmup, the six-epoch flat phase, cosine decay and the final
two-epoch plateau follow the upstream 30-epoch recipe. Shorter runs bound these
phases to fit their budget.

Training transforms receive the original-resolution image and apply square
resize, ImageNet normalization, horizontal flip, photometric distortion,
zoom-out and IoU cropping. The final two epochs disable the strong transforms.
The upstream Mosaic and batch MixUp are included: during the first
`mosaic_epochs` (default 6) epochs, each sample becomes a four-tile mosaic with
probability `mosaic_prob` (default 0.5), built from a 50-image per-worker cache
of half-resolution tiles and followed by a random affine (`degrees`,
`translate`, `mosaic_scale`). Mosaic samples skip zoom-out and IoU cropping.
Each batch is blended with its shifted copy with probability `mixup_prob`
(default 0.5) at a 0.45-0.55 ratio, keeping both label sets. Set
`mosaic_prob=0` or `mixup_prob=0` to turn either off. `mosaic_epochs` is a
Python training argument only. Multi-scale training defaults off.
Unsupported optimizer and scheduler choices are rejected instead of being
silently ignored.

`lora=True` fine-tunes with LoRA adapters on the backbone q/k/v projections
and the decoder layers while the backbone base stays frozen (see
[lora.md](lora.md)); it needs `pip install "libreyolo[lora]"` and works from
Python and the CLI. Adapter checkpoints reload directly and export merges the
adapters into dense weights. Semantic segmentation is described below; the
other GTR tasks (segmentation, pose, depth, OBB) are not implemented.

```python
model = LibreYOLO("LibreGTRs.pt")
model.train(data="dataset.yaml", epochs=30, lora=True)
```

ONNX and TorchScript export use a fixed square FP32 graph. Other export
formats and dynamic spatial shapes are not enabled. The portable recurrence
exports as an ONNX Loop rather than an unrolled graph.
GTR-S at 640px exports in about 2.2 seconds locally with 4,094 graph nodes.
This is export usability evidence, not a GPU deployment-speed benchmark.

## Validation evidence

- All four real EMA checkpoints strictly load without missing learned tensors.
- At 640px, all four models match the pinned upstream graph exactly for logits
  and boxes on CPU. Both graphs use the same portable attention/normalization
  substitutes. This establishes graph and key-mapping parity, not parity with
  the upstream Triton or custom CUDA kernels.
- The portable recurrence and its gradients agree with an independent explicit
  causal-sum formulation in unit tests.
- All four sizes pass real-weight ONNX and TorchScript reload and public
  detection parity at the default 640px on CPU. The ONNX gate uses a stable
  log-sigmoid expression to
  avoid underflow in the legacy exporter's log(sigmoid(x)) lowering.
- CPU training completes and saves checkpoints. A 30-epoch, two-image synthetic
  square fixture is memorized; the reloaded best checkpoint reaches 0.9731 box
  IoU against its training target. This is a pipeline smoke check, not a
  real-data convergence result.
- Resume restores saved training settings before applying explicit overrides;
  image size, batch, learning rate, EMA and schedule settings are regression-tested.
- A clean-directory canonical download and the HTTP UI inference path render
  an annotated image with a detection summary.

Still required before performance or production-readiness claims: native FLA
and custom-kernel parity on CUDA, COCO AP reproduction, RF1/RF5 training,
real-data convergence and deployment-speed measurements. True multi-rank DDP
has not been validated. Native MPS prediction was smoke-tested
at 160px and 640px. GPU, COCO and RF1 checks are explicitly deferred to the
v1.6.0 release validation.

Reproduce the opt-in real-weight checks without network access:

```sh
GTR_UPSTREAM=/path/to/GTR GTR_CHECKPOINTS=/path/to/detection/weights \
  pytest tests/unit/test_gtr_parity.py -m 'unit and external_data'
GTR_CHECKPOINTS=/path/to/detection/weights \
  pytest tests/unit/test_gtr_export.py -m 'unit and external_data'
```

## Semantic segmentation

`LibreYOLO("LibreGTR{s,m,l,x}-sem.pt")` loads the upstream Cityscapes
checkpoints (`semseg/gtrsemseg_*_cityscapes.pth` at the pinned weight
revision, EMA tensors unchanged) into the same `LibreGTR` class with
`task="semantic"`. The network is the GTR backbone, a one-level stride-8
encoder and an FCN head (conv3x3, BN, ReLU, dropout, conv1x1) over the 19
Cityscapes train IDs; 255 is ignore.

```python
model = LibreYOLO("LibreGTRs-sem.pt")
mask = model.predict("street.jpg")[0].semantic_mask
model.val(data="cityscapes.yaml")
model.train(data="my_semantic.yaml", epochs=30)
```

These pretrained weights are trained on Cityscapes, whose terms restrict the
dataset and derived models to non-commercial use; a download notice says so.
Fine-tuned or from-scratch weights carry no such term.

Geometry follows upstream evaluation. The canvas is 1024x2048. Images are
letterboxed into it (top-left, grey pad) and the network averages the logits
of overlapping 1024px square windows at a 768px stride, then crops the padding
and resizes to the source image. A Cityscapes frame therefore runs exactly as
upstream's slide inference: three windows, no resizing. A square input whose
side is divisible by 32 runs in one pass; inputs shorter than 1024px are
rescaled for the windows and rescaled back. `predict`, `val` and exported
backends share this letterbox, and ImageNet normalization happens inside the
network (the input is RGB in [0, 1]).

ONNX and TorchScript export a fixed 1024x2048 FP32 graph that contains the
three windows. ONNX export skips onnxsim by default because it spends about
15 minutes folding the recurrence Loops on CPU without shrinking the graph;
pass `simplify=True` to run it anyway. GTR-S exports in about 8 seconds.

Training uses the shared semantic trainer with the upstream recipe: AdamW,
base LR 5e-4, backbone LR multiplier 0.3 (S), 0.36 (M) or 0.24 (L/X), weight
decay 1e-4 (S/M) or 1.25e-4 (L/X) and none on norm/BN/bias, 2000-iteration
quadratic warmup, 6 flat epochs, cosine decay to half the LR, EMA 0.9999,
gradient clipping 0.1, FP32, 30 epochs, batch 8 and 1024px square crops.
Augmentation is flip, torchvision photometric distortion and large-scale
jitter (long side fit to 1024 times 1 to 4, then a crop), through the shared
semantic dataset. It differs from upstream in the grey image padding and in
not re-sampling crops dominated by one class. The head and patch embedding use
BatchNorm; upstream trains with SyncBN over a global batch of 8, and much
smaller single-process batches give noisy batch statistics.

Semantic validation evidence:

- All four checkpoints strictly load. On CPU with the portable operators, all
  four match the pinned upstream graph exactly (max abs diff 0.0) for a
  single 1024px window and for upstream's own `slide_inference` over a
  1024x2048 input.
- GTR-S predicts sensible maps on the bundled photos (sky, building, person,
  vegetation, sidewalk).
- GTR-S ONNX and TorchScript exports reload and reproduce the PyTorch mask
  with 100% pixel agreement on a 3072x1194 photo.
- `val()` on two photos labeled with the model's own `predict()` output gives
  99.5% pixel accuracy; the residue is the canvas-resolution comparison.
- A two-epoch CPU fine-tune from GTR-S trains, validates, saves and reloads.
  It is a pipeline check, not a convergence result.

Not yet measured: Cityscapes mIoU (no Cityscapes copy was available), GPU
speed, and real-data fine-tuning convergence.

```sh
GTR_UPSTREAM=/path/to/GTR GTR_SEM_CHECKPOINTS=/path/to/semseg/weights \
  pytest tests/unit/test_gtr_sem_parity.py -m 'unit and external_data'
```
