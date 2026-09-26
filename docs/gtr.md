# GTR detection

`LibreGTR` supports detection with the upstream S, M, L and X architectures,
and monocular depth (see [Depth](#depth)).
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
adapters into dense weights. Detection LoRA does not apply to depth (see
[Depth](#depth)). GTR segmentation, pose, OBB and semantic segmentation are
not implemented.

```python
model = LibreYOLO("LibreGTRs.pt")
model.train(data="dataset.yaml", epochs=30, lora=True)
```

ONNX and TorchScript export use a fixed square FP32 graph. Other export
formats and dynamic spatial shapes are not enabled. The portable recurrence
exports as an ONNX Loop rather than an unrolled graph.
GTR-S at 640px exports in about 2.2 seconds locally with 4,094 graph nodes.
This is export usability evidence, not a GPU deployment-speed benchmark.

## Depth

`task="depth"` loads the upstream `gtrdepth_{s,m,l,x}` models: the GTR
backbone and encoder with a DPT depth head (upstream's adaptation of the
Depth-Anything-V2 metric head, Apache-2.0), pretrained on upstream's mixed
metric-depth corpus with a log-depth head (metres, `exp(clamp(logit, -4, 5))`).
There is no NYU fine-tune; upstream reports NYU Depth V2 after a per-image
log-affine fit with six test-time views.

```python
model = LibreYOLO("LibreGTRs-depth.pt")
result = model.predict("image.jpg")[0]
inverse_depth = result.depth_map  # higher is closer; 1 / value is metres
```

`Results.depth_map` follows the LibreYOLO depth contract (ADR 0006): relative
inverse depth on the original canvas. The graph emits the exact reciprocal of
upstream's metre output, so `1 / depth_map` recovers it; the metre scale holds
only for cameras and scenes like the training data. ImageNet normalization is
inside the graph, and native preprocessing stretch-resizes to a square
`imgsz` (default 640), as upstream validation does. Exported backends use the
shared cv2 stretch resize, so exported predictions on downscaled images are an
approximation of native ones; given the same input tensor, ONNX matches within
5e-6 and TorchScript exactly.

`val()` uses the shared depth validator (per-image scale and shift fitted in
inverse-depth space), not upstream's log-affine, six-view, two-tile protocol,
so its numbers are not comparable to the upstream table. Fine-tuning uses the
upstream SILog loss and `configs/depth/gtrdepth_base.yml` recipe (AdamW 2e-4,
backbone at 0.1x, warmup, flat then cosine to 0.1x, EMA 0.999, clipping 0.1,
FP32) on the shared depth dataset; supply depth in metres to keep the metric
scale. LoRA, TTA, tiled inference and tracking are rejected for depth.

Depth evidence:

- All four real checkpoints load strictly, and at 640px the converted models
  match the pinned upstream graph exactly (max abs difference 0.0 on
  `pred_depth`) with the same portable operators as the detection check.
- On 9 NYU Depth V2 validation images (metres, stretch 640, shared
  validator): delta1 0.706 / 0.762 / 0.808 / 0.843 and AbsRel 0.172 / 0.155 /
  0.141 / 0.127 for S / M / L / X. This is a pipeline check, not a benchmark.
- Converted checkpoints are `LibreYOLO/LibreGTR{s,m,l,x}-depth`. Download
  revisions are placeholders in `LibreGTR.HF_TASK_REVISIONS` until the
  repositories are published.

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

## Oriented boxes (DOTA)

`task="obb"` runs the upstream DOTA v1.0 oriented-box models. Upstream publishes
S and X weights only; its M and L OBB configs use a different, unreleased plain
ViTAdapter backbone, so LibreYOLO builds S and X.

```python
model = LibreYOLO("LibreGTRs-obb.pt")
result = model.predict("aerial.jpg")
result.obb.xywhr  # (cx, cy, w, h, theta) in pixels, theta in [0, pi)
```

- Canonical names `LibreGTR{s,x}-obb.pt` download from `LibreYOLO/LibreGTR{s,x}-obb`.
  Raw upstream `gtrobb_{s,x}_dota.pth` files auto-convert with EMA tensors unchanged.
- Input is a fixed 1024px square. Images are resized to fit without distortion,
  padded at the bottom and right with zeros (the DOTA split convention),
  converted to RGB 0-1 and ImageNet-normalized. Decoding is NMS-free top-K.
- Angles use the long-edge convention: `w >= h` and `theta` in `[0, pi)`, the same
  contract as RT-DETRv2 OBB.
- Class ids follow the upstream DOTA v1.0 order (`plane`, `baseball-diamond`,
  `bridge`, `ground-track-field`, `small-vehicle`, `large-vehicle`, `ship`,
  `tennis-court`, `basketball-court`, `storage-tank`, `soccer-ball-field`,
  `roundabout`, `harbor`, `swimming-pool`, `helicopter`). Datasets labeled in a
  different order, such as Ultralytics' DOTA8, need their label ids remapped
  before `val()`.
- ONNX and TorchScript export a fixed 1024px FP32 graph; exported models predict
  and validate through the same decoding. GTR-S OBB ONNX export takes about three
  minutes on CPU.
- OBB is inference-only: `train()` raises. Porting the upstream oriented
  criterion (MAL, KLD box loss, six-distribution FGL, Chamfer/KLD matching) and
  its flip/rotate augmentation is future work.

Evidence, CPU:

- Both checkpoints strictly load, and their logits and boxes match the pinned
  upstream graph exactly (max abs diff 0.0) at 1024px with the same portable
  attention substitution as detection. Decoded rotated boxes and scores match
  upstream's `OBBPostProcessor` exactly (`tests/unit/test_gtr_obb_parity.py`).
- The publisher reports DOTA-v1.0 test AP50 of 80.0 (S) and 81.3 (X); LibreYOLO
  has not reproduced these.
- On the 4 DOTA8 val tiles (label ids remapped to the upstream order), S reaches
  mAP50 1.000 / mAP50-95 0.525 and X 1.000 / 0.657. This is a pipeline check on
  a tiny sample that may overlap upstream training data, not a DOTA benchmark.
- ONNX and TorchScript exports of S reproduce PyTorch predictions (max abs diff
  9e-4 px and 6e-8) and the same DOTA8 metrics.

Reproduce the parity check without network access:

```sh
GTR_UPSTREAM=/path/to/GTR GTR_CHECKPOINTS=/path/to/weights \
  pytest tests/unit/test_gtr_obb_parity.py -m 'unit and external_data'
```
