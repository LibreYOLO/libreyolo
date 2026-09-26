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
adapters into dense weights. The other GTR tasks (segmentation, pose, depth,
OBB, semantic segmentation) are not implemented.

```python
model = LibreYOLO("LibreGTRs.pt")
model.train(data="dataset.yaml", epochs=30, lora=True)
```

ONNX and TorchScript export use a fixed square FP32 graph. Other export
formats and dynamic spatial shapes are not enabled. The portable recurrence
exports as an ONNX Loop rather than an unrolled graph.
GTR-S at 640px exports in about 2.2 seconds locally with 4,094 graph nodes.
This is export usability evidence, not a GPU deployment-speed benchmark.

## Instance segmentation

`task="segment"` loads the upstream GTRSeg model: the detector plus a per-query
mask head fed by the highest-resolution encoder level
(`configs/seg/coco_seg_finetune`). Canonical names are
`LibreGTR{s,m,l,x}-seg.pt`, converted from `seg/gtrseg_{s,m,l,x}_coco.pth` in
the same MIT weight repository with EMA tensors unchanged; the raw upstream
files also auto-convert. Masks come out as logits at a quarter of the input
resolution and are thresholded at zero after bilinear upsampling, as upstream.

```python
model = LibreYOLO("LibreGTRs-seg.pt")
results = model.predict("image.jpg")  # results.masks
model.train(data="dataset.yaml", epochs=30)  # YOLO polygon labels
```

Training reuses the EC segmentation data path (square resize, polygon
rasterization, ImageNet normalization) with the GTR recipe: grouped matching,
MAL/box/FGL/DDF losses plus point-sampled mask BCE and Dice (weights from
`gtrseg_base.yml`), the GTR schedule and backbone LR multipliers. Mosaic and
MixUp are not applied to segmentation. `lora=True` works as for detection and
keeps the mask head trainable. Upstream initializes segmentation from the whole
COCO detector; the same start is available as an explicit transfer,
`LibreGTR("LibreGTRs.pt", size="s", task="segment",
allow_detect_to_segment_transfer=True)` or
`libreyolo train model=LibreGTRs.pt task=segment ...`, and the mask head then
starts untrained. ONNX and TorchScript export the raw logits, boxes and mask
logits with the same fixed-shape FP32 constraints as detection.

Segmentation evidence:

- All four checkpoints convert with identical tensors and load strictly. At
  640px on CPU they match the pinned upstream GTRSeg graph exactly
  (logits, boxes and mask logits, maximum absolute difference 0) with the same
  portable operator substitutes as the detection check.
- GTR-S on a 200-image COCO val2017 subset (the coco1000 validation split,
  polygons from `instances_val2017.json`, one polygon per instance, crowd
  regions excluded): GTR-S box mAP50-95 0.545, mask mAP50-95 0.493, mask
  mAP50 0.698; GTR-X box 0.605, mask 0.551, mask mAP50 0.773. Upstream
  reports mask AP 45.0/49.8 and AP50 67.9/74.2 (S/X) on full val2017. This is
  a subset sanity check, not a reproduction.
- GTR-S ONNX and TorchScript exports reproduce PyTorch predictions on a COCO
  image (same classes, score difference below 3e-6, mask IoU 1.0).
- CPU segment training runs end to end, including LoRA and the CLI
  detect-to-segment transfer. Starting from GTR-S detect weights, 90 plain
  AdamW steps on one 13-instance COCO image raise the mean best-query mask IoU
  from 0.085 to 0.62, so the fresh mask head learns. Short runs with the
  default EMA (tau 2000) still validate with near-initial EMA weights, so an
  untrained mask head scores zero mask mAP for the first few thousand steps.

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
