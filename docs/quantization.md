# PyTorch-Native Quantization

LibreYOLO quantizes models directly in PyTorch. Quantized models keep the
normal `predict` / `val` / `train` / `save` contract, so accuracy is measured
with the same validators as float models and accuracy recovery reuses the
existing training and distillation notation.

## Grammar

Two steps. Step 1 always happens; step 2 is optional accuracy recovery.

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLO9s.pt")

# Step 1: quantize (structure + calibration). calib is a small UNLABELED
# image set used forward-only to derive activation ranges and scales.
qmodel = model.quantize(recipe="int8", calib="coco128.yaml", samples=128)

qmodel.val(data="coco8.yaml")            # honest accuracy, same validators
qmodel.predict("bus.jpg")
qmodel.save("LibreYOLO9s-int8.pt")       # manifest-carrying checkpoint

# Step 2 (optional): QAT is plain train() on the quantized model.
qmodel.train(data="coco.yaml", epochs=5)

# QAD: same step plus the existing distillation kwargs.
qmodel.train(data="coco.yaml", epochs=5, distill_model="LibreYOLO9m.pt")
```

CLI:

```bash
libreyolo quantize --model LibreYOLO9s.pt --recipe int8 --calib coco8.yaml
libreyolo train --model LibreYOLO9s-int8.pt --data coco.yaml --epochs 5
```

`LibreYOLO("LibreYOLO9s-int8.pt")` restores the quantized structure and
scales automatically (checkpoints carry a `quant` manifest; see
`checkpoint_schema.md`). Trainer checkpoints written during QAT/QAD carry the
manifest too, so `best.pt` from a QAT run is itself a quantized checkpoint.

## Recipes

| Recipe | What it does | Families (v1) | Calibration |
|---|---|---|---|
| `fp16` | Cast to half precision with a float32 I/O contract. Inference-only. | yolo9, rfdetr, birefnet, feynobg | none |
| `bf16` | Cast to bfloat16 (fp32's exponent range at half storage; the fix when fp16 overflows on DETR-style models). Inference-only. | yolo9, rfdetr, birefnet, feynobg | none |
| `fp8` | E4M3 W+A simulation: per-channel weight scales by default, calibrated per-tensor activation scales, on `Conv2d` and `Linear`. Selected Linear modules may use a manifest-recorded tensorwise weight scale when that is faster and validation-safe. | yolo9, rfdetr, birefnet, feynobg | required for activations |
| `int8` | W8A8 simulation: per-channel symmetric INT8 weights, per-tensor affine INT8 activations, on `Conv2d` and `Linear`. | yolo9, rfdetr, birefnet, feynobg | required for activations (skipped with `calib=None`, weights-only) |
| `w4a16` | Grouped symmetric INT4 weights (group 128 along in_features), float activations, on `Linear`. | rfdetr, birefnet, feynobg | not needed (weight-only) |
| `w4a8` | Grouped INT4 weights plus calibrated INT8 activations, on `Linear`. Maps to NPU W4A8 deployments (Hexagon, Hailo `a8_w4`). | rfdetr, birefnet, feynobg | required for activations |
| `nvfp4` | W4A4 NVFP4 simulation on `Linear`: E2M1 elements, 16-element blocks, FP8 E4M3 block scales, FP32 tensor scale. Dynamic activation scaling. | rfdetr, birefnet, feynobg | not needed (dynamic) |
| `mxfp4` | OCP MXFP4 on `Linear`: E2M1 elements, 32-element blocks, power-of-two (E8M0) block scales. Dynamic activation scaling. | rfdetr, birefnet, feynobg | not needed (dynamic) |
| `int2` | Research preview: grouped 2-bit weights (group 64) plus INT8 activations, on `Linear`. PTQ alone is unusable; QAT/QAD required. | rfdetr | required for activations |

Linear-only recipes are rejected for conv-heavy families such as yolo9 on
purpose: sub-8-bit acceleration is GEMM-only on current hardware, so
convolutions stay in higher precision. Transformer families (RF-DETR, and
the Swin-backed birefnet/feynobg matte families) are the target; yolo9 uses
`int8` or `fp8`. birefnet and feynobg are inference-only, so QAT/QAD healing
is unavailable there; `int2` is rejected for both for that reason (PTQ-only
int2 is unusable).

Per-family `keep_high_precision` defaults protect the first layer and the
heads (and always the YOLO9 DFL conv). For birefnet and feynobg that means
the Swin patch embed, the final matte-logit conv (`conv_out1`), the tiny
bilateral-reference attention gates (`gdt_convs_attn`), and the
training-only supervision heads (`gdt_convs_pred`, `conv_ms_spvn`), which
never run at inference and would otherwise sit permanently uncalibrated in
the manifest, and the deformable-conv weight containers (`regular_conv`),
whose weights are read directly by `torchvision.ops.deform_conv2d` rather
than through the module forward, so module-swap quantization cannot cover
them. Override with `quantize(..., keep_high_precision=("head.",))` if you
know what you are doing.

## Calibration data is not training data

- `calib=` (quantize): a few hundred images, no labels read, forward-only.
  Purpose: activation ranges and scale generation. Default: `coco128.yaml`
  (auto-downloaded); multiple batches matter because ranges are estimated
  across them.
- `data=` (train/val): the labeled dataset. Purpose: gradients and metrics.

Activation range estimation (`algorithm=`): the default `minmax` keeps the
absolute extremes seen across calibration batches; `percentile` uses the mean
of per-batch 0.1/99.9 percentiles. Measured on
coco128, minmax with a multi-batch calibration set wins for every tested
model, and percentile clipping collapses DETR-family accuracy because
transformer activation outliers are functionally load-bearing. What
actually fixes small-model int8 sensitivity is calibrating on enough
batches (hence the coco128 default: with it, YOLO9-t lands within about one
mAP point of fp32). The chosen algorithm is recorded in the checkpoint
manifest.

Two histogram-based algorithms complement those. `mse` sweeps candidate
clipping thresholds over a 2048-bin histogram of absolute activation values
and picks the threshold minimizing the quantization reconstruction error, so
it only clips as far as squared error actually improves; on clean
distributions it lands next to minmax. The sweep simulates the affine
codebook the calibrated layer actually deploys: activations that never
cross zero (post-ReLU layers) keep all 256 int8 codes on one side, so they
are judged at that doubled resolution rather than a symmetric stand-in. `entropy` picks the threshold
minimizing the KL divergence between the original and quantized activation
distributions, which clips more aggressively because it optimizes
information preservation rather than squared error. Both help on layers
whose activations carry rare large outliers that would otherwise stretch the
int8 range and crush resolution for the bulk of values. Their cost is
histogram collection during the calibration pass plus a threshold sweep per
layer at the end; calibration takes somewhat longer, inference is unchanged.
Both set the per-tensor activation range only; weight scales stay
per-channel absolute-max under every algorithm. The same caution as
percentile applies to transformer families, where clipping activation
outliers can cost accuracy: validate with `val()` before deploying.

## Execution tiers

v1 executes quantized arithmetic in **simulation** (fake-quantization with
straight-through-estimator gradients, computed in fp32 islands even under
AMP). Simulation is numerics-true: a `val()` score on any device is a real
claim about the quantized arithmetic. It is not a speed claim; packed
low-bit kernels are a separate deployment concern. The `fp16` and `bf16`
casts are the exception: they execute natively.

**Native fp8 tier** (finalized checkpoints on fp8 tensor cores, Ada sm_89 /
Hopper / Blackwell): finalized fp8 `QuantLinear` modules run their GEMM
directly on the packed E4M3 weights via `torch._scaled_mm` (the `fp8_gemm`
registry kernel), using the same calibrated static activation scales as the
simulation. The optional Triton tier fuses activation scaling, saturation,
and E4M3 conversion into one pass. For per-channel weights it also fuses the
bounded row-scale and bias epilogue into one pass; modules explicitly listed
in the manifest's `fp8_tensorwise_weights` fuse the weight scale and bias
directly into cuBLASLt. Without Triton, the same arithmetic uses stock PyTorch
operations. Finalized fp8 `QuantConv2d` modules convolve in fp16 against
weights dequantized from the packed E4M3 codes (the standard fp8-deployment
convention; the E4M3 activation snap on conv inputs is simulation-only).
Finalize with `remainder="fp16"` so the non-quantized interior runs in half
precision (the loader installs the same float32 I/O root hooks the cast
recipes use). Residual drift vs the simulated tier is half-precision
rounding plus GEMM summation order; `LIBREYOLO_KERNELS=off` (legacy alias
`LIBREYOLO_QUANT_KERNELS`) restores the exact simulated path everywhere. Measured on LibreFeyNobg (263M Swin-L
matte, RTX 5070 Ti, 1024px, controlled ABBA runs): fp8 vs fp16 is
85.7 vs 95.0 ms for a batch-1 graphed forward and 123.1 vs 129.3 ms through
the full graphed `predict` path. At batch 4 the full path is 515.4 vs
535.3 ms. The finalized fp8 file is 275 MB vs 531 MB for fp16.

`model.quant_info()` reports the recipe and module state;
`libreyolo.kernels.active()` reports the selected implementations (the
registry lives in `libreyolo/kernels/`; see `docs/kernels.md`).
Linux CUDA PyTorch environments commonly already include Triton. On Windows,
install a PyTorch-compatible `triton-windows` build to enable the fused cast
and epilogue; inference remains functional without it.

### LibreMODUS local weight-only FP8

LibreMODUS's `dtype="fp8"` path is deliberately separate from the finalized
checkpoint tier above. Its external checkpoint cannot be redistributed, so it
is converted locally and cached as sharded safetensors. Eligible linear weights
in the interior decoder blocks use E4M3 plus one FP16 scale per output row;
forward dequantizes them to the input dtype before `F.linear`.

This reduces checkpoint/parameter storage but does **not** quantize activations
or call the native `fp8_gemm` registry. It therefore makes no tensor-core speed
claim. Embeddings, `lm_head`, norms, timestep/AdaLN modulation, first/last
decoder blocks, projectors, SigLIP, and the FLUX VAE remain BF16. The cache key
contains the immutable source revision (or local file SHA-256) and full recipe,
and the cache is never uploaded. See [`libremodus.md`](libremodus.md) for usage
and [`testing.md`](testing.md) for its quality/VRAM acceptance gates.

The remaining `quant_info()` fields report module counts, calibration state,
and execution tier.

## Export

### Finalized PyTorch checkpoints (`format="pt"`)

A prepared checkpoint keeps fp32 masters because training needs them. When
you are done, crystallize:

```python
qmodel.export(format="pt")   # -> <name>-final.pt, packed low-bit weights
```

Finalized checkpoints store real packed weights (int8 tensors + per-channel
scales; nvfp4 as two-codes-per-byte E2M1 payload + E4M3 block scales),
strip the masters, and cast the non-quantized remainder to fp16
(`remainder="fp32"` keeps it exact). Measured: YOLO9-s int8 29.5 to 9.6 MB,
RF-DETR-n nvfp4 122 to 26 MB. The packing invariant: unpacking reproduces
the simulation bit for bit on the device you finalized on, so a
`remainder="fp32"` file scores exactly what you validated. The default fp16
remainder adds half-precision rounding on top; re-run `val()` on the
finalized file if you need its exact score. Loading one gives an
inference-ready model; `train()` on it re-prepares masters from the packed
weights automatically (QAT-from-PTQ); ONNX export from it re-prepares
internally and emits the same QDQ graph. The packed layout is documented in
`checkpoint_schema.md` as the connection contract for external exporters
and runtimes.

### ONNX (`format="onnx"`)

int8-quantized models export directly to ONNX with in-graph
QuantizeLinear/DequantizeLinear pairs carrying the model's own calibrated
(or QAT-trained) scales:

```python
qmodel = LibreYOLO("LibreYOLO9s-int8.pt")   # PTQ or QAT/QAD checkpoint
qmodel.export(format="onnx")                # scale-exact QDQ INT8 ONNX
```

ONNX Runtime and TensorRT consume the QDQ graph with real INT8 kernels; on
coco8 the exported artifact tracks the PyTorch simulation within sub-point
noise. The CLI equivalent is
`libreyolo export --model model-int8.pt --format onnx`. Notes:

- Cast recipes (`fp16`/`bf16`): call `dequantize()` and use the float
  exporters (`half=True` gives fp16 ONNX).
- Sub-8-bit linear recipes (`w4a16`, `w4a8`, `nvfp4`, `mxfp4`, `int2`) and
  `fp8` have no deployable ONNX form here yet; they execute in PyTorch and
  crystallize via `format="pt"`.
- Other deployment formats for int8 are built downstream from the QDQ ONNX;
  direct engine export is planned.
- `dequantize()` remains available to restore float masters (QAT-trained
  weights are kept) and use any float exporter.

## QAT and QAD mechanics

Quantized modules keep fp32 master weights; fake-quantization applies STE so
gradients flow to the masters. AMP, checkpoint resume, and the `distill_*`
kwargs (MGD/CWD) compose with QAT. At setup, QAT automatically disables EMA
and SyncBatchNorm and logs the changes because both can interfere with
fake-quant observer and scale state. Float training is unchanged.
`fp16`- and `bf16`-quantized models are inference-only; the trainer rejects
them with a pointer to `amp=True`.

QAT is a finetune of an already-trained model: use finetune learning rates
(for example `lr0=1e-4` for yolo9), not the from-scratch defaults, or the
short run will destroy the pretrained weights regardless of quantization.

QAD availability follows family distillation support: it works wherever the
family implements `get_distill_config()` (yolo9 and rfdetr today; the
RF-DETR tap point is the stride-16 backbone projector output, probed from
the live model so future sizes stay correct).

Family notes: RF-DETR calibration exercises the inference path, so modules
that only run during training (denoising branches) keep their activation
observers open and stay unquantized on activations until QAT runs;
`quant_info()["calibrated"]` reports this honestly. The RF-DETR trainer also
reinitializes the detection head when the dataset class count differs from
the checkpoint head width (COCO checkpoints have a 91-wide head), which
applies to quantized finetunes exactly as it does to float ones.
