# RF-DETR training implementation

The native trainer incorporates training optimizations from RF-DETR development
commit `b49df291cdd6b8d19fb722ab402e9a018d2edcd4` (Apache-2.0). It keeps the
LibreYOLO training API, task extensions, checkpoint format and portable paths.

## Matching

Detection matching computes each image's own costs. Eligible decoder levels share
one safety check and one compact CPU transfer. A bounded cost-storage budget,
small accelerator workloads, heavily skewed targets and unsafe inputs retain the
original path. Segmentation, keypoint and oriented-box matching retain their
specialized costs and fallback behavior.

SciPy remains available without an additional package. For the optional batched
GPU solver, install:

```bash
pip install 'libreyolo[rfdetr-train]'
```

`matcher_backend="auto"` uses the pinned `torch-hungarian==0.1.0rc0` backend on
eligible Linux NVIDIA CUDA devices (compute capability >= 8.0). CPU, MPS, Windows,
ROCm and installations without that package use SciPy. The solver's own supported
input policy still applies. Large padded solver batches use exact-size buckets.
`matcher_backend="scipy"` explicitly selects the original solver.

Equal-cost assignments can have multiple valid solutions. GPU and SciPy solvers
must agree on the optimal objective and assignment validity, but need not choose
the same indices on ties. Legacy checkpoints without a backend setting resume
with SciPy unless the caller explicitly overrides it.

## Model execution

The proposal box MLP runs only on selected encoder rows. Single-level feature
packing avoids redundant copies, and dense contiguous image batches reuse their
storage when padding is unnecessary.

Eager CUDA can additionally use upstream's batched proposal selection, grouped
self-attention, fused linear/ReLU, dtype-aware positional adds and interleaved
sine/cosine embedding. Type, shape and hook checks preserve ordinary module calls
for customized heads. CPU/MPS, tracing, compilation and torch.func transforms
retain their applicable portable operations. Learned parameter names and shapes
are unchanged.

Batched GEMMs can round differently from separate GEMMs. Near-tied proposal
scores can therefore change top-k selection under mixed precision. These are
execution optimizations, not a claim of bitwise-identical training trajectories
across devices or kernels.

## Optimizer and resume

Equivalent AdamW groups are merged after their defaults are resolved. Gradient
clipping retains the original parameter order. Legacy one-parameter groups and
new grouped checkpoints restore their Adam moments through a pre-load migration.
New optimizer group metadata records original parameter positions so subsequent
regrouping does not scramble state. Appended adapter groups remain separate.

Resolved learning rates are reapplied by the scheduler. Saved weight decay is
preserved unless the caller explicitly overrides it on resume. An unsafe
migration raises an error instead of silently continuing with fresh moments.
The number/order of optimizer groups is not a per-parameter indexing contract;
custom callbacks should inspect the groups' parameters and settings.

The RF-DETR trainer uses AdamW. Unsupported optimizer requests emit a warning,
and the resolved run configuration records AdamW.

## Configuration and numerical corrections

- `lr_encoder`, `lr_vit_layer_decay`, `lr_component_decay`, `focal_alpha` and
  `cls_loss_coef` can be supplied to `train`. Unspecified values resolve from the
  model and are recorded in the training configuration.
- Fresh classification-loss coefficients match upstream: 1 for detection,
  segmentation and oriented boxes; 2 for pose. Legacy resumed runs retain their
  historical coefficients (5 for segmentation, 1 otherwise) unless overridden.
- Matcher focal alpha follows the configured loss setting. IoU/GIoU denominators
  have upstream's finite guards, and cardinality logging uses sigmoid confidence.
- `early_stopping=True` maps to patience 10 when no explicit `patience` is supplied.
  The canonical `patience` setting takes precedence.
- Image normalization uses contiguous owned storage and preserves the previous
  FP32 pixel values. Detection target packing preserves valid-row order while
  avoiding a separate GPU count synchronization for every image.
- RF-DETR combines scalar loss/component logging into one transfer and honors
  `log_interval` for progress-text refreshes; epoch aggregates still include all
  batches. Other model families retain their existing logging path.
- `pin_memory`, `prefetch_factor` and `persistent_workers` are honored.
  `None` retains the previous effective task defaults: detection uses CUDA
  pinning, prefetch 2 and non-persistent workers; pose uses no pinning, prefetch 1
  and persistent workers. Worker-only options are omitted when `workers=0`.
  Detection workers limit OpenCV threading without changing PyTorch's RNG seeding.
  Legacy resumes preserve the old effective loader defaults unless overridden.

## Scope of the upstream comparison

Included here: matching/solver batching, optimizer grouping and migration,
proposal and decoder execution, redundant-copy removal, target/loss assembly,
normalization/loader plumbing and the numerical/configuration corrections above.

The native DDP strategy, precision defaults, CUDA-graph/compile runners and COCO
metric backend are retained. Upstream's Lightning orchestration, FP8/Transformer
Engine, XLA, custom fused optimizer/EMA runtime and alternative augmentation
frameworks are not substituted for LibreYOLO's trainer. These require different
runtime or recipe contracts and are not prerequisites for this update.

Model/loss parity, legacy resume, task fallbacks and native integration checks are
separate from long-run accuracy validation. Published checkpoint AP and complete
training convergence cannot be certified by a short performance probe alone.
