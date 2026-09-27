# ADR 0018: Edge task and specialist contract

Status: accepted
Date: 2026-07-29

## Context

Edge detectors emit a dense boundary-confidence field rather than boxes,
classes, or an opaque visualization. Existing model releases vary in channel
order, input mean, side-output handling, output inversion, thresholding, and
benchmark code. A stable task boundary is required before specialists or
exported runtimes can share one API.

Source-code licensing and learned-weight licensing are separate. TEED and
DexiNed publish their architectures under MIT, while their released weights
were trained on BIPED, whose published dataset terms are non-commercial.

## Decision

### Public payload

Add canonical task `edge`, filename suffix `-edge`, and aliases `edges`,
`boundary`, `boundaries`, `edge-detection`, and `edge_detection`.

`Results.edges` is an `EdgeMap` containing float32 `(H, W)` probabilities in
`[0, 1]` on the original image canvas. `EdgeMap.binary(threshold)` returns a
boolean mask without mutating the continuous payload. Plotting uses inverted
grayscale: probability one is black and zero is white.

### Dataset and metric

The canonical dataset pairs each image with a same-stem, single-channel,
lossless edge map and an optional validity mask. Dataset YAML may explicitly
invert black-on-white source annotations at ingestion.

Validation performs four-direction gradient non-maximum thinning internally.
At each probability threshold, predicted and target pixels are matched
one-to-one within a normalized correspondence radius:

```text
radius = edge_max_dist * hypot(height, width)
```

The default `edge_max_dist` is `0.0075`. ODS is the best aggregate-dataset
F-measure over the threshold sweep; OIS is the mean of each image's best
F-measure.

### Specialists

`LibreTEEDt-edge.pt` and `LibreDexiNedb-edge.pt` implement the edge task. Their
native graphs accept canonical RGB float32 `[0, 1]`, then perform the released
BGR mean subtraction inside the graph. Only the fused final logit is exposed,
after sigmoid; side outputs stay internal.

Architecture ports are pinned to:

- `xavysp/TEED` commit
  `40fa4b1391dc6424f88989d0ca75d5b592c8681d` (MIT);
- `xavysp/DexiNed` commit
  `08ed67ad0579f3969536a9719cdc1b829fb74fc1` (MIT).

Tensor parity against those checkouts is a gated test. ONNX export uses a
fixed-resolution, batch-one graph with one `edges` output. Backends apply the
same RGB stretch preprocessing and resize probabilities back to the source
canvas.

### Weight distribution

LibreYOLO does not bundle, mirror, or auto-download the authors' released
BIPED-trained checkpoints. Local conversion scripts only add the `core.`
runtime prefix and LibreYOLO checkpoint metadata; learned tensors are
unchanged. Conversion does not change the checkpoint's applicable terms.
Independently trained checkpoints may be distributed only under terms
established by their trainer and training-data provenance.

## Consequences

Task/results/dataset/validation/export code is usable with independently
trained compatible checkpoints under permissive terms. Users holding an
upstream checkpoint may convert and run it locally after determining that
their use complies with its terms.

No public edge checkpoint is available from the LibreYOLO organization until
a compatibly licensed training-data and weight grant is documented. Calling a
canonical missing filename fails with local conversion guidance instead of
silently downloading a restricted artifact.

## Non-goals

- Training recipes are not integrated in this version.
- Edge maps are not instance masks or semantic segmentation.
- Thresholded masks are not stored in `Results`; thresholding is a view.
- Tiled inference and test-time augmentation are not defined for dense edge
  maps.
- This ADR does not approve the BIPED dataset or BIPED-trained checkpoints for
  redistribution.
- The generic loader likewise does not distribute or relicense BSDS500, whose
  publisher limits dataset use to non-commercial research and education.
