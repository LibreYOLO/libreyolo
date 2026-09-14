# ADR 0025: Relative depth encodings

Status: implemented
Date: 2026-09-10

## Decision

Extend `DepthMap` with the keyword-only `encoding` field. Its default remains
`inverse_depth`, preserving existing model outputs and validation. Additional
values are `depth` and `log_depth`. They describe an affine-relative prediction
in the named space; none implies metres or a scale shared across images.

This is required for Marigold V2. Its default checkpoint predicts normalized
log-depth. Exponentiating that prediction before recovering its unknown scale
and shift would change the represented geometry. Negating it would preserve
ordering but would not turn it into inverse depth.

The stored data remains the original numeric prediction on the original image
canvas. `near_is_high` is true only for `inverse_depth`. Rendering reverses the
colour ordering for the other encodings. `normalized()` continues to perform
ordinary finite-value min/max normalization without changing the encoding.
Device transfers, NumPy conversion and Results slicing retain the field.

Validation reads the model's `depth_encoding`, defaulting to `inverse_depth`.
It fits a positive scale and shift in that space against ground truth, then
decodes to ground-truth depth units for AbsRel, RMSE and delta metrics. Invalid
or non-positive fits retain the existing median-shift fallback. This shared
least-squares protocol does not reproduce the upstream Marigold RANSAC benchmark.

Checkpoints may record `depth_encoding`. Families using a non-default encoding
must validate it against their variant and return it from postprocessing.
Existing exported backends remain inverse-depth-only; a family with another
encoding must not enable export without extending backend metadata and parsing.

## Compatibility

No existing family changes its data or default encoding. Default inverse-depth
JSON summaries keep their existing shape; non-default summaries include
`encoding`. The new field is optional and keyword-only.
