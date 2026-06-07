# ADR 0003: Point Task Contract

## Status

Accepted.

## Context

Some object-localization models learn a single coordinate per object rather
than a bounding box. These models still solve object detection as a product
problem, but their output geometry is not compatible with the `detect` task's
box contract, box NMS, and IoU-based validation metrics.

Examples include centroid-based object detectors, point-supervised crowd
localizers, and microscopy or cell-localization models that emit `(x, y)`
locations with class confidence.

## Decision

LibreYOLO defines a canonical `point` task for models whose public prediction
primitive is:

```text
x, y, class, confidence
```

`point` uses `-point` as its filename suffix. No public aliases are defined for
this task; callers must use the canonical spelling `point`.

The row order is intentionally point-specific. It follows the natural
coordinate-first FOMO-style tuple instead of the packed box/OBB convention,
where confidence and class appear after geometry. Point prediction order is
model-defined unless a model family documents a stronger ordering guarantee.

`detect` remains the task for axis-aligned box detectors. Point models must not
fabricate boxes only to satisfy box-oriented APIs. They should expose
`Results.points` and use point-specific validation metrics.

## Consequences

- Box-specific paths such as tracking, tiled box merging, validation, export,
  and exported backend box parsers must reject point results until point-aware
  implementations are added.
- Point model families may derive training targets from existing box labels,
  but that adapter is family-specific until a canonical point dataset schema is
  accepted.
- Official point checkpoints must write `task="point"` in LibreYOLO checkpoint
  metadata.

## Non-Goals

This ADR does not add a point model family, a point dataset file format, point
tracking, or exported backend point decoding.
