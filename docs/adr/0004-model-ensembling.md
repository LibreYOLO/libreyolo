# ADR 0004: Model Ensembling and Standalone Fusion Ops

- Status: Accepted (Phase 1 implemented)
- Date: 2026-06-10
- Scope: Inference architecture, shared ops primitives

## Context

LibreYOLO had no way to combine two detectors. Users who want higher accuracy
than any single model, or redundancy in safety-critical deployments (boxes
confirmed by independent models), had to hand-roll fusion glue outside the
library.

Every family — grid (YOLO9, YOLOX, …) and DETR (RF-DETR, D-FINE, …) — and
every exported backend already funnels into one uniform contract:
`predict()` → `Results` with `Boxes(xyxy, conf, cls)` in original-image
pixels plus a `names` dict. That contract is the ensembling seam.

## Decision

1. **Fuse at the detection level**, after each member's own
   preprocess → forward → postprocess. Never at the tensor level. Members
   keep their own input sizes, normalization, and suppression. This is what
   allows YOLO9 + RF-DETR, different class counts, and mixing a `.pt` member
   with an exported `.onnx` member.
2. **Fusion primitives are standalone torch ops** in `libreyolo/ops/`,
   model-free and importable on their own. Pure torch is a hard constraint:
   it keeps fusion traceable for a future baked-ensemble export and
   GPU-capable for free. `libreyolo/ops` also seeds the longer-term
   postprocessing consolidation (own ADR when pursued).
3. **`LibreEnsemble` is a thin wrapper** in `libreyolo/ensemble/` over
   duck-typed members. It is not a `BaseModel` subclass (it has no single
   forward); it implements the prediction surface and returns ordinary
   `Results`.
4. **Class spaces union by name.** Identical `names` dicts pass through;
   otherwise names are unioned, member class ids are remapped through small
   LUTs, and the fused `Results.names` is the union. Fusion only merges boxes
   within the same unified class; a class known to only one member passes
   through unfused. Mismatched spaces log a loud warning at construction.
5. **Consensus is first-class.** WBF clustering tracks which members
   contributed to each cluster; `min_votes` is the hard filter on that
   bookkeeping. The votes required for a class are capped at the number of
   members whose label space contains that class, so consensus stays
   meaningful on partially shared label spaces.
6. **Weight-space merging** (parameter averaging across same-architecture
   checkpoints) is a different feature with zero inference overhead and is
   out of scope here.

## Public API

```python
from libreyolo import LibreEnsemble
from libreyolo.ensemble import ExternalDetector

# The two-liner. Paths load via the LibreYOLO() factory; constructed models
# and exported backends are accepted as-is.
ens = LibreEnsemble(["LibreYOLO9s.pt", "LibreRFDETRs.pt"])
results = ens("image.jpg", conf=0.25)        # ordinary Results, union names

# All knobs.
ens = LibreEnsemble(
    ["LibreYOLO9s.pt", "LibreRFDETRs.pt"],
    weights=[2.0, 1.0],      # per-member trust (convention: proportional to val mAP)
    fusion="wbf",            # "wbf" | "wbf_seeded" | "nms" | callable
    fusion_iou=0.55,         # cluster threshold
    min_votes=1,             # 2 = consensus: keep only boxes both members found
)

# Per-member overrides where score calibration differs.
results = ens("image.jpg", conf=[0.25, 0.4])

# Members from outside the library: a callable returning detections in
# original-image pixels plus a names dict. LibreYOLO imports nothing foreign.
ens = LibreEnsemble([
    "LibreYOLO9s.pt",
    ExternalDetector(my_fn, names={0: "person"}),  # my_fn(pil) -> (boxes, scores, labels)
])
```

Custom combination functions receive stacked, unified detections:

```python
def my_fusion(boxes, scores, labels, model_ids, *, weights, num_models, **kw):
    """Tensors, original-image pixels, unified label ids; model_ids says
    which member produced each row. Returns (boxes, scores, labels)."""
```

`model_ids` is deliberate: it is what makes voting / veto / cascade rules
expressible.

## Pinned semantics

- Predict kwargs (`conf`, `iou`, `imgsz`, `device`, `classes`, `max_det`,
  `save`, `augment`) keep their standard meaning and broadcast to members;
  `conf`, `iou`, and `device` also accept one value per member, and `imgsz`
  accepts a *list* with one entry per member — an int or tuple broadcasts to
  everyone (each entry must be valid for that member's family — e.g.
  divisible-by-32 constraints still apply). `augment` broadcasts to members
  that support test-time augmentation; exported-backend members ignore it.
  `iou` remains the member NMS threshold; the fusion threshold is
  `fusion_iou` on the constructor.
- `classes` (union ids) and `max_det` apply to the fused result; members run
  generously and the ensemble trims once.
- The source image is decoded once and handed to members as PIL.
- Members may live on different devices; fusion runs on the first member's
  output device (post-suppression row counts are small).
- v1 task scope is detect only; any non-detect member raises at
  construction. Sources: image path / PIL / numpy / bytes / directory.
  Video and stream raise in v1 (Phase 2).
- `Results.speed` reports per-member inference times plus fusion
  (`member_0`, `member_1`, …, `fusion`), so the N× cost is visible.
- WBF's `min(W_T, W_N) / W_N` rescale means fused scores can drop below the
  per-member `conf`: a box only one of two members found keeps half its
  score. That soft-consensus signal is intentional and documented. `W_N` is
  per-class — the summed weight of the members whose label space contains the
  class — so a class only one member knows is *not* penalized for members
  that could never have confirmed it (consistent with the per-class vote
  cap); when label spaces are identical this reduces to all members, and
  with unit weights the paper's `min(T, N) / N` is recovered exactly.
- Fusion quality depends on checkpoint `names` metadata: a member carrying
  placeholder names (`class_0`, …) builds a disjoint union with every other
  member — the construction warning fires loudly, no cross-member fusion
  happens, and `min_votes` degrades to its per-class cap. Fix the member's
  `names` rather than the ensemble.

## Fusion ops (`libreyolo.ops`)

| op | semantics | notes |
|---|---|---|
| `weighted_boxes_fusion` | sequential WBF: confidence-sorted clustering against the running fused boxes, confidence-weighted coordinate averaging, `min(W_T, W_N)/W_N` rescale, `conf_type` ∈ {avg, max} | eager default, paper-faithful |
| `wbf_seeded` | parallel one-pass WBF: class-aware NMS picks seeds, candidates assign to their best seed, same cluster reduction | fixed-shape tensor math → the traceable fast path |
| `nms_fusion` | concat + class-aware NMS; per-member weights rank the suppression, survivors keep original scores | trivially explainable; no vote counting |

All ops share one signature — stacked `(boxes, scores, labels, model_ids)`
tensors in, fused tensors out — are scale-invariant and device-agnostic, and
are registered in a `FUSIONS` dict that `LibreEnsemble(fusion=...)` resolves
strings against. `min_votes` is implemented inside both WBF variants from
cluster membership; `nms` + `min_votes > 1` raises.

The WBF algorithm is implemented from the method description in Solovyev et
al., "Weighted boxes fusion: Ensembling boxes from different object detection
models" (arXiv:1910.13302). The sequential and seeded variants agree on
unambiguous clusters and may differ slightly on overlapping cluster chains;
both are documented and tested.

## Phases

- **Phase 1 (this ADR, implemented):** `libreyolo/ops/fusion.py`,
  `libreyolo/ensemble/`, stub-member unit tests, lazy exports. Zero changes
  to model families, no new dependencies, nothing imported unless used.
- **Phase 2 (fast follows):** `ensemble.val()` (the measured-mAP number that
  justifies the N× latency), video/stream, CLI comma-list models, a
  `class_map` hook for name aliases ("person" vs "pedestrian").
- **Phase 3:** baked single-file ONNX ensemble — one input, one
  `(1, max_det, 6)` output in union label ids, indistinguishable from a
  single exported detector with embedded suppression. The graph compiles
  `fusion="wbf"` to the seeded variant.
- **Phase 4 (decoupled, own ADR):** consolidate the library's duplicated
  batched-NMS merge blocks into `libreyolo/ops`.

## Consequences

Positive: cross-architecture ensembling behind one `predict()`, consensus
voting as an API, all families and exported backends become members on day
one, and `libreyolo.ops` starts the standalone-primitives surface.

Accepted costs: N× inference (surfaced via `Results.speed`); two WBF
variants to document honestly; score calibration across heterogeneous
families remains the user's judgment call (mitigations: per-member `conf`,
`weights`); `libreyolo.ops` becomes public API surface requiring semver
discipline.
