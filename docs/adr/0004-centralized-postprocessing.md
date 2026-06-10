# ADR 0004 — Centralized postprocessing package

- Status: Accepted
- Date: 2026-06-11
- Scope: `libreyolo/postprocess/`, `libreyolo/models/*/utils.py`,
  `libreyolo/models/ec/postprocess.py`, `libreyolo/utils/general.py`

## Context

Inference postprocessing (decode, candidate selection, NMS, coordinate
scaling, output-dict packing) lived inside each family's
`models/<family>/utils.py` (plus an inline method for RT-DETR and
`utils/general.py` for the tail shared by YOLOX/RTMDet). The code was
heavily duplicated and hard to locate; new families started from
copy-paste.

## Decision

1. All family postprocessing lives in `libreyolo/postprocess/`, one
   module per family, plus `common.py` for the genuinely shared
   detection tail (`postprocess_detections`).
2. The move is behavior-preserving: code is relocated verbatim, numeric
   divergences between families are kept, and per-family semantics are
   NOT unified. Any semantic consolidation is a separate, explicit
   decision.
3. Old import paths remain valid: `models/<family>/utils.py` (and
   `models/ec/postprocess.py`, `utils/general.py`) re-export the moved
   symbols, so external code and tests keep working. Note that the
   functions resolve their globals in `libreyolo.postprocess.<family>`,
   so monkeypatching module attributes must target the new package, not
   the shims.
4. Import-direction rule: modules in `libreyolo/postprocess/` must not
   import from `libreyolo.models` at module level.
   `models/__init__.py` eagerly imports every model class and model
   modules import from this package, so the reverse edge would be
   circular. Helpers that live under `libreyolo.models` (the DETR
   `box_cxcywh_to_xyxy`) are imported inside function bodies.
5. Preprocessing, checkpoint unwrapping, and model-side helpers stay
   family-local in `models/<family>/` — only postprocessing is
   centralized.

This supersedes the family-local postprocess layout that ADR 0001
prescribed for new multi-task families: new families now add
`libreyolo/postprocess/<family>.py` instead of
`models/<family>/postprocess.py`.

## Consequences

- New families implement postprocessing in the central package and get
  discoverability and side-by-side comparison for free.
- The inference backends (`backends/`), the export graph
  (`export/nms.py`), and validation are intentionally untouched; they
  consume the same re-exported symbols as before.
- First call of the DETR-lineage postprocess in a process that never
  imported `libreyolo.models` triggers the model-registry import inside
  that call (lazy-import consequence of rule 4).
