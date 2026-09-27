# ADR 0020: LibreGround Contract For GUI / Referring Grounding

- Status: Accepted
- Date: 2026-08-16
- Scope: New model tier (instruction → click point). Not a new task.

## Context

Agents that cannot see (or should not spend a frontier VLM call on every
screenshot) need a cheap local answer to: "where do I click?". That is GUI
grounding / referring-expression pointing: image + instruction in, one
coordinate out.

LibreYOLO already has the output contract. ADR 0003 defines `point` as
`x, y, class, confidence` on the original image canvas. LocateAnything and
FOMO already emit `Results.points`. A new `ground` / `gui` / `click` task
would name the *domain*, not the output, the same mistake as making `deblur`
a task instead of `restore`.

The I/O is not `LibreVLM` either. VLM-as-detector is a sticky vocabulary
(`set_classes`) that returns every instance as a box. A grounder takes a
per-call referring expression and returns a click. Same reason Grounding
DINO is not `LibreVLM`.

## Decision

Add a sibling factory `LibreGround`, inference-only, Hugging Face snapshots,
same weight-acquisition machinery as `LibreVLM`. The line is the input
contract, not the architecture:

- `LibreYOLO(...)` — closed-set detector, real scores, `.pt` checkpoints.
- `LibreVLM(...)` — generative detector, sticky class list, `Results.boxes`.
- `LibreOpenVocab(...)` — discriminative text-conditioned boxes.
- `LibreGround(...)` — instruction → `Results.points` (click on the
  original canvas).

`task` stays `point`. No new row in `tasks.py`. No filename suffix change.

## Public API

```python
from libreyolo import LibreGround

model = LibreGround()                              # ShowUI-2B default
r = model.predict("screen.png", prompt="Bluetooth")
r.points.xy                                        # [[x, y]] pixels
r.points.xyn                                       # normalized

model.set_query("the red Save button")             # sticky, like set_classes
r = model.predict("screens/")                      # one query, many images

r = model.predict("screen.png", prompt=["Wi-Fi", "Bluetooth"])
# one row per query; r.names maps class id → query string
```

`query=` is an alias of `prompt=`. `set_classes` forwards to `set_query` so
VLM muscle memory still works. There is no default COCO vocabulary: a call
without a query raises.

`chat()` remains the escape hatch on chat-template families.

Coordinates are always original-canvas pixels on `Results.points`. Families
that emit `[0,1]`, `0-1000`, or resized-view pixels convert in the adapter.

## Confidence, train, val, export

Same honesty as ADR 0002: generated points carry a placeholder score of
`1.0`. `train()`, `val()`, and `export()` raise. ScreenSpot click-in-box is
the right future metric; COCO mAP is not.

## Out of scope

- Moving the OS mouse. The library returns coordinates.
- Training a UI-element detector or a native GUI VLM.
- MCP / set-of-mark overlay (a later agent-surface PR).
- A new `ground` task or `Results.clicks`.

## Licensing

Same rules as LibreVLM: pin `HF_REVISIONS` to a commit SHA when
`trust_remote_code` is on; log a notice for non-permissive weights; do not
redistribute upstream repos. Default model (ShowUI-2B) weights are MIT on
the pinned Hugging Face card; the ShowUI GitHub repository and the
Qwen2-VL-2B base are Apache-2.0. Ground wrappers that share an upstream
with a LibreVLM family use a distinct `FAMILY` id (`ground_florence2`,
not `florence2`) so they do not replace the VLM inventory row.
