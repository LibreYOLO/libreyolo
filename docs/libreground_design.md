# LibreGround Design Decisions

Companion to [`adr/0020-libreground-contract.md`](adr/0020-libreground-contract.md).

## What LibreGround is

`LibreGround` is the factory for models whose job is: screenshot (or photo)
plus a referring expression, out comes a click. It is the grounding
sibling of `LibreVLM`. The public primitive is `Results.points`, the same
payload FOMO and LocateAnything already use.

It is not a computer-use agent. It does not move a mouse. It does not
plan. An agent (GLM, Claude, Grok, a script) calls it and clicks.

## Why not a new task

A task is an output contract (`docs` + `tasks.py`). Grounding emits
`x, y, class, confidence`. That is `point`. Adding `gui` would be naming
the domain.

## Why not LibreVLM

| | `LibreVLM` | `LibreGround` |
|---|---|---|
| Input | sticky class list | per-call instruction |
| Output | every instance, boxes | one click per query, points |
| Default vocab | COCO-80 | none; query is required |
| Typical prompt | "detect all boats" | "Bluetooth" |

A grounder stuffed into `set_classes` would hide the thing users actually
type.

## The two layers

```python
from libreyolo import LibreGround

# Layer 1 — the one-liner agents want
r = LibreGround()("screen.png", prompt="Bluetooth")
x, y = r.points.xy[0].tolist()

# Layer 2 — sticky query, YOLO-shaped predict
model = LibreGround("showui-2b")
model.set_query("the submit button")
model.predict("folder/")          # same Results as any point model
model.chat("screen.png", "...")   # raw text, chat-template families only
```

`prompt=` and `query=` are the same argument. `set_classes` forwards to
`set_query`.

A list of queries on **one** image runs one generate per query and merges
points (class id = query index). A list of queries on a folder or image
list raises: that would be too magical.

## Available models

Pass any alias to `LibreGround(name)`. A bare family name resolves to the
size marked `*`. The authoritative table is `_ALIASES` in
`libreyolo/models/ground/__init__.py`.

| Aliases | Family | License | Coord space | Notes |
|---|---|---|---|---|
| `showui`, `showui-2b`* | ShowUI-2B | MIT weights; Apache-2.0 code/base | `[0,1]` | default; 2B Qwen2-VL |
| `florence-2`, `-base`*, `-large` | Florence-2 | MIT | pixel boxes → center | `FAMILY=ground_florence2` |
| `qwen3-vl`, `-2b`*, `-4b`, `-8b` | Qwen3-VL | Apache-2.0 | 0–1000 | `FAMILY=ground_qwen3vl` |

Default is **ShowUI-2B**: small enough for a consumer GPU, native
`transformers`, documented `[0,1]` clicks. TinyClick, Holo, UI-TARS,
LocateAnything, and Moondream are not factory aliases until they load and
satisfy the one-click contract. Moondream 2 was removed after a ten-image
click-in-box probe showed center-biased points rather than reliable grounding.

A per-call `prompt=` does not become sticky. Coordinates that fall well
outside the image after scaling are dropped, not clamped into the frame.

## Coordinate knobs

Each family declares `COORD_SPACE`:

- `unit` — model emits `[0,1]`
- `milli` — model emits `0–1000`
- `pixel` — model emits original-image pixels
- `pixel_view` — model emits pixels on the processor's resized view. The
  adapter records that view size in `_preprocess` and scales back to the
  original canvas.

The valid `unit` and `milli` upper endpoints are the continuous right and
bottom canvas edges. They snap to the final pixel; values more than half a
pixel beyond an edge are dropped.

Always verify empirically with a synthetic screenshot before trusting a
new family. The shared parser (`libreyolo/models/ground/parsing.py`) is
deliberately sloppy about *syntax* (`Click(x,y)`, `<point>`, JSON, bare
`[x, y]`, Florence loc tokens, box-as-center) and strict about *scaling*
(the family knob).

## Adding a family

1. Subclass `LibreGroundModel`.
2. Set `FAMILY`, `FILENAME_PREFIX`, `HF_REPOS`, `INPUT_SIZES`, `COORD_SPACE`.
3. Override `_format_grounding_prompt` if the model was trained on a
   specific ask.
4. Pin `HF_REVISIONS` if `TRUST_REMOTE_CODE` is on.
5. Add aliases in `__init__.py`.
6. Probe a known click; add a parser unit test if the syntax is new.

A model that already emits `[x, y]` in `[0,1]` through a chat template
needs almost no code (ShowUI is that case plus a system sentence).

## Known limits (v1)

- Confidence is a placeholder (`1.0`).
- No `train` / `val` / `export`.
- No batched generate. A folder is one forward per image.
- High-resolution professional UIs (ScreenSpot-Pro) are hard for every
  small grounder. Do not promise pixel-perfect IDE clicks.
- Python API only, same as `LibreVLM` v1.
