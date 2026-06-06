# LibreVLM Design Decisions

This document records the user-facing API and internal-contract decisions for
the `LibreVLM` tier, and the reasoning behind them. It is the companion to the
formal contract in [`adr/0002-librevlm-contract.md`](adr/0002-librevlm-contract.md).

## What LibreVLM is

LibreVLM lets you drop in a general vision-language model straight from Hugging
Face and use it as an open-vocabulary object detector. A model here is an
autoregressive chat model: you give it an image and a text prompt, and it
generates text back. When that text is a structured list of boxes, LibreVLM
parses it into the same `Results` object every LibreYOLO model returns.

There is no detection head and no fixed class set. The "vocabulary" is just a
list of words you supply, so any label works ("pink car", "island", "wheel"),
and a new label costs nothing.

This tier is deliberately for *general* VLMs, not for purpose-built open-vocab
detectors. The bet is that general VLMs keep improving at grounding, and that
integrating them as-is keeps LibreYOLO current with that progress.

## Decision 1: two layers, raw chat under a detection convenience

A VLM is fundamentally a chat model that can also draw boxes. The API reflects
that honestly with two layers:

```python
# Layer 1, the raw model:
text = model.chat("image.jpg", "Describe the boats and count them.")

# Layer 2, the detection convenience:
model.set_classes(["boat"])
results = model.predict("image.jpg")     # -> Results(boxes, cls, conf)
```

`predict()` is `chat()` with a canonical detection prompt and a parser bolted
on. Keeping `chat()` first-class means power users are never boxed in by the
detection wrapper: free-form questions, custom output formats, counting, and
reasoning are all one call away. This is the property that makes the tier
future-proof, so it is a first-class method rather than an internal detail.

## Decision 2: `set_classes()` is the open-vocabulary surface, and it is sticky

The vocabulary is set with a method, not baked only into the constructor and not
passed on every `predict()` call:

```python
model = LibreVLM()                      # default model, no vocabulary yet
model.set_classes(["pink car", "wheel"])
model.predict("a.jpg")                   # uses the vocabulary
model.predict("b.jpg")                   # still uses it, no need to repeat
model.set_classes(["person", "dog"])     # change it whenever
```

Rationale:

- It keeps `predict()` signature-compatible with the closed-vocab detectors, so
  the two tiers feel the same from the caller's side.
- The vocabulary is conceptually a property of the configured model, not of a
  single image, so it belongs on the object and should persist.
- `names=[...]` at construction is kept as a convenience that simply calls
  `set_classes` for you.

Per-image, free-form queries are served by `chat()`, which is the right place
for genuinely per-call prompts.

## Decision 3: the output is `Results`, and confidence is honestly soft

The tier returns the standard `Results` (`boxes.xyxy`, `boxes.cls`,
`boxes.conf`, `.plot()`, `.save()`), so folders, video, tracking, and drawing
all work unchanged. No new output type is invented.

But these models emit no calibrated per-box score, so `conf` is a placeholder.
We do not pretend otherwise:

- `conf=` filtering and ranking are soft, not calibrated.
- `val()` (mAP) is intentionally unsupported, because it would be misleading.
- `_score_detections()` is the documented hook for a real signal later (decoder
  token log-probabilities or self-consistency).

This is the honest boundary of the tier: it gives you boxes and labels, not a
calibrated detector contract. For calibrated scores and tight boxes, the
detector families behind `LibreYOLO(...)` remain the right tool.

## Decision 4: each model keeps its own output format; we adapt per family

Every model writes boxes in its own scheme, learned from its training data. We
do not try to force one universal format on them. Instead, each family declares
its convention and the shared parser handles the rest. A family is a small
adapter:

```python
class LibreQwen3VL(LibreVLMModel):
    FAMILY = "qwen3vl"
    FILENAME_PREFIX = "LibreQwen3VL"
    HF_REPOS = {"4b": "Qwen/Qwen3-VL-4B-Instruct", ...}
    INPUT_SIZES = {"4b": 1024, ...}
    BBOX_KEY = "bbox_2d"        # this model's JSON key
    COORD_DIVISOR = 1000.0      # this model's coordinate scale
    # _detection_prompt() overridden to ask in this model's expected style
```

The tolerant parser (`libreyolo/models/vlm/parsing.py`) absorbs the rest of the
variation: markdown fences, prose around the JSON, single quotes, truncated
arrays, duplicate boxes from a generation loop, and out-of-vocabulary labels.

### Always verify the coordinate convention empirically

Documentation is often ambiguous about whether a model emits `[0,1]`, `0-1000`,
or absolute pixels. Before trusting a new family's parser, feed the model a
synthetic image with a known box and read back the numbers. Verified so far:

| Model     | Box key   | Coordinate scale | `COORD_DIVISOR` |
|-----------|-----------|------------------|-----------------|
| Qwen3-VL  | `bbox_2d` | 0-1000           | 1000.0          |
| LFM2-VL   | `bbox`    | [0, 1]           | 1.0             |

## Decision 5: default model and licensing

The default model is **Qwen3-VL-4B** (`LibreVLM()` with no arguments), chosen
because it is the strongest general open-weight VLM that runs on a single
consumer GPU and is **Apache-2.0**, so it is clean for LibreYOLO to ship and
needs no license notice.

Weights autodownload on first inference into `weights/<FILENAME_PREFIX><size>/`.
Note that `FILENAME_PREFIX` here is a weights-directory prefix, not a LibreYOLO
`.pt` checkpoint name: VLM families download Hugging Face repos rather than
emitting `Libre<FAMILY><size>.pt` checkpoints, so the checkpoint-filename
nomenclature does not apply and brand casing (`LibreQwen3VL`) is kept.
Models under non-permissive licenses print a one-time license notice before the
download (following the existing download-notice pattern in the repo); LFM2-VL
is the current example (LFM Open License v1.0, with a revenue threshold).

LibreYOLO contributes no model source code: families load through the Apache-2.0
`transformers` API and do not redistribute weights.

## Adding a new model: checklist

1. Create `libreyolo/models/vlm/<family>.py` subclassing `LibreVLMModel`.
2. Set `FAMILY`, `FILENAME_PREFIX`, `HF_REPOS`, `INPUT_SIZES`.
3. Probe the model on a known box; set `BBOX_KEY` and `COORD_DIVISOR`, and
   override `_detection_prompt()` if its expected ask differs.
4. Add an alias in `libreyolo/models/vlm/__init__.py` (and to the top-level lazy
   exports if it should be importable as `libreyolo.Libre<Name>`).
5. Add a `_LICENSE_NOTICE` only if the weights are non-permissive.
6. Verify with a real inference; the parser and the predict/track surface are
   shared, so there is usually no other code to write.
