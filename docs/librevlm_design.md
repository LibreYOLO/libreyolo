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

## Available models

Pass any of these names to `LibreVLM(name)`. A bare family name resolves to the
size marked with `*`. The authoritative list is `_ALIASES` in
`libreyolo/models/vlm/__init__.py`, and passing an unknown name raises a
`ValueError` listing every alias.

| Aliases                                      | Family    | License             | Notes                                    |
|----------------------------------------------|-----------|---------------------|------------------------------------------|
| `qwen3-vl`, `-2b`, `-4b`*, `-8b`             | Qwen3-VL  | Apache-2.0          | default model; strongest detector here   |
| `lfm2-vl`, `-450m`*, `-1.6b`                 | LFM2-VL   | LFM Open License v1.0 | edge VLM; non-permissive, notice-gated  |
| `internvl3`, `-1b`, `-2b`*, `-8b`            | InternVL3 | Qwen License        | Qwen-backbone weights, notice-gated; weak small |
| `florence-2`, `-base`*, `-large`             | Florence-2 | MIT                | small purpose-built detector; tight boxes |
| `kosmos-2`*                                   | Kosmos-2  | MIT                 | 2023 grounder; loads clean, coarse boxes |
| `smolvlm2`, `-2.2b`*, `-500m`                | SmolVLM2  | Apache-2.0          | tiny; weak detector, zero-code family    |

Florence-2 and Kosmos-2 do not use a chat template: they are driven by task /
grounding prompts and decode boxes via the processor's `post_process_generation`,
so their families override the three inference hooks (and Florence-2's boxes come
back in pixels, no scaling needed). Use the `florence-community/*` Florence-2
checkpoints; the original `microsoft/*` ones do not load on current transformers.

Larger Qwen3-VL tiers (30B and up) and Qwen2.5-VL are not included: the big ones
do not fit a single consumer GPU, and Qwen2.5-VL uses a different coordinate
convention that would need its own family. Some strong models are deliberately
left out for being remote-code (Ovis2.5, MiniCPM-V, Moondream2, Molmo2; fragile
on current transformers), gated (PaliGemma2, Gemma3), too large for ~16 GB
(GLM-4.1V-9B), or not a clean drop-in (Rex-Omni crashes on the standard Qwen2.5-VL
path despite being the strongest generative detector). `LibreVLM()` defaults to
`qwen3-vl-4b`. Detection quality varies a lot by family and size; Qwen3-VL,
LFM2-VL, and Florence-2 are the strong ones.

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

`chat()` applies to the chat-template families (Qwen3-VL, LFM2-VL, SmolVLM2,
InternVL3). The task-prompt families (Florence-2, Kosmos-2) are not chat models:
they are driven by fixed task / grounding tokens, so their `chat()` raises
`NotImplementedError` and only `predict()` is supported. `predict()` (the
detection layer) works on every family.

## Decision 2: `set_classes()` is the open-vocabulary surface, and it is sticky

The vocabulary is set with a method, not baked only into the constructor and not
passed on every `predict()` call:

```python
model = LibreVLM()                      # default model, vocabulary defaults to COCO-80
model.set_classes(["pink car", "wheel"])
model.predict("a.jpg")                   # uses the vocabulary
model.predict("b.jpg")                   # still uses it, no need to repeat
model.set_classes(["person", "dog"])     # change it whenever
```

If you never call `set_classes`, the vocabulary defaults to the COCO-80 labels
(like the detector tier), so a bare `predict()` asks the model for the 80 COCO
classes. `set_classes` replaces that with your own open-vocabulary list.

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

| Model      | Box key   | Scale  | Layout | Knobs                                   |
|------------|-----------|--------|--------|-----------------------------------------|
| Qwen3-VL   | `bbox_2d` | 0-1000 | xyxy   | `COORD_DIVISOR=1000`                    |
| LFM2-VL    | `bbox`    | [0, 1] | xyxy   | defaults                                |
| SmolVLM2   | `bbox`    | [0, 1] | xyxy   | defaults                                |
| InternVL3  | `bbox`    | 0-1000 | xyxy   | `COORD_DIVISOR=1000` + flatten override |

Three knobs cover the output variation without touching the parser:

- `BBOX_KEY` : the JSON key holding the box (`bbox`, `bbox_2d`, ...).
- `COORD_DIVISOR` : the numeric scale (1.0 for [0,1], 1000.0 for 0-1000).
- `BOX_FORMAT` : the box layout, `xyxy` (default), `xywh`, or `cxcywh`.

A model that already emits the default shape (a `bbox` key, [0,1], xyxy, through
a chat template) needs no code at all; SmolVLM2 is such a case and its family is
just a repo table. A model that only differs by scale/key/layout sets the knobs
(Qwen3-VL). A model whose output shape is genuinely different overrides a hook:
InternVL3 wraps each object's boxes in an extra list, so its family overrides
`_postprocess` to flatten before the shared builder runs; a grounding-token
model whose boxes come from a processor `post_process_generation` call overrides
`_preprocess`/`_forward`/`_postprocess` (Florence-2 and Kosmos-2 do exactly this).
The tier ships six distinct families (Qwen3-VL, LFM2-VL, SmolVLM2, InternVL3,
Florence-2, Kosmos-2) spanning all three integration styles, confirming it is
genuinely model-agnostic. Detection *quality* varies a lot by model and size
(Qwen3-VL, LFM2-VL, and Florence-2 are the strong ones); the framework is what is
general, not every model's accuracy.

## Decision 5: default model and licensing

The default model is **Qwen3-VL-4B** (`LibreVLM()` with no arguments), chosen
because it is the strongest general open-weight VLM that runs on a single
consumer GPU and is **Apache-2.0**, so it is clean for LibreYOLO to ship and
needs no license notice.

Weights autodownload on construction (first use) into
`weights/<FILENAME_PREFIX><size>/`, resolved relative to the current working
directory, matching LibreYOLO's existing `weights/` convention. Note that
`FILENAME_PREFIX` here is a weights-directory prefix, not a LibreYOLO `.pt`
checkpoint name: VLM families download Hugging Face repos rather than emitting
`Libre<FAMILY><size>.pt` checkpoints, so the checkpoint-filename nomenclature
does not apply and brand casing (`LibreQwen3VL`) is kept.
Models under non-permissive licenses log a one-time license notice before the
download (following the existing download-notice pattern in the repo); LFM2-VL
(LFM Open License v1.0, with a revenue threshold) and InternVL3 (Qwen License, on
its `-hf` weights) are the current examples.

LibreYOLO contributes no model source code: families load through the Apache-2.0
`transformers` API and do not redistribute weights.

## Known limitations (v1)

These are deliberate v1 scoping choices, called out so behavior matches expectations:

- **Confidence is synthetic.** Every box is scored `1.0` (no calibrated per-box
  score from a generative model). `conf=` filtering works mechanically but is not
  calibrated, and in `track()` this makes ByteTrack's two-stage, score-stratified
  association inert (no low-confidence recovery stage). `val()`/mAP is unsupported.
- **`batch=` does not speed up VLMs.** `predict("folder/")` works, but generation
  runs one image at a time, so a larger `batch=` gives no throughput gain in v1.
- **Python-API only.** The `libreyolo` CLI does not resolve VLM aliases yet; use
  `LibreVLM(...)` from Python. `predict`/`track` parity is at the API level.
- **`chat()` and `prompt=`** apply to the chat-template families only; Florence-2
  and Kosmos-2 are task-token driven (`chat()` raises, `prompt=` is ignored).

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
