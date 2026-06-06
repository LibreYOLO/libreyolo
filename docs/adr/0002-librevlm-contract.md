# ADR 0002: LibreVLM Contract For Vision-Language Detectors

- Status: Proposed
- Date: 2026-06-05 (updated 2026-06-06)
- Scope: New model tier (vision-language models used as open-vocab detectors)

## Context

LibreYOLO's detector families are loaded by the `LibreYOLO(...)` factory, which
sniffs a `.pt` state dict (`can_load`), detects size from keys, and runs a
single forward pass producing calibrated `(boxes, scores, classes)`. RF-DETR
proves a transformer detector fits this path.

Vision-language models (Qwen3-VL, LFM2-VL, and others) do not fit it:

- They are multi-file Hugging Face repos, not a single sniffable state dict.
- They are autoregressive: image plus text prompt in, generated text out. A box
  is text the model types, not a tensor from a detection head.
- They have no per-box confidence, and the class list is open vocabulary (any
  words), not a fixed head.

Forcing them through `LibreYOLO(...)` would misrepresent what `conf` means and
the latency profile. But the boxes a VLM produces are exactly what LibreYOLO
already renders, so the user-facing experience can and should stay familiar.

## Decision

Add a parallel tier, `LibreVLM`, for generative open-vocabulary models. The line
is drawn on **contract fidelity**, not architecture:

- Faithful detector (real scores, closed-set-able, single forward) stays in the
  `LibreYOLO(...)` factory. This includes transformer detectors, exactly as
  RF-DETR already does.
- Generative VLM (soft confidence, prompt-driven, open vocabulary) is loaded by
  `LibreVLM(...)`.

Both return the same `Results`, so downstream code is unchanged. They are
separated because their *contract* differs, not because the network differs.

The model defaults to **Qwen3-VL-4B** (Apache-2.0), autodownloaded on first use.

## Public API

Two layers, intentionally:

1. The raw model (`chat`): the honest truth, an image-plus-text chat model.
2. The detection convenience (`set_classes` + `predict`/`track`): a cached
   detection prompt and a per-family parser, returning `Results`.

```python
from libreyolo import LibreVLM

model = LibreVLM()                          # Qwen3-VL-4B by default, autodownloads
model.set_classes(["pink car", "wheel"])    # open vocabulary, sticky, any words
result = model.predict("image.jpg")          # same Results as a YOLO model
results = model.predict("folder/")           # folders, video, stream, track()
result.boxes.xyxy        # pixel xyxy
result.boxes.cls         # ids into the vocabulary set above
result.plot(); result.save()

text = model.chat("image.jpg", "How many cars are pink?")  # raw escape hatch
```

- `set_classes(labels)` is the primary way to set the vocabulary. It is sticky:
  set once, reused by every later `predict()`/`track()` until set again. This
  keeps `predict()` signature-compatible with the closed-vocab detectors.
- `names=[...]` at construction is a convenience that calls `set_classes` for you.
- `chat(image, prompt)` exposes the underlying model for anything the detection
  wrapper does not cover (free-form questions, custom formats, counting).
- `prompt="..."` overrides the detection prompt; `max_new_tokens`, `device` as usual.

## Internal Contract

`LibreVLMModel(BaseModel)` is the shared base. It does NOT define `can_load`, so
`BaseModel.__init_subclass__` never registers VLM families into the detector
`_registry`; they stay out of the weight-sniffing factory.

To support a new model, subclass it and declare the adapter:

| Field             | Meaning                                                  |
|-------------------|----------------------------------------------------------|
| `FAMILY`          | family id (e.g. `qwen3vl`)                               |
| `FILENAME_PREFIX` | `Libre`-prefixed display name and weights dir prefix     |
| `HF_REPOS`        | `{size: hf_repo_id}`; drives autodownload                |
| `INPUT_SIZES`     | `{size: nominal_px}`; nominal, the processor owns resize |
| `_detection_prompt()` | how to ask THIS model for boxes (override if needed)  |
| `BBOX_KEY`        | JSON key holding the box (`bbox`, `bbox_2d`, ...)        |
| `COORD_DIVISOR`   | scale of the coords (1.0 for [0,1], 1000.0 for 0-1000)  |
| `_LICENSE_NOTICE` | text printed once before the first download (if needed)  |

The base implements the predict/track surface by satisfying the four hooks the
shared `InferenceRunner` drives:

- `_get_input_size()` returns the nominal `imgsz`.
- `_preprocess(image, ...)` builds the chat-template inputs from the image plus
  the detection prompt; returns `(inputs, pil_image, (W, H), ratio=1.0)`. Boxes
  come back normalized to the image, so there is no letterbox/unpad math.
- `_forward(inputs)` runs `model.generate(...)` greedily and returns only the
  newly generated tokens.
- `_postprocess(output, conf, ...)` decodes, tolerantly parses the JSON, scales
  the coordinates per `BBOX_KEY`/`COORD_DIVISOR`, and returns the standard
  detection dict `{boxes, scores, classes, num_detections}` that
  `InferenceRunner._wrap_results` converts to `Results`.

Parsing lives in `libreyolo/models/vlm/parsing.py` (pure, unit-tested offline):
it tolerates markdown fences, prose, single quotes, and truncated arrays; clamps
boxes and orders corners; dedupes identical boxes (a generative loop can repeat
one box); and maps labels case-insensitively to class ids, dropping
out-of-vocabulary labels. That label mapping is what makes an open-vocab
generator behave as a closed-set detector against `set_classes`.

### Coordinate conventions differ per model

Each model writes boxes in its own scheme, learned from its training labels, so
the convention must be verified empirically (feed a known box, read the output)
and declared via `BBOX_KEY`/`COORD_DIVISOR`. The verified per-model table lives
in [`../librevlm_design.md`](../librevlm_design.md).

## Confidence

Generated detections carry no calibrated per-box score. The tier assigns a
constant placeholder (`DEFAULT_SCORE = 1.0`), so `predict`/draw/`track` behave
normally and `conf=` filtering still functions mechanically. Consequences:

- `conf=` thresholds and ranking are soft, not calibrated.
- `val()` (mAP) is intentionally unsupported; it would be misleading.

`_score_detections(items)` is the documented override point for a real signal
(decoder token log-probs or self-consistency) in a later iteration.

## Licensing

LibreYOLO ships no VLM source code: families load through the Apache-2.0
`transformers` API and do not redistribute weights. The default model
(Qwen3-VL-4B) is Apache-2.0, so it needs no notice. When a model's weights are
under a non-permissive license (for example LFM2-VL, under the LFM Open License
v1.0 with a revenue threshold), the download is gated behind a one-time printed
license notice, following the existing download-notice pattern in
`libreyolo/utils/download.py` and `libreyolo/models/l2cs/model.py`.

## Out Of Scope (v1)

- Training / fine-tuning (`train()` raises; fine-tune upstream).
- Dataset validation / mAP (`val()` raises; see "Confidence").
- Export to ONNX/TensorRT/etc. (`export()` raises; generative decode).

## Consequences

### Positive

- Open-vocabulary, zero-setup detection behind a familiar predict/track surface.
- No change to the detector factory; VLM families are fully isolated.
- A new model is a small adapter class (repos, prompt, coordinate convention).

### Negative

- Confidence is synthetic until the log-prob path lands.
- Generation is slower and less deterministic than a detector forward.
- Adds `transformers` (already an optional extra) to the `vlm` extra.

## Implementation Status

- `LibreVLMModel` base with `set_classes()` and `chat()`.
- `LibreQwen3VL` (Qwen3-VL 2B/4B/8B, Apache-2.0) as the default family.
- `LibreLFM2VL` (LFM2.5-VL 450M/1.6B, license-gated) as a second family.
- Offline parser unit tests plus a `vlm`-marked end-to-end smoke test.
