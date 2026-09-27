# ADR 0015: Generalize the Embed Task

## Status

Accepted. This ADR amends ADR 0017 (the embed task contract, numbered 0013
until a post-1.5.0 renumbering resolved a collision); it does not replace the
face-recognition region contract defined there.

## Context

ADR 0017 chose the generic canonical task name `embed`, but its first
implementation produced only face-region identity vectors. The same primitive
also serves whole-image similarity and image/text retrieval: a float32,
L2-normalized row whose dot product with another row measures agreement.

Keeping these uses under one task lets galleries, result serialization, and
similarity arithmetic remain modality-agnostic. The row's alignment defines
its meaning; a separate task name does not.

## Decision

`embed` has three result shapes:

1. **Whole image:** `Results.embeddings` is `(1, D)` and
   `Results.boxes is None`. CLIP and SigLIP2 image towers provide vectors with a
   paired text space; DINOv2 provides image-only features.
2. **Region:** `Results.embeddings` is `(N, D)` and row-aligned with
   `Results.boxes` under ADR 0001. Face recognition is the first region
   implementation.
3. **Text:** paired image/text families expose
   `model.embed_text(texts) -> (M, D)`. Text returns a tensor, not `Results`,
   because it is not an image prediction source.

Every shape uses float32 unit rows. Whole-image output remains two-dimensional
even for one image; `(D,)` is not a permitted result special case.

Text is a method, never an inferred source modality. A string passed to
`model(...)` or `model.predict(...)` remains a path or URL. LibreYOLO does not
guess that a string is prose.

`Gallery` is the generic named-reference class. It stores each reference row
separately, scores each name by its maximum reference cosine, keeps unknown as
`name=None` below threshold while retaining the best score, and binds persisted
data to an embedding dimension plus weights fingerprint. Matching remains a
dense matrix multiplication. `FaceGallery` is a permanent alias, and legacy
face-gallery archives remain readable.

`BaseModel.embed(source, **kwargs)` runs embed prediction and concatenates all
result rows into `(N_total, D)`. A family without `embed` in
`SUPPORTED_TASKS` raises `NotImplementedError`.

CLIP and SigLIP2 support both `classify` and `embed`; `classify` remains their
default. Their existing `-cls` checkpoint is the shared two-tower artifact and
is loaded with `task="embed"` when vectors are wanted. No duplicate `-embed`
checkpoint is published for identical weights.

DINOv2 embedding bypasses semantic and classification heads and uses the final
normalized CLS token at 224 pixels. Current `n`, `s`, `m`, and `l` variants all
share the DINOv2-S encoder, so each produces `D=384`. DINOv2 has no text tower
and does not expose `embed_text`.

Embedding vectors remain absent from `summary()` and JSON by default.
`summary(embeddings=True)` opts in.

## Alternatives ruled out

- Separate image-embed and text-embed tasks would duplicate one vector
  primitive and split gallery consumers across unnecessary task names.
- Treating text as a source would make an ordinary string ambiguous between a
  path and prose.
- Returning `(D,)` for a whole image would force every consumer to special-case
  the one-row case and break the row-alignment invariant.
- Keeping galleries under the face family would make generic image retrieval
  depend on a biometric model package.

## Consequences

- Existing face-recognition APIs and `FaceGallery` imports remain compatible.
- `gallery=` also identifies whole images, enabling small-scale duplicate and
  reference-image retrieval workflows.
- CLIP/SigLIP2 zero-shot classification and its validation path are unchanged.
- DINOv2 embed vectors are image-only and are tied to the exact checkpoint
  fingerprint.
- A unified retrieval validator is future work. ANN indexes, vector databases,
  clustering, captioning, and general sentence embeddings remain out of scope.
