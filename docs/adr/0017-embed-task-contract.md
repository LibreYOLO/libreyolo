# ADR 0017: Embed (Facial Recognition) Task Contract

## Status

Accepted. The face-region contract remains in force and is generalized to
whole-image and paired-text embeddings by ADR 0015.

## Context

Face verification ("are these two photos the same person?") and identification
("who is this, out of the people I enrolled?") are among the most requested
vision capabilities, and no existing task carries them. The output primitive
that serves both is an identity embedding: a unit-length vector per face whose
cosine similarity with another vector measures identity agreement.

No existing contract fits it. `detect` and `pose` describe where things are,
not who they are. `classify` assigns a fixed label set, but identity is an
open set: the people a deployment cares about are enrolled at runtime, not at
training time, and a model that could only recognize its training identities
would be useless. An embedding is a distinct output primitive, so a new task is
warranted (the same test that promoted `point` in ADR 0003, `depth` in ADR
0006, and `matte` in ADR 0010).

The task is two-stage by nature: a detector locates faces and five landmarks,
each face is warped onto a canonical 112x112 template, and a recognition head
emits the embedding. This mirrors the `gaze` (L2CS) shape, and reuses its
face-detector protocol rather than inventing a second one.

Recognition heads are consumed as opaque ONNX graphs. No third-party
architecture code is ported, and the family carries no training path, which is
the accepted single-purpose exception to "new features cover the flagships"
(as with `gaze` and `matte`).

## Decision

LibreYOLO defines a canonical `embed` task (suffix `-embed`; aliases
`facial-recognition`, `face-recognition`, `recognition`, `face`, `faceid`,
`embedding`, `reid`). The single-word canonical name matches the other task
names and the short filename suffix; the human-facing spellings resolve to it.

Its prediction primitive is `Results.embeddings`: an `(N, D)` float32 payload
of L2-normalized rows, row-aligned with `Results.boxes` (the face boxes), where
`D` is the head's embedding dimension. Cosine similarity is therefore a dot
product, and verification is a threshold on it.

Identification is arithmetic on that primitive rather than a second model
output. `FaceGallery` holds named reference embeddings and is matched against
query embeddings to produce `Results.identities`: an `(N,)` payload of matched
names and scores, row-aligned with the boxes.

Contract decisions worth stating, because each rules out a plausible
alternative:

- **Per-reference storage, max-cosine scoring.** Enrolling K images of a person
  stores K vectors and the identity scores as the best cosine over them.
  Averaging references into a centroid discards pose, age, and lighting
  variance, which is precisely the variance multiple references exist to cover.
- **Unknown is a first-class outcome.** Below threshold, `identities.name[i]`
  is `None`, never the nearest enrolled person. The best score stays visible in
  `identities.score[i]` so callers can inspect a near miss. A recognition API
  whose worst case is confidently naming the wrong human is not acceptable.
- **Galleries are bound to their embedder.** `save()` records the embedding
  dimension and a fingerprint of the weights file; matching against a different
  model raises rather than silently comparing incompatible vector spaces.
- **Brute-force matching only.** Cosine against a few thousand identities is a
  single matmul. No ANN index and no vector-store dependency: callers at larger
  scale export `results.embeddings` to their own store.
- **Thresholds are model-specific and documented, not hidden.** The shipped
  default is stated with the benchmark it came from, because a threshold
  transplanted between embedders is meaningless.

Embedding vectors are omitted from `summary()` and JSON output by default (a
512-float vector is roughly 2 KB per face); `summary(embeddings=True)` opts in.

Out of scope in v1: training, dataset validation, and re-export
(`model.train()`, `.val()`, `.export()` raise), because the family wraps an
existing ONNX graph. Also out of scope: face attribute prediction (age, gender,
emotion), which is a different ethical surface, and clustering of unlabeled
collections, which callers can do from the raw embeddings.

## Consequences

- Embed results always carry face boxes row-aligned with the embeddings, so the
  ADR 0001 alignment rule holds and `Results.__getitem__` slices both together.
- Any ArcFace-convention ONNX (aligned 112x112 in, `(N, D)` out) works as a
  bring-your-own-weights model, which covers recognition heads whose licenses
  do not permit LibreYOLO to redistribute them.
- The detector is swappable: any LibreYOLO detector, a callable, or an OpenCV
  face-detector ONNX, and detection can be bypassed entirely with `face_boxes=`.
  A default detector is auto-downloaded when none is supplied.
- Hosted weights state their license and training-data provenance on the model
  card, and accuracy claims on those cards must be reproducible from a
  documented protocol rather than asserted.
- The task documentation carries a responsible-use section: these are biometric
  identifiers, the intended uses are consent-based, and remote biometric
  identification is restricted in several jurisdictions. That is a
  documentation obligation, not a runtime gate.
