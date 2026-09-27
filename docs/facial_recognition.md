# Facial recognition (`embed` task)

Face embedding, 1:1 verification, and 1:N identification. Two-stage and
inference-only: a face detector locates faces and 5 landmarks, each face is
aligned to a canonical 112x112 crop, and an ONNX recognition head emits an
L2-normalized identity embedding. Verification and identification are cosine
similarity on those embeddings.

Face recognition is the region-aligned shape of LibreYOLO's general `embed`
task. Whole-image and paired text embedding use the same vector and gallery
contract without face boxes; see
[`adr/0015-embed-generalization.md`](adr/0015-embed-generalization.md).

Canonical task name: `embed`. Aliases: `facial-recognition`, `face`,
`faceid`, `recognition`, `embedding`.

## Quickstart

```python
from libreyolo import Gallery, LibreYOLO

model = LibreYOLO("librefacerec-l")   # auto-downloads embedder + default detector

# Embeddings
results = model("group.jpg")
results.embeddings                     # (N, 512) unit vectors, row-aligned with boxes

# 1:1 verification
model.verify("a.jpg", "b.jpg", threshold=0.4)
# {'similarity': 0.72, 'same_person': True, 'threshold': 0.4}

# 1:N identification
gallery = Gallery(model)
gallery.enroll("alice", ["alice1.jpg", "alice2.jpg"])
gallery.save("team.gallery.npz")

results = model("group.jpg", gallery=gallery, threshold=0.4)
results.identities.name                # ["alice", None, ...] (None = unknown)
results.identities.score               # best gallery cosine per face
```

`FaceGallery` remains a permanent alias of `Gallery`, including the legacy
`libreyolo.models.facerec` import path.

```bash
libreyolo compare model=facerec-l source=a.jpg source2=b.jpg
libreyolo enroll  model=facerec-l source=people/ gallery=team.gallery.npz
libreyolo predict model=facerec-l source=group.jpg gallery=team.gallery.npz save=true
```

`enroll` expects a folder-per-person tree (`people/<identity>/*.jpg`, folder
name becomes the identity), the same convention as classification datasets.

## Models

| Name | Role | Dim | Weights license | Notes |
|---|---|---|---|---|
| `librefacerec-l` | embedder (default) | 512 | Apache-2.0 | iResNet100, mirrored from AuraFace-v1; LFW dev-test ROC-AUC 0.980, 96.9% accuracy through this pipeline |
| `librefacerec-det` | default detector | - | MIT | YuNet with 5 landmarks, runs on OpenCV's bundled `cv2.FaceDetectorYN` |

Weights carry their own licenses on their Hugging Face model cards; review
them for your use case.

### Bring your own weights

Any ArcFace-convention ONNX (aligned 112x112 RGB in, `(N, D)` embeddings
out) loads directly:

```python
model = LibreYOLO("path/to/recognition.onnx", task="facial-recognition")
```

This covers third-party recognition heads whose licenses do not allow
LibreYOLO to redistribute them (for example InsightFace `w600k_r50.onnx`
from the buffalo_l pack, or OpenCV SFace with `preproc="raw_bgr"`). The
detector is also swappable: pass `face_detector=` any LibreYOLO detector, an
OpenCV face-detector ONNX path, or a callable; or bypass detection entirely
with `face_boxes=[...]`.

## Design notes

- **Per-reference storage, max-cosine scoring.** Enrolling K images of one
  person keeps K vectors; an identity's score is the best cosine over its
  references. References survive pose and age variance better than a
  centroid.
- **Unknown is a first-class outcome.** Faces below the threshold get
  `name=None`, never the nearest wrong person. The raw best score stays
  visible in `identities.score`.
- **Galleries are bound to their embedder.** `save()` records the embedding
  dim and a fingerprint of the model file; matching against a different
  model raises instead of silently comparing incompatible vector spaces.
- **Brute-force matching only.** Thousands of identities cost one matmul.
  For larger scale, export `results.embeddings` to a dedicated vector
  store.
- **Thresholds are model-specific.** Measured on the LFW dev pairs with
  `librefacerec-l` (threshold picked on the train split, scored on the
  held-out test split): ROC-AUC 0.980, 96.9% accuracy at threshold 0.227,
  95.6% at the 0.4 default. No different-person pair scored above 0.30 in
  that split, so 0.4 is a conservative default that trades recall for
  near-zero false accepts; lower it toward 0.3 when missed matches cost
  more than false ones. Recalibrate for other embedders and for your
  population, and note LFW is a saturated, frontal, celebrity-photo
  benchmark where leading models report roughly 99.5% and above.
- Training, validation, and export raise `NotImplementedError`: like gaze,
  this is an inference product consuming opaque ONNX graphs.

The face-region contract is recorded in
[`adr/0017-embed-task-contract.md`](adr/0017-embed-task-contract.md) and amended
by the general three-shape contract in
[`adr/0015-embed-generalization.md`](adr/0015-embed-generalization.md).

## Responsible use

Face embeddings are biometric identifiers. The task is intended for
consent-based applications: device unlock, photo-library organization,
opt-in event galleries. Remote biometric identification of people in public
spaces is prohibited or heavily restricted in several jurisdictions (EU AI
Act, BIPA and similar state laws), and compliance is the deployer's
responsibility. Recognition accuracy varies across demographics; calibrate
thresholds on data representative of your deployment.
