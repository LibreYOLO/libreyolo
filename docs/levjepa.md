# LeVJEPA inference

> **Weight license:** `LibreLeVJEPAl-embed.pt` is CC BY-NC 4.0. It requires
> attribution and is for non-commercial use only. LibreYOLO's native runtime
> remains MIT licensed.

LeVJEPA is a video representation model. It turns a short video clip into a
normalized vector for similarity, retrieval, clustering, or a downstream
classifier. It can also return one feature vector per spatial patch and frame
for applications that need the structure of the clip instead of one pooled
vector.

## Clip embedding

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreLeVJEPAl-embed.pt")
result = model.predict("video.mp4")[0]
embedding = result.embeddings.data  # (1, 1024), L2-normalized
```

For a video path, LibreYOLO samples one centered 16-frame window at
approximately 7.5 frames per second. A long video is therefore not summarized
in full. Split a long video into windows and embed each window when full-video
coverage matters.

Cosine similarity is a dot product because the clip vectors are normalized:

```python
a = model.predict("clip-a.mp4")[0].embeddings.data
b = model.predict("clip-b.mp4")[0].embeddings.data
similarity = a @ b.T
```

## Patch embeddings

```python
tokens = model.embed_tokens("video.mp4")
print(tokens.shape)  # (1, 16, 14, 14, 1024)
```

The axes are batch, frame, patch row, patch column, and feature. Patch tokens
are not L2-normalized. They are useful as input to a temporal or spatial
downstream head; they are not detections or class probabilities.

## Explicit clip input

`predict()` accepts a preprocessed tensor shaped `(B, 16, 3, 224, 224)`.
Tensor input is assumed to have already been resized, center-cropped, and
normalized with ImageNet mean and standard deviation. `embed_tokens()` also
accepts a Python list of exactly 16 images and preprocesses it with LibreYOLO.

## LeVJEPA and V-JEPA 2

Both families facilitate downstream video processing by producing features;
neither produces boxes, tracks, captions, or actions by itself. LeVJEPA's
released model consumes 16 frames and exposes a CLS clip embedding plus patch
tokens. LibreYOLO's V-JEPA 2 encoders consume 64 frames and expose patch tokens
whose mean is used for the public clip embedding; V-JEPA 2 also supports its
released attentive classification probes. LeVJEPA is inference-only and has no
classification probe or pretraining workflow in LibreYOLO.
