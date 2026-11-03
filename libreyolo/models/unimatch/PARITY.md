# UniMatch parity

Upstream `evaluate_flow.inference_flow` (preprocessing, model call and
postprocessing copied verbatim into `weights/parity_unimatch.py`) against the
public LibreYOLO sequence API, `LibreYOLO("<converted .pt>")(frames)`, on the
same frames.

Setup: RTX 3090, torch 2.11.0+cu128, fp32, TF32 disabled for matmul and cuDNN,
upstream commit `943a99006950ab6c84ea72a72beb147fdd03c19a`. Input: the
one-second campaign clip, 24 PNG frames at 640x360, so 23 consecutive pairs.
Both sizes resize to the inference canvas (640x368 for `s`, 640x384 for `l`)
and back, so the resize and vector rescale path is exercised.

| Size | Checkpoint | Pairs | Max abs diff (px) | Mean abs diff (px) | Mean EPE between the two (px) | Mean flow magnitude (px) |
|---|---|---|---|---|---|---|
| `s` | gmflow-scale1-things-e9887eda | 23 | 0.0 | 0.0 | 0.0 | 28.01 |
| `l` | gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d | 23 | 0.0 | 0.0 | 0.0 | 29.64 |

The outputs are bitwise identical: the vendored model code and the tensors are
unchanged, and the wrapper reproduces upstream's geometry exactly.

Reproduce:

```
python weights/parity_unimatch.py --upstream <unimatch checkout> \
    --frames <frame dir> --size l \
    --upstream-ckpt gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth \
    --libre-ckpt LibreUniMatchl-flow.pt
```

Known deviation, outside the parity set: for portrait frames upstream
transposes the image to landscape and transposes the flow back without
swapping its two channels, so its `dx` and `dy` come out exchanged. LibreYOLO
swaps them, as the flow contract requires. Checked on a portrait 360x480 crop
pair with a 6 px leftward shift: LibreYOLO's median `(dx, dy)` is
`(-6.03, 0.00)`.

The Sintel and KITTI variants (`LibreUniMatchl-flow-sintel.pt`,
`LibreUniMatchl-flow-kitti.pt`) share the `l` graph and the same metadata-wrap
conversion; they were not run through the clip parity.
