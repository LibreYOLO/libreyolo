# PP-DocLayoutV3

`LibrePPDocLayoutV3` detects the layout regions of a document page and ranks
them in reading order. It is a port of PP-DocLayoutV3 from the PaddleOCR
project (Apache-2.0 code and weights): an RT-DETR-style set-prediction detector
with an HGNetV2-L backbone, mask-enhanced query initialisation and a
global-pointer head that scores every pair of regions.

| Checkpoint | Size | Input | Classes | Weights |
| --- | --- | --- | --- | --- |
| `LibrePPDocLayoutV3l.pt` | `l` | 800x800 | 25 | Apache-2.0, [LibreYOLO/LibrePPDocLayoutV3l](https://huggingface.co/LibreYOLO/LibrePPDocLayoutV3l) |

## Usage

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibrePPDocLayoutV3l.pt")
result = model("page.png", conf=0.5)
for order, box, cls, score in zip(
    result.reading_order, result.boxes.xyxy, result.boxes.cls, result.boxes.conf
):
    print(int(order), result.names[int(cls)], round(float(score), 3), box.tolist())
```

The upstream pipeline uses `conf=0.5`; LibreYOLO's default `conf=0.25` keeps
more low-confidence regions. Upstream's official safetensors file also loads
directly: `LibreYOLO("model.safetensors")` converts it once and writes a
LibreYOLO checkpoint beside it.

## Output contract

- Task `detect`: `Results.boxes` holds xyxy boxes on the original page,
  confidences and class ids over the 25 classes below.
- Rows are sorted in reading order. `Results.reading_order` is a one-based
  int64 vector aligned with the boxes; it slices with `results[i]` like
  `track_id` and appears as `reading_order` in `Results.summary()` rows.
- Ranks are computed among the regions kept at `conf`. A `classes=` filter or a
  `max_det` cut keeps the surviving boxes' ranks, so the values can have gaps.
- As upstream, scores are per-class sigmoids and the top 300 (query, class)
  pairs are kept, so one region can be reported with two labels. No NMS.
- Preprocessing: stretch resize to 800x800 with cv2 bicubic, RGB scaled to
  `[0, 1]`, no mean/std normalization.

Classes: `abstract`, `algorithm`, `aside_text`, `chart`, `content`,
`display_formula`, `doc_title`, `figure_title`, `footer`, `footer_image`,
`footnote`, `formula_number`, `header`, `header_image`, `image`,
`inline_formula`, `number`, `paragraph_title`, `reference`,
`reference_content`, `seal`, `table`, `text`, `vertical_text`,
`vision_footnote`.

## Validation evidence

Against the Apache-2.0 PyTorch reference (`transformers` v5.22.1, eager
attention) on CUDA, fp32, same input tensor: class logits, boxes,
reading-order logits and masks match with `max_abs_diff == 0` on three real
pages and a random tensor, and the decoded regions (labels, scores, boxes,
reading order) are identical. With each side's own preprocessing (torchvision
resize in the reference, cv2 here) the same regions and order come out, boxes
within 0.04 px. Script: `weights/parity_ppdoclayoutv3.py`.

## Export

ONNX (opset 17) and TorchScript, fixed 800x800, FP32. The graph outputs
`pred_logits`, `pred_boxes` and `order_votes` (one reading-order vote per
query); exported backends return the same `Results`, including
`reading_order`. See `docs/export_support.md`.

## Limits

- Inference-only. Fine-tuning needs reading-order and per-region mask
  supervision that LibreYOLO's detect datasets do not carry.
- The per-region polygons upstream derives from the mask branch are not
  exposed; `model.model(x, return_masks=True)` returns the raw mask logits.
- Devices: pure PyTorch, so CPU, CUDA and MPS work. On MPS the positional
  encoding is computed in float32 (no float64 there), so results can differ
  from CPU/CUDA in the last bits.
