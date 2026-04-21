# Files Checklist for Native Detection Model Integration

Every file you create or edit, grouped by category. Required files are marked **REQUIRED**; conditional files have their trigger noted.

All line numbers refer to the current branch state; treat them as navigation hints, not fixed constants.

---

## 1. New model-package files

All of these live in `libreyolo/models/<family>/`.

### `__init__.py` — **REQUIRED**

Empty module marker. Can contain a one-line docstring at most.

### `model.py` — **REQUIRED**

Contains: `LibreYOLO<Family>(BaseModel)` wrapper class.

Must define:
- Class attributes: `FAMILY`, `FILENAME_PREFIX`, `INPUT_SIZES`, `val_preprocessor_class`
- Classmethods: `can_load`, `detect_size`, `detect_nb_classes`
- Instance methods: `_init_model`, `_get_available_layers`, `_get_preprocess_numpy`, `_preprocess`, `_forward`, `_postprocess`

See `reference/contract.md` section 1 for signatures.

Size budget: YOLOX is 270 LOC; YOLOv9 is 311 LOC. Expect 200-350 LOC.

### `nn.py` — **REQUIRED**

Contains: the `torch.nn.Module` modules. The root should be `Libre<Family>Model(nn.Module)` with signature `forward(x, targets=None)`.

The detection head must expose a boolean `export` attribute used by the ONNX exporter:

```python
class <Family>Head(nn.Module):
    def __init__(self, ...):
        ...
        self.export = False  # flipped True by libreyolo/export/exporter.py:260-266
```

This is the largest file — typically 800-1200 LOC for a full architecture port.

### `trainer.py` — **REQUIRED**

Contains: `<Family>Trainer(BaseTrainer)`.

Typically short (~70 LOC). See YOLOX's at `yolox/trainer.py` (68 LOC) for a minimal template.

Required overrides: `_config_class`, `get_model_family`, `get_model_tag`, `create_transforms`, `create_scheduler`, `get_loss_components`.

### `utils.py` — **REQUIRED**

Contains: `preprocess_numpy(img_rgb_hwc, input_size)`, `preprocess_image(image, input_size)`, `postprocess(outputs, ...)`. These are imported by the wrapper's `_preprocess`, `_get_preprocess_numpy`, and `_postprocess`.

### `loss.py` — **OPTIONAL** (almost always needed)

If your loss is more than a line or two, put it here. YOLOX and YOLOv9 both have `loss.py`. Keeps the model and trainer files readable.

### `transforms.py` — **OPTIONAL** (needed if augmentation diverges)

If your augmentation pipeline differs materially from the YOLOX default (`libreyolo/training/augment.py` — `TrainTransform` + `MosaicMixupDataset`), create `<Family>TrainTransform` + `<Family>MosaicMixupDataset` here. YOLOv9 does this (`yolo9/transforms.py`) because its boxes are in normalized xyxy while YOLOX uses pixel cxcywh.

---

## 2. Central registry edits

### `libreyolo/models/__init__.py` — **REQUIRED**

Add ONE line after the existing YOLOX/YOLOv9 imports (currently around line 24):

```python
from .<family>.model import LibreYOLO<Family>  # noqa: E402
```

Do not add anything else — registration is automatic via `__init_subclass__`.

Import order determines `can_load()` priority. If your heuristic might overlap another family's, position your import before the overlap.

### `libreyolo/__init__.py` — **REQUIRED**

Line 7 (or thereabouts):

```python
from .models import LibreYOLO, LibreYOLOX, LibreYOLO9, LibreYOLO<Family>
```

Add `"LibreYOLO<Family>"` to `__all__` (around line 50-77).

### `libreyolo/training/config.py` — **REQUIRED**

Append the dataclass after `YOLO9Config` (currently line 138):

```python
@dataclass(kw_only=True)
class <Family>Config(TrainConfig):
    """<Family>-specific training defaults."""
    momentum: float = ...
    warmup_epochs: int = ...
    # override only what differs
```

`kw_only=True` is mandatory — `TrainConfig` is declared that way.

### `libreyolo/validation/preprocessors.py` — **REQUIRED**

Append a class:

```python
class <Family>ValPreprocessor(BaseValPreprocessor):
    """<Family> preprocessor: <letterbox|resize>, <RGB|BGR>, <0-1|0-255>."""

    @property
    def normalize(self) -> bool: ...

    @property
    def uses_letterbox(self) -> bool: ...

    def __call__(self, img, targets, input_size): ...
```

See `YOLO9ValPreprocessor` at preprocessors.py:188-232 for a copy-paste starting point.

---

## 3. Cross-cutting extension points (conditional)

Only edit these if your model diverges from the YOLO-grid output convention.

### `libreyolo/backends/base.py` — **CONDITIONAL**

Add `_preprocess_<family>` and `_parse_<family>` branches only if your ONNX/TensorRT output format differs from standard YOLO grid outputs.

- YOLOX and YOLOv9 did NOT need this — they share the default YOLO parsing path.
- RF-DETR and the historical RT-DETR both needed it (query-based outputs).

### `libreyolo/backends/tensorrt.py` — **CONDITIONAL**

Add output-tensor-name → family detection only if the exported ONNX produces outputs with non-standard names (e.g. `pred_logits` + `pred_boxes` for DETR-style).

### `pyproject.toml` — **CONDITIONAL**

Only if your model needs a PyPI dependency beyond the core `[dependencies]`:

```toml
[project.optional-dependencies]
<family> = ["some_dep>=1.0.0"]
all = [
    "libreyolo[onnx]",
    "libreyolo[rfdetr]",
    "libreyolo[<family>]",  # ← add here
    "libreyolo[tensorrt]",
    ...
]
```

YOLOX and YOLOv9 required no extra (pure torch).

---

## 4. Tests & CI

### `tests/e2e/conftest.py` — **REQUIRED**

Append rows to `MODEL_CATALOG` (currently line 325):

```python
MODEL_CATALOG = [
    ("yolox", "n", "LibreYOLOXn.pt"),
    ...existing rows...
    ("<family>", "s", "Libre<Family>s.pt"),
    ("<family>", "m", "Libre<Family>m.pt"),
]
```

This is the single source of truth — derived lists `YOLOX_SIZES`, `FULL_TEST_MODELS`, `ALL_MODELS`, etc. compute from it automatically (lines 343-358).

Consider adding your smallest variant to `QUICK_TEST_MODELS` (line 352) if it's small and fast enough for CI.

### `tests/e2e/configs/<family>.yaml` — **REQUIRED**

Copy `tests/e2e/configs/yolo9.yaml` or `yolox.yaml` as a template. Edit:
- `model_type: <family>`
- `sizes:` dict — one entry per size with `imgsz`, `lr0`, `weights` filename
- All training fields (`epochs`, `batch_size`, `optimizer`, `scheduler`, etc.)
- Augmentation fields (`mosaic_prob`, `mixup_prob`, etc.)

### `tests/unit/test_<family>_layers.py` — **REQUIRED**

Use `tests/unit/test_yolo9_layers.py` as a comprehensive template. Minimum viable coverage:

```python
import torch
from libreyolo.models.<family>.model import LibreYOLO<Family>
from libreyolo.models.base import BaseModel


def test_registered():
    assert LibreYOLO<Family> in BaseModel._registry


def test_can_load_positive():
    fake_state = {"<a key your can_load recognizes>": torch.zeros(...)}
    assert LibreYOLO<Family>.can_load(fake_state)


def test_can_load_negative():
    fake_state = {"head.stems.0.conv.weight": torch.zeros(...)}  # YOLOX key
    assert not LibreYOLO<Family>.can_load(fake_state)


def test_detect_size():
    fake_state = {"<distinctive weight key>": torch.zeros(32, 3, 3, 3)}
    assert LibreYOLO<Family>.detect_size(fake_state) == "s"


def test_forward_shape():
    model = LibreYOLO<Family>(size="s", nb_classes=80)
    x = torch.randn(1, 3, 640, 640)
    out = model.model(x)
    # assert expected output shape
```

For thoroughness, also add per-layer shape tests (see `test_yolo9_layers.py` for coverage of Conv, Bottleneck, ELAN, head, etc.).

---

## 5. Docs (optional)

### `README.md` — **OPTIONAL**

Update the supported-architectures line (currently around line 7) to include your model.

### `CHANGELOG.md` — **N/A**

No changelog in this repo. Release notes live on GitHub releases.

---

## Summary checklist

Quick verification that nothing is missed:

```
libreyolo/models/<family>/
├── __init__.py                    ☐ REQUIRED (empty)
├── model.py                       ☐ REQUIRED
├── nn.py                          ☐ REQUIRED
├── trainer.py                     ☐ REQUIRED
├── utils.py                       ☐ REQUIRED
├── loss.py                        ☐ OPTIONAL
└── transforms.py                  ☐ OPTIONAL

Central edits:
☐ libreyolo/__init__.py            (export + __all__)
☐ libreyolo/models/__init__.py     (one-line import)
☐ libreyolo/training/config.py     (append <Family>Config)
☐ libreyolo/validation/preprocessors.py  (append <Family>ValPreprocessor)

Conditional edits:
☐ libreyolo/backends/base.py       (if non-YOLO output)
☐ libreyolo/backends/tensorrt.py   (if non-standard ONNX output names)
☐ pyproject.toml                   (if extra PyPI deps)

Tests:
☐ tests/e2e/conftest.py            (MODEL_CATALOG rows)
☐ tests/e2e/configs/<family>.yaml  (new)
☐ tests/unit/test_<family>_layers.py  (new)

Weights:
☐ HuggingFace repos created at huggingface.co/LibreYOLO/Libre<Family><size>/
☐ Weight files uploaded, one per size
```
