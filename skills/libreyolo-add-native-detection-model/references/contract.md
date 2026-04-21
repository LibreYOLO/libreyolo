# Interface Contract for Native Detection Models

Every native detection model plugs into four abstract base classes:

1. `BaseModel` — inference wrapper (`libreyolo/models/base/model.py`)
2. `BaseTrainer` — training loop (`libreyolo/training/trainer.py`)
3. `TrainConfig` — hyperparameter dataclass (`libreyolo/training/config.py`)
4. `BaseValPreprocessor` — validation preprocessing (`libreyolo/validation/preprocessors.py`)

This document catalogs every required and optional override, with signatures and canonical examples from YOLOX and YOLOv9.

---

## 1. BaseModel

File: `libreyolo/models/base/model.py`

### 1.1 Class attributes (required)

| Attribute | Type | Example | Purpose |
|-----------|------|---------|---------|
| `FAMILY` | `str` | `"yolox"` | Canonical family identifier. Used in checkpoint metadata and error messages. |
| `FILENAME_PREFIX` | `str` | `"LibreYOLOX"` | Prefix for weight filenames. Drives the HuggingFace URL. |
| `WEIGHT_EXT` | `str` | `".pt"` (default — don't change) | File extension for weight filenames. |
| `INPUT_SIZES` | `dict[str, int]` | `{"n": 416, "s": 640}` | Maps size code → input resolution. Keys become the valid values for the `size` kwarg. |
| `val_preprocessor_class` | class | `YOLOXValPreprocessor` | The `BaseValPreprocessor` subclass to use during validation. |

Set these as class-level `ClassVar`s at the top of your wrapper.

### 1.2 Classmethods (required — you must implement)

These three classmethods make your model discoverable by the unified `LibreYOLO()` factory.

#### `can_load(weights_dict: dict) -> bool`

Inspect state-dict keys and return `True` if the weights belong to this family.

**Rule**: make the heuristic specific. Check for tokens unique to your architecture — not generic ones like `"backbone"` or `"weight"`.

```python
# YOLOX (yolox/model.py:45-47)
@classmethod
def can_load(cls, weights_dict: dict) -> bool:
    return any("backbone.backbone" in k or "head.stems" in k for k in weights_dict)
```

```python
# YOLOv9 (yolo9/model.py:44-49)
@classmethod
def can_load(cls, weights_dict: dict) -> bool:
    keys_lower = [k.lower() for k in weights_dict]
    return (
        any("repncspelan" in k or "adown" in k or "sppelan" in k for k in keys_lower)
        or any("backbone.elan" in k or "neck.elan" in k for k in weights_dict)
    )
```

Registry iteration order is import order. If your heuristic might collide with an existing family, make it more specific (do not loosen the other one).

#### `detect_size(weights_dict: dict) -> Optional[str]`

Infer size code from a distinctive weight's shape.

```python
# YOLOX (yolox/model.py:49-55) — read stem conv channels
@classmethod
def detect_size(cls, weights_dict: dict) -> Optional[str]:
    key = "backbone.backbone.stem.conv.conv.weight"
    if key not in weights_dict:
        return None
    ch = weights_dict[key].shape[0]
    return {16: "n", 24: "t", 32: "s", 48: "m", 64: "l", 80: "x"}.get(ch)
```

Return `None` if the size cannot be inferred — the factory will fall through to filename-based detection.

#### `detect_nb_classes(weights_dict: dict) -> Optional[int]`

Infer class count from the head's output channels.

```python
# YOLOX (yolox/model.py:57-60)
@classmethod
def detect_nb_classes(cls, weights_dict: dict) -> Optional[int]:
    key = "head.cls_preds.0.weight"
    return weights_dict[key].shape[0] if key in weights_dict else None
```

### 1.3 Classmethods (inherited — work automatically once class attributes are set)

You do not need to override these. They are built on top of `FILENAME_PREFIX`, `INPUT_SIZES`, and `WEIGHT_EXT`.

- `detect_size_from_filename(filename: str) -> Optional[str]` — regex-matches `<FILENAME_PREFIX><size>(-seg)?<WEIGHT_EXT>` (base/model.py:230-237)
- `detect_task_from_filename(filename: str) -> Optional[str]` — extracts `seg` suffix if present (base/model.py:239-248)
- `get_download_url(filename: str) -> Optional[str]` — returns `https://huggingface.co/LibreYOLO/<name>/resolve/main/<name><ext>` (base/model.py:250-259)

### 1.4 Instance methods (abstract — you must implement)

#### `_init_model(self) -> nn.Module`

Construct and return the neural network. Called from `BaseModel.__init__`.

```python
# YOLOX (yolox/model.py:96-97)
def _init_model(self) -> nn.Module:
    return LibreYOLOXModel(config=self.size, nb_classes=self.nb_classes)
```

#### `_get_available_layers(self) -> Dict[str, nn.Module]`

Return a dict of human-readable layer names → `nn.Module` references. Used by feature-extraction and layer-freezing utilities.

```python
# YOLOX (yolox/model.py:99-106)
def _get_available_layers(self) -> Dict[str, nn.Module]:
    return {
        "backbone_stem": self.model.backbone.stem,
        "backbone_dark2": self.model.backbone.dark2,
        ...
    }
```

#### `_get_preprocess_numpy() -> callable` (static)

Return a `preprocess_numpy(img_rgb_hwc, input_size)` callable. Used on the CPU/numpy export path where the ONNX-exported model expects pre-transformed input.

```python
# YOLOX (yolox/model.py:115-119)
@staticmethod
def _get_preprocess_numpy():
    from .utils import preprocess_numpy
    return preprocess_numpy
```

#### `_preprocess(self, image, color_format, input_size) -> Tuple[torch.Tensor, Image.Image, Tuple[int, int], float]`

Preprocess one image for inference. Returns `(input_tensor, original_pil_image, (H, W), ratio)`.

#### `_forward(self, input_tensor) -> Any`

Run the model forward pass. The return value is opaque to `BaseModel` — it's passed through to `_postprocess`.

#### `_postprocess(self, output, conf_thres, iou_thres, original_size, max_det, ratio) -> Dict`

Decode raw model output into a detections dict. Structure: `{"num_detections": int, "boxes": np.ndarray, "scores": np.ndarray, "classes": np.ndarray}`.

### 1.5 Instance methods (concrete — optional overrides)

Override only when the default doesn't work.

- `_strict_loading() -> bool` (default `True`) — set to `False` if checkpoints contain profiling buffers, EMA artifacts, or legacy keys. YOLOX overrides at `yolox/model.py:108-109`.
- `_prepare_state_dict(state_dict) -> dict` (default identity) — defined as an extension hook, but the current shared loader does **not** call it. YOLOv9 overrides it to rewrite `detect.*` → `head.*` (`yolo9/model.py:133-141`), but that override does not fire unless you wire it into your load path explicitly.
- `_rebuild_for_new_classes(new_nb_classes)` (default works for most models) — override only if you need special handling for class-count changes.
- `_get_model_name() -> str` (default `self.FAMILY`) — rarely overridden.

### 1.6 Auto-registration

`BaseModel.__init_subclass__` (base/model.py:47-54) appends any non-abstract subclass to `BaseModel._registry` automatically. Nothing to do on your side beyond importing your class once.

```python
def __init_subclass__(cls, **kwargs):
    super().__init_subclass__(**kwargs)
    if (
        hasattr(cls, "can_load")
        and not getattr(cls.can_load, "__isabstractmethod__", False)
        and cls not in BaseModel._registry
    ):
        BaseModel._registry.append(cls)
```

---

## 2. BaseTrainer

File: `libreyolo/training/trainer.py`

### 2.1 Abstract methods (required)

#### `get_model_family(self) -> str`

Canonical family string, used in checkpoint metadata to reject cross-family loads.

```python
def get_model_family(self) -> str:
    return "<family>"
```

#### `get_model_tag(self) -> str`

Human-readable tag for log messages.

```python
def get_model_tag(self) -> str:
    return f"<Family>-{self.config.size}"
```

#### `create_transforms(self) -> Tuple[transform, MosaicDatasetClass]`

Return a `(preproc_transform, MosaicDatasetClass)` tuple. The preproc transform is applied per-sample; the mosaic class wraps the base dataset.

```python
# YOLOX (yolox/trainer.py:30-36)
def create_transforms(self):
    preproc = TrainTransform(
        max_labels=50,
        flip_prob=self.config.flip_prob,
        hsv_prob=self.config.hsv_prob,
    )
    return preproc, MosaicMixupDataset
```

#### `create_scheduler(self, iters_per_epoch: int) -> Scheduler`

Return a scheduler object with an `update_lr(iters: int)` method. The training loop calls `update_lr` every iteration.

Available schedulers in `libreyolo/training/scheduler.py`:
- `WarmupCosineScheduler` — warmup + cosine (YOLOX default)
- `LinearLRScheduler` — warmup + linear decay (YOLOv9 default)
- `CosineAnnealingScheduler` — warmup + cosine annealing

You can also write your own and return it here — just implement `update_lr(iters)`.

#### `get_loss_components(self, outputs: Dict) -> Dict[str, float]`

Extract named loss components from the model's output dict for progress-bar and TensorBoard logging.

Training forward must return a dict containing `total_loss`. `BaseTrainer` reads `outputs["total_loss"]` directly during backprop; `get_loss_components()` is only for logging the extra terms.

```python
# YOLOX (yolox/trainer.py:49-55) — note the key names match what the YOLOX head returns
def get_loss_components(self, outputs: Dict) -> Dict[str, float]:
    return {
        "iou": outputs.get("iou_loss", 0),
        "obj": outputs.get("obj_loss", 0),
        "cls": outputs.get("cls_loss", 0),
        "l1": outputs.get("l1_loss", 0),
    }
```

The `total_loss` is handled by the base trainer and does not need to be extracted.

### 2.2 Concrete hooks (optional — override when needed)

#### `on_setup(self)`

Called after the model is placed on the device, before data loaders are built. Good for bias initialization, profiling setup, etc.

```python
# YOLOX (yolox/trainer.py:57-61)
def on_setup(self):
    if hasattr(self.model, "head") and hasattr(self.model.head, "initialize_biases"):
        self.model.head.initialize_biases(0.01)
```

#### `on_mosaic_disable(self)`

Called once when mosaic/mixup is turned off for the final `no_aug_epochs`. Default implementation calls `self.train_loader.dataset.close_mosaic()`.

```python
# YOLOX (yolox/trainer.py:63-65) — flips on L1 loss phase
def on_mosaic_disable(self):
    self.train_loader.dataset.close_mosaic()
    self.model.head.use_l1 = True
```

#### `on_forward(self, imgs, targets) -> Dict`

Run the forward pass. Default: `self.model(imgs, targets)`. Override only if your model's call signature differs.

```python
# YOLOv9 (yolo9/trainer.py:71-72) — explicit keyword arg
def on_forward(self, imgs, targets):
    return self.model(imgs, targets=targets)
```

### 2.3 Config binding

```python
@classmethod
def _config_class(cls) -> Type[TrainConfig]:
    return <Family>Config
```

Override this classmethod to make the trainer use your family's config dataclass when constructing `self.config` from kwargs.

If you forget this override, `BaseTrainer` falls back to plain `TrainConfig`. That base config is broadly YOLO-style but still carries defaults like `scheduler="yoloxwarmcos"` and `mixup_prob=1.0`, so your family's intended recipe can be silently replaced by the wrong defaults.

---

## 3. TrainConfig

File: `libreyolo/training/config.py`

### 3.1 Subclassing pattern

```python
@dataclass(kw_only=True)
class <Family>Config(TrainConfig):
    """<Family>-specific training defaults."""
    # Override ONLY the fields that differ from TrainConfig base defaults
    momentum: float = 0.9
    warmup_epochs: int = 3
    # ...
```

**Important**: `kw_only=True` is required — `TrainConfig` is defined as `@dataclass(kw_only=True)` and all subclasses must match.

### 3.2 Base fields (TrainConfig — see config.py:14-74)

Inherit these defaults unless your recipe needs a different value. Categories:

- **Model**: `size`, `num_classes`
- **Data**: `data`, `data_dir`, `imgsz`
- **Training**: `epochs`, `batch`, `device`
- **Optimizer**: `optimizer`, `lr0`, `momentum`, `weight_decay`, `nesterov`
- **Scheduler**: `scheduler`, `warmup_epochs`, `warmup_lr_start`, `no_aug_epochs`, `min_lr_ratio`
- **Augmentation**: `mosaic_prob`, `mixup_prob`, `hsv_prob`, `flip_prob`, `degrees`, `translate`, `mosaic_scale`, `mixup_scale`, `shear`
- **Training features**: `ema`, `ema_decay`, `amp`
- **Checkpointing**: `project`, `name`, `exist_ok`, `save_period`, `eval_interval`
- **System**: `workers`, `patience`, `resume`, `log_interval`, `seed`

### 3.3 Example subclasses

See `YOLOXConfig` (config.py:105-119) and `YOLO9Config` (config.py:122-138) — compact examples of which fields to override.

---

## 4. BaseValPreprocessor

File: `libreyolo/validation/preprocessors.py`

### 4.1 Constructor

```python
def __init__(self, img_size: Tuple[int, int], max_labels: int = 120):
    self.img_size = img_size
    self.max_labels = max_labels
```

Override only if you need additional parameters (e.g. `YOLOXValPreprocessor` adds `pad_value` at line 92).

### 4.2 Required methods

#### `__call__(self, img, targets, input_size) -> Tuple[np.ndarray, np.ndarray]`

Preprocess one image (H, W, C BGR) and its targets (N, 5) `[x1, y1, x2, y2, class]`.

Return `(preprocessed_img_chw, padded_targets_max_labels_5)`.

#### `@property normalize -> bool`

`True` if the preprocessor divides by 255 (outputs 0–1), `False` if it leaves the image in 0–255 range.

### 4.3 Optional properties

#### `@property uses_letterbox -> bool` (default `False`)

`True` if you use aspect-preserving resize with padding; `False` for plain square resize. `DetectionValidator` reads this to scale target boxes correctly.

#### `@property custom_normalization -> bool` (default `False`)

`True` if you apply ImageNet-style mean/std normalization inside `__call__` (validator must NOT rescale). `False` for vanilla YOLO preprocessors. Almost all native detectors leave this `False`.

### 4.4 Canonical examples

Review these side-by-side to see how properties drive different paths:

- `YOLOXValPreprocessor` (preprocessors.py:89-134) — letterbox, BGR, 0-255 (`normalize=False`, `uses_letterbox=True`)
- `YOLO9ValPreprocessor` (preprocessors.py:188-232) — letterbox, RGB, 0-1 (`normalize=True`, `uses_letterbox=True`)
- `StandardValPreprocessor` (preprocessors.py:49-86) — plain resize, RGB, 0-1 (`normalize=True`, `uses_letterbox=False`)
- `RFDETRValPreprocessor` (preprocessors.py:137-185) — plain resize, ImageNet-normalized (`normalize=False`, `custom_normalization=True`)

**The preprocessor must exactly match your training transform's color space, normalization, and letterbox behavior.** See `gotchas.md` section 1 — mismatches here silently collapse validation mAP.
