# LoRA fine-tuning

Parameter-efficient fine-tuning for RF-DETR on low-VRAM GPUs. Freezes the DINOv2
backbone and trains small adapters on it plus the projector, decoder, and head.

## Use

```python
model = LibreYOLO("LibreRFDETRn.pt")
model.train(data="data.yaml", lora=True)
```

`lora=True` is the whole API. Needs the optional extra:

```
pip install "libreyolo[lora]"
```

## Recipe

DoRA, rank 16, alpha 16, on the backbone attention `query`/`key`/`value`. Matches
the RF-DETR reference. Adapters merge into dense weights on `export()`, so exported
models have no `peft` dependency.

Training checkpoints (`best.pt` and `last.pt`) keep the adapter tensors so they
can be resumed or inspected. Loading those checkpoints requires the `lora` extra.
Use `export()` when you need a dense inference artifact without `peft`.

## Scope

- RF-DETR only. Other families raise instead of silently ignoring `lora=True`.
- The detection head always stays trainable (custom class counts need it).
- Saves optimizer/gradient memory, not activations. For the tightest VRAM, lower
  `batch` or `imgsz`.
