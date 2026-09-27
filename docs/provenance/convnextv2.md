# ConvNeXt V2 provenance and validation

## Scope

Eight native 224px ImageNet-1K classifiers: Atto, Femto, Pico, Nano, Tiny,
Base, Large, Huge. Canonical size codes are atto/femto/pico/n/t/b/l/h.
Prediction, top-1/top-5 validation, supervised fine-tuning, checkpoint resume,
ONNX and TorchScript are implemented. Fine-tuning uses the shared
classification trainer with AdamW, lr0=1e-4, and zero stochastic depth.
FCMAE pretraining, LoRA, and the upstream ImageNet reproduction recipe are
not implemented. Official 384/512px and ImageNet-22K variants are not published
by this family.

## Source and license

Architecture: [facebookresearch/ConvNeXt-V2](https://github.com/facebookresearch/ConvNeXt-V2)
at commit `2553895753323c6fe0b2bf390683f5ea358a42b9`.
The dense graph comes from `models/convnextv2.py` and the LayerNorm/GRN
classes in `models/utils.py`, under MIT. No sparse pretraining code is bundled.
The existing classification preprocessing and trainer are reused from LibreYOLO.

Weights: the official `im1k/convnextv2_<variant>_1k_224_ema.pt` files linked
in the pinned upstream README, under CC-BY-NC-4.0. Conversion changes the
container metadata only; learned tensors and their keys are unchanged.
The converter verifies the source SHA-256 identities in
`libreyolo/models/convnextv2/utils.py`. Runtime auto-conversion also recognizes
these identities and attaches their weight terms. A custom tensor layout alone
does not establish a weight license. Save, fine-tune, resume and DDP bootstrap
preserve declared weight terms; explicit scratch initialization clears them.

## Published artifacts

Each repository has exactly five files: checkpoint, README.md, LICENSE,
NOTICE and .gitattributes. Cards use `license: cc-by-nc-4.0`, lead with the
non-commercial restriction, and are in the LibreYOLO classification collection.
The LICENSE is the upstream combined MIT-code/CC-BY-NC-weight file, verbatim.

| Repository | HF revision | Converted checkpoint SHA-256 |
|---|---|---|
| [LibreConvNeXtV2atto-cls](https://huggingface.co/LibreYOLO/LibreConvNeXtV2atto-cls) | `4087f290ad1e3dcffaa205129939769c11faaea5` | `8653554a1e9d9da04d012f6b18e3aed7b1d261a8b170baf2f1399bc7cea00986` |
| [LibreConvNeXtV2femto-cls](https://huggingface.co/LibreYOLO/LibreConvNeXtV2femto-cls) | `5bfcad10d385959e9225d9c5c9146d1d349ff0ec` | `ae17593e70f42925782ee19c2e30f037646e7141ae18c13111b34c5daf0ab8a7` |
| [LibreConvNeXtV2pico-cls](https://huggingface.co/LibreYOLO/LibreConvNeXtV2pico-cls) | `6df121e77e3ae67f47fd6ba4c5b3ce57cc08bf63` | `92bd22d50a6b8ce9998a7cb58c3054e496b1b435b3db029d02445c84970d8548` |
| [LibreConvNeXtV2n-cls](https://huggingface.co/LibreYOLO/LibreConvNeXtV2n-cls) | `43019e820e324e2c135c4dc5c7d326d0efc30f1e` | `c9a8c818cd892dfa2903819623dd4e5fb285c2f31f36ab00c7abb8ccfbae8b57` |
| [LibreConvNeXtV2t-cls](https://huggingface.co/LibreYOLO/LibreConvNeXtV2t-cls) | `4a14ce7d03fe7c595ea30ed842b8f29e9335a03e` | `26b596eb679e9911f09be19eb414542d393eb1746a6ef89be2b031fc37b7e491` |
| [LibreConvNeXtV2b-cls](https://huggingface.co/LibreYOLO/LibreConvNeXtV2b-cls) | `463a8be031d439095e509eaacda3389a3f11101e` | `cf9d984fa981879f01abef443cb991f74879d43ab7d76110d7110d642897e0dd` |
| [LibreConvNeXtV2l-cls](https://huggingface.co/LibreYOLO/LibreConvNeXtV2l-cls) | `87f2eeb40894352c9b366faf737801128df0f84d` | `ff9b112932d912c529fd2e1adb5ed022e925b3b59724f36ab2823a609aea27ca` |
| [LibreConvNeXtV2h-cls](https://huggingface.co/LibreYOLO/LibreConvNeXtV2h-cls) | `b0ffd45fb6c34d71d16db745c9ba394e9dc03362` | `a43a1ebaefa406c04540e33b38fae7a4644cd0694218657f514e07a6fcf9da44` |

## Completed checks (2026-09-13)

- All eight official checkpoints: strict tensor loading and exact FP32 logits
  against the pinned upstream graph on identical seeded 224px input. CPU;
  maximum absolute difference 0 for every size.
- All eight published files: remote LFS SHA-256 equals the local converted
  artifact; license tag, five-file layout and collection membership verified.
- Atto: fresh-directory canonical-filename auto-download and prediction.
- Atto: ONNX and TorchScript CPU logits/probability parity and metadata reload
  with nonzero GRN parameters (`test_convnextv2_export.py`).
- Real weights: inference for all eight sizes and a three-epoch Atto fine-tune
  on smoke10, including class-head replacement and reload (nine e2e tests).
- Atto overfit: six RGB color images, three classes, 40 epochs at 64px,
  batch 6, lr0=1e-4, warmup 0, AMP/EMA off. Reached 100% accuracy at epoch 3;
  final cross-entropy 0.000267; best checkpoint reload retained 100% accuracy.
  This is an overfit fixture, not evidence of generalization.
- Unit tests cover GRN gradients, V1/V2 discrimination, all size signatures,
  raw conversion, preprocessing, checkpoint save/reload, training/resume,
  and preservation/clearing of weight terms.
- Local UI: Atto selected, bundled sample classified, result image and
  classification summary rendered.

Not measured: full ImageNet accuracy, upstream training reproduction, CUDA,
actual multi-GPU DDP, or other export runtimes. TorchScript runtime validation
uses device="cpu"; the local PyTorch build rejected automatic MPS loading of
traced scalar constants. DDP checkpoint preservation is unit-tested separately.
