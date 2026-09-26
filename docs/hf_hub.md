# Hugging Face Hub integration

LibreYOLO can load checkpoints from any Hugging Face Hub repository and push
your own fine-tuned checkpoints back, in both cases using the standard
LibreYOLO metadata schema (v1.0, see `docs/checkpoint_schema.md`).

The integration is optional:

```bash
pip install "libreyolo[hf]"
```

`import libreyolo` never imports `huggingface_hub`; it is only loaded when a
Hub reference or push is actually used.

## Loading models from the Hub

Any repository that contains a LibreYOLO checkpoint can be loaded directly:

```python
from libreyolo import LibreYOLO

model = LibreYOLO("someuser/their-finetune")          # bare owner/repo
model = LibreYOLO("hf://someuser/their-finetune")     # explicit form
model = LibreYOLO("hf://someuser/repo@main/best.pt")  # pin revision and file
```

Resolution rules:

- A local file or directory with the same name always wins over the bare
  `owner/repo` form. Use the `hf://` form to bypass that precedence.
- The bare form is only treated as a Hub reference when the last segment has
  no file extension, so paths like `weights/model.pt` or
  `ckpts/model_best.pth.tar` keep their meaning. A dot followed by a digit is
  a version, not an extension: `owner/model-v1.5` is still a Hub reference.
  `librefacerec-*` names such as `models/librefacerec-l` are never Hub
  references; they auto-download from the LibreYOLO org as before.
- Without an explicit filename, the single `*.pt` (or `*.safetensors`) file
  in the repo is used. Repos with several checkpoints raise an error listing
  them; select one with `hf://owner/repo/<filename>`.

- A revision passed as `@revision` must not contain `/`; use a commit SHA or a
  slash-free branch name.

Downloads go to the shared `huggingface_hub` cache and are loaded through the
same safe path as local files: `torch.load(weights_only=True)` plus metadata
validation.

Hub checkpoints are identified by their LibreYOLO metadata. Recognized
upstream checkpoints are still auto-converted, but a file with no metadata
that cannot be converted is rejected rather than guessed at, because raw
tensor keys from an arbitrary Hub repo can match an unrelated family and fail
much later with a confusing error. To load such a file anyway, download it and
pass a local path, which re-enables legacy architecture detection.

Private and gated repos work once you are authenticated (see below).

## Pushing models to the Hub

Any loaded model can be published, together with an auto-generated model card
derived from its checkpoint metadata (family, size, task, classes, metrics):

```python
model = LibreYOLO("runs/train/yolo9_exp/weights/best.pt")
model.push_to_hub("someuser/my-finetune")                  # public
model.push_to_hub("someuser/my-finetune", private=True)    # private
model.push_to_hub("someuser/my-finetune", license="mit",
                  metrics={"mAP50": 0.62})
```

The repo is created if missing. The checkpoint is uploaded as `model.pt`, so
repeated pushes update the same file. Anyone can then load it back with
`LibreYOLO("someuser/my-finetune")`.

To upload automatically when a training run finishes, use the Hub logger:

```python
model.train(data="data.yaml", epochs=100, loggers="hf:someuser/my-finetune")
```

or, with options:

```python
from libreyolo.training import HuggingFaceHubLogger

model.train(
    data="data.yaml",
    loggers=HuggingFaceHubLogger("someuser/my-finetune", private=False),
)
```

The logger verifies write access at construction time, which also creates the
target repository up front, so a credential or repository-name problem fails
before training starts instead of discarding hours of work at the end. On
train end it pushes `weights/best.pt` (falling back to `last.pt`) with the
final metrics rendered into the model card.

Note the differing defaults: `push_to_hub` creates a public repo (you are
explicitly publishing), while the logger creates a private one, because it
uploads unattended and a model trained on proprietary data must not become
public by surprise. Pass `private=False` to publish from training. Repos that
already exist keep their current visibility either way.

## Authentication

Reading public repos needs no login. Private repos, gated repos, and all
pushes need a Hugging Face token, resolved the standard way:

1. Run `hf auth login` once (stores a token on the machine), or
2. set the `HF_TOKEN` environment variable, or
3. pass `token="hf_..."` to `push_to_hub` / `HuggingFaceHubLogger`.

Pushing requires a token with write scope; create one at
<https://huggingface.co/settings/tokens>. Error messages repeat these steps
whenever authentication is the likely cause.
