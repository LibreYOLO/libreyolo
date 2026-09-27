"""End-to-end VLM training smoke test on a tiny-random Qwen3-VL.

Exercises the full loop on CPU with a few-MB random-weight checkpoint that
shares the real Qwen3-VL architecture, processor, and chat template:

    train() -> adapter checkpoint + contract -> LibreVLM(path) reload -> predict

Random weights mean the loss is meaningless; what this validates is every
contract in the chain: dataset rendering, the real chat template's
prompt-prefix property (the label-masking assumption), LoRA injection scope,
checkpoint save/load, base-repo pinning from the contract, and that a reloaded
fine-tune predicts through the standard Results path.

Needs network on first run (pinned tiny repo download); skips cleanly offline.
"""

import json
from pathlib import Path

import pytest

pytestmark = [pytest.mark.vlm, pytest.mark.network, pytest.mark.e2e]

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

from PIL import Image  # noqa: E402

from libreyolo.models.vlm import LibreVLM  # noqa: E402
from libreyolo.models.vlm.qwen3vl import LibreQwen3VL  # noqa: E402

TINY_REPO = "tiny-random/qwen3-vl"
TINY_REVISION = "be61f02fa193813901f6bc60707eb4ee08f27b02"


class TinyQwen3VL(LibreQwen3VL):
    """Qwen3-VL adapter pointed at a pinned tiny-random checkpoint."""

    FILENAME_PREFIX = "TinyQwen3VLTest"
    HF_REPOS = {"2b": TINY_REPO}
    HF_REVISIONS = {"2b": TINY_REVISION}


def _make_dataset(root: Path) -> Path:
    for split in ("train", "val"):
        images = root / "images" / split
        labels = root / "labels" / split
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        for i in range(2):
            Image.new("RGB", (64, 64), (40 * i + 40, 80, 120)).save(
                images / f"img{i}.png"
            )
            rows = "0 0.5 0.5 0.4 0.4\n" if i == 0 else "1 0.25 0.25 0.3 0.3\n"
            (labels / f"img{i}.txt").write_text(rows, encoding="utf-8")
    yaml_path = root / "data.yaml"
    yaml_path.write_text(
        "path: {root}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n  0: ripe strawberry\n  1: leaf\n".format(root=root.as_posix()),
        encoding="utf-8",
    )
    return yaml_path


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    workdir = tmp_path_factory.mktemp("vlm-train-e2e")
    import os

    cwd = os.getcwd()
    os.chdir(workdir)  # weights/ and runs/ land in the temp dir
    try:
        try:
            model = TinyQwen3VL(size="2b", device="cpu")
        except Exception as exc:  # pragma: no cover - offline environments
            pytest.skip(f"tiny Qwen3-VL checkpoint unavailable: {exc}")
        yield workdir, model
    finally:
        os.chdir(cwd)


def test_train_checkpoint_reload_predict(tiny_model):
    workdir, model = tiny_model
    yaml_path = _make_dataset(workdir / "dataset")

    results = model.train(
        data=str(yaml_path),
        epochs=1,
        batch=1,
        accumulate=2,
        workers=0,
        hflip=0.0,
        gradient_checkpointing=False,
        seed=0,
    )

    # Results dict and run layout.
    best = Path(results["best"])
    assert best.is_dir()
    assert Path(results["last"]).is_dir()
    assert results["epochs"] == 1
    assert results["metric_name"] == "val/loss"
    assert results["best_metric"] is not None

    # Checkpoint contract: adapter-only (no base weights), contract file, and
    # the vocabulary taken from the dataset names in id order.
    contract = json.loads((best / "libreyolo_vlm.json").read_text(encoding="utf-8"))
    assert contract["family"] == "qwen3vl"
    assert contract["base_repo"] == TINY_REPO
    assert contract["base_revision"] == TINY_REVISION
    assert contract["names"] == ["ripe strawberry", "leaf"]
    assert contract["bbox_key"] == "bbox_2d"
    assert (best / "adapter_config.json").exists()
    assert not (best / "config.json").exists(), "adapter checkpoint must not embed base weights"
    adapter_config = json.loads((best / "adapter_config.json").read_text(encoding="utf-8"))
    assert adapter_config["r"] == 16

    # The wrapper stays usable after training: adapters merged, vocab sticky.
    assert model.names == {0: "ripe strawberry", 1: "leaf"}
    assert not any(
        getattr(module, "lora_A", None) is not None for module in model.model.modules()
    ), "adapters must be merged out of the inference model"

    # Reload through the factory; contract pins the tiny base repo.
    reloaded = LibreVLM(str(best), device="cpu")
    assert isinstance(reloaded, LibreQwen3VL)
    assert reloaded.names == {0: "ripe strawberry", 1: "leaf"}

    # Predict runs the standard Results path (random weights, so no boxes are
    # expected; the contract under test is the pipeline, not accuracy).
    image = workdir / "dataset" / "images" / "val" / "img0.png"
    result = reloaded.predict(str(image), max_det=10)
    assert result is not None
    assert hasattr(result, "boxes")


def test_lora_stays_out_of_vision_tower(tiny_model):
    workdir, model = tiny_model
    from peft import LoraConfig, get_peft_model

    from libreyolo.models.vlm.training.recipes import get_recipe

    recipe = get_recipe("qwen3vl")
    peft_model = get_peft_model(
        model.model,
        LoraConfig(
            r=recipe.lora_r,
            lora_alpha=recipe.lora_alpha,
            target_modules=recipe.target_modules,
        ),
    )
    try:
        injected = [
            name
            for name, module in peft_model.named_modules()
            if getattr(module, "lora_A", None) is not None
        ]
        assert injected, "recipe matched no modules"
        assert not [n for n in injected if "visual" in n], "vision tower got adapters"
        assert all("language_model" in n for n in injected)
    finally:
        merged = peft_model.merge_and_unload()
        model.model = merged
