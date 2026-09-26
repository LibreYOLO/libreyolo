"""Per-family CUDA graph *training* capture, through the real train() API.

The unit suite covers dispatch and gating with fakes. This file runs actual
training runs on a generated dataset and checks the two things a user cares
about:

* capture engages and stays engaged. Every failure mode in this feature
  degrades to eager, silently, so a run that merely completes proves
  nothing; the capture log line has to be asserted. Five families shipped
  with the flag doing literally nothing until this assertion was added.
* enabling the flag does not move the loss trajectory.

The second check adapts to the family rather than assuming determinism: the
eager arm runs twice to measure how far it moves on its own, and the graphed
arm is held to that spread plus a documented per-family ``rel_tol``.
Deformable attention accumulates its backward with atomics and TF32
convolutions pick their reduction order per launch, so several families do
not reproduce their own eager run either, and a fixed band would test that
noise instead of the capture.

No downloads: the datasets are generated and every model is built with
``model_path=None``. Families that pull a pretrained backbone at
construction are marked ``external_data``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

# Not ``general_nightly``: that marker has a documented contract as the broad
# *inference* sweep (docs/testing.md), and these are training runs. The full
# file is ``e2e`` + ``slow``; a representative subset is additionally marked
# ``training_nightly`` so the real capture paths run on GPU every night. That
# lane exists because neither inference nightly marker selects this file, and
# a feature whose every failure mode is a silent fallback to eager is exactly
# the kind that must not rely on someone running it by hand.
pytestmark = [pytest.mark.e2e, pytest.mark.slow]

# One case per distinct mechanism rather than every family: the reference
# bespoke spec (yolo9), a spec built on a refactored family head plus the
# mid-run invalidation path (yolox), the encoder-boundary DETR mixin (dfine),
# the classify mixin (resnet), and the semantic mixin, which is also the one
# family that draws its own RNG stream inside the graph (segformer).
NIGHTLY_FAMILIES = frozenset({"yolo9", "yolox", "dfine", "resnet", "segformer"})


def _cases(cases):
    """Attach ``training_nightly`` to the representative subset."""
    return [
        pytest.param(
            *case,
            marks=[pytest.mark.training_nightly] if case[0] in NIGHTLY_FAMILIES else [],
        )
        for case in cases
    ]

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA graph training capture requires a CUDA device",
)

CAPTURE_LOGGER = "libreyolo.training.cuda_graph"


# (family id, class name, size, imgsz, task, rel_tol, extra train kwargs)
#
# ``rel_tol`` is how far the graphed loss may sit from the eager one, as a
# fraction of the loss magnitude, *on top of* whatever spread two eager runs
# show on their own. 0.0 means the family's whole training step is
# reproducible and capture must reproduce it exactly.
CASES = [
    ("yolo9", "LibreYOLO9", "t", 320, "detect", 0.0, {}),
    ("yolo9_p2", "LibreYOLO9P2", "t", 320, "detect", 0.0, {}),
    ("yolo9_e2e", "LibreYOLO9E2E", "t", 320, "detect", 0.0, {}),
    ("yolox", "LibreYOLOX", "t", 320, "detect", 0.0, {}),
    ("picodet", "LibrePICODET", "s", 320, "detect", 0.0, {}),
    ("yolo7", "LibreYOLO7", "b", 320, "detect", 0.0, {}),
    # YOLO-NAS captures bit-identically on a fixed batch (the unit suite gates
    # that), but its affine/mixup pipeline draws from generators the training
    # seed does not reach, so two eager train() runs already differ.
    ("yolonas", "LibreYOLONAS", "s", 320, "detect", 0.01, {}),
    # RTMDet shares its head convolutions across all three pyramid levels
    # (``share_conv=True`` aliases cls_convs[n][i].conv to cls_convs[0][i]),
    # so those two weights' gradient is a sum of three contributions. Eager
    # autograd and the graphed backward sum them in a different order, which
    # under fp16 differs in the last bits — 137 of 139 gradients are still
    # bit-identical — and the dynamic assigner's discrete choices amplify it.
    ("rtmdet", "LibreRTMDet", "t", 320, "detect", 0.05, {}),
    # Encoder-boundary capture. None of these reproduce their own eager run:
    # deformable attention accumulates its backward with atomics.
    # multi_scale=False because they resize every batch by default and a graph
    # is valid for exactly one input shape, so with jitter on a short run
    # never sees one shape often enough to capture at all.
    ("dfine", "LibreDFINE", "n", 320, "detect", 0.01, {"multi_scale": False}),
    ("deim", "LibreDEIM", "n", 320, "detect", 0.01, {"multi_scale": False}),
    ("deimv2", "LibreDEIMv2", "atto", 320, "detect", 0.01, {"multi_scale": False}),
    ("rtdetr", "LibreRTDETR", "r18", 320, "detect", 0.01, {}),
    ("rtdetrv2", "LibreRTDETRv2", "r18", 320, "detect", 0.01, {}),
    ("rtdetrv4", "LibreRTDETRv4", "s", 320, "detect", 0.01, {"multi_scale": False}),
    (
        "ec",
        "LibreEC",
        "s",
        320,
        "detect",
        0.01,
        {"multi_scale": False},
    ),
]


@pytest.fixture(scope="module")
def detect_dataset(tmp_path_factory) -> Path:
    """A generated 24-image YOLO-format detection dataset.

    Deliberately not a download: these tests must be able to run offline and
    must not depend on a dataset's contents staying fixed.
    """
    root = tmp_path_factory.mktemp("cuda_graph_train_data")
    rng = np.random.default_rng(0)
    for split, count in (("train", 16), ("valid", 8)):
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir(parents=True)
        for i in range(count):
            pixels = rng.integers(0, 255, (320, 320, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(root / split / "images" / f"{i:03d}.jpg")
            rows = []
            for _ in range(rng.integers(2, 6)):
                cx, cy = rng.uniform(0.2, 0.8, 2)
                w, h = rng.uniform(0.05, 0.15, 2)
                rows.append(f"{rng.integers(0, 2)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            (root / split / "labels" / f"{i:03d}.txt").write_text("\n".join(rows))
    (root / "data.yaml").write_text(
        yaml.dump(
            {
                "path": str(root),
                "train": "train/images",
                "val": "valid/images",
                "nc": 2,
                "names": ["a", "b"],
            }
        )
    )
    return root / "data.yaml"


def _train(
    class_name: str,
    size: str,
    imgsz: int,
    task: str,
    data: Path,
    save_dir: Path,
    cuda_graph: bool,
    extra: dict,
) -> dict:
    import libreyolo

    torch.manual_seed(0)
    model_kwargs = {"task": task} if task != "detect" else {}
    model = getattr(libreyolo, class_name)(None, size, **model_kwargs)
    return model.train(
        data=str(data),
        epochs=2,
        batch=4,
        imgsz=imgsz,
        device=0,
        workers=0,
        seed=0,
        project=str(save_dir),
        name=f"{'graph' if cuda_graph else 'eager'}",
        exist_ok=True,
        cuda_graph=cuda_graph,
        # Validation is not what this file measures, and mAP on random
        # images is meaningless; keep the runs short.
        eval_interval=100,
        **extra,
    )


@requires_cuda
@pytest.mark.external_data  # several families fetch a pretrained backbone
@pytest.mark.parametrize(
    "family,class_name,size,imgsz,task,deterministic,extra",
    CASES,
    ids=[case[0] for case in CASES],
)
def test_capture_engages_and_holds(
    family,
    class_name,
    size,
    imgsz,
    task,
    deterministic,
    extra,
    detect_dataset,
    tmp_path,
    caplog,
):
    """A real run must capture, keep the graph, and finish with a finite loss."""
    with caplog.at_level(logging.INFO, logger=CAPTURE_LOGGER):
        result = _train(
            class_name, size, imgsz, task, detect_dataset, tmp_path, True, extra
        )

    messages = [record.getMessage() for record in caplog.records]
    assert any("captured training forward/backward" in m for m in messages), (
        f"{family}: capture never engaged; log was {messages}"
    )
    disabled = [m for m in messages if "capture disabled" in m]
    assert not disabled, f"{family}: capture fell back to eager: {disabled}"
    assert np.isfinite(result["final_loss"]), f"{family}: {result['final_loss']}"


@requires_cuda
@pytest.mark.external_data
@pytest.mark.parametrize(
    "family,class_name,size,imgsz,task,rel_tol,extra",
    _cases(CASES),
    ids=[case[0] for case in CASES],
)
def test_loss_trajectory_stays_within_eager_noise(
    family, class_name, size, imgsz, task, rel_tol, extra, detect_dataset, tmp_path
):
    """Capture must not move the loss further than eager moves on its own.

    A fixed seed does not make every family's ``train()`` reproducible, so
    asserting equality across the board would test the family's determinism
    rather than the capture. The eager arm runs twice to measure that spread
    first; the graphed arm is then held to it plus the family's documented
    ``rel_tol``. A family listed at ``rel_tol=0`` must show zero spread and
    reproduce it exactly — if that ever stops holding, this fails and says so
    rather than quietly widening.

    A wrong capture boundary, a stale buffer or a dropped gradient moves the
    loss by tens of percent or produces NaN, so this band is loose enough to
    be stable and tight enough to catch every failure seen while building it.
    """
    eager_a = _train(
        class_name, size, imgsz, task, detect_dataset, tmp_path / "e1", False, extra
    )["epoch_losses"]
    eager_b = _train(
        class_name, size, imgsz, task, detect_dataset, tmp_path / "e2", False, extra
    )["epoch_losses"]
    graphed = _train(
        class_name, size, imgsz, task, detect_dataset, tmp_path / "g", True, extra
    )["epoch_losses"]

    assert all(np.isfinite(graphed)), f"{family}: {graphed}"
    noise = max(abs(a - b) for a, b in zip(eager_a, eager_b))
    gap = max(abs(a - g) for a, g in zip(eager_a, graphed))
    scale = max(abs(x) for x in eager_a)
    if rel_tol == 0.0:
        assert noise == 0.0, (
            f"{family} is listed as exactly reproducible but two eager runs "
            f"differ by {noise:.3e}: {eager_a} vs {eager_b}"
        )
    assert gap <= max(noise * 4.0, rel_tol * scale), (
        f"{family}: graph moved the loss by {gap:.3e}; eager's own spread is "
        f"{noise:.3e} and the allowance is {rel_tol:.3g} x {scale:.3g}"
        f"\n  eager  {eager_a}\n  eager2 {eager_b}\n  graph  {graphed}"
    )


# =============================================================================
# Non-detect tasks
# =============================================================================


@pytest.fixture(scope="module")
def classify_dataset(tmp_path_factory) -> Path:
    """A generated 3-class ImageFolder dataset."""
    root = tmp_path_factory.mktemp("cuda_graph_train_cls")
    rng = np.random.default_rng(1)
    for split, count in (("train", 18), ("val", 6)):
        for i in range(count):
            folder = root / split / f"class{i % 3}"
            folder.mkdir(parents=True, exist_ok=True)
            pixels = rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(folder / f"{i:03d}.jpg")
    return root


@pytest.fixture(scope="module")
def semantic_dataset(tmp_path_factory) -> Path:
    """A generated 3-class dense-mask dataset."""
    root = tmp_path_factory.mktemp("cuda_graph_train_sem")
    rng = np.random.default_rng(2)
    for split, count in (("train", 16), ("val", 4)):
        (root / "images" / split).mkdir(parents=True)
        (root / "masks" / split).mkdir(parents=True)
        for i in range(count):
            pixels = rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(root / "images" / split / f"{i:03d}.jpg")
            mask = np.zeros((256, 256), np.uint8)
            for label in (1, 2):
                y, x = rng.integers(0, 150, 2)
                mask[y : y + 80, x : x + 80] = label
            Image.fromarray(mask).save(root / "masks" / split / f"{i:03d}.png")
    (root / "data.yaml").write_text(
        yaml.dump(
            {
                "path": str(root),
                "train": "images/train",
                "val": "images/val",
                "masks_dir": "masks",
                "nc": 3,
                "names": ["background", "a", "b"],
            }
        )
    )
    return root / "data.yaml"


@pytest.fixture(scope="module")
def restore_dataset(tmp_path_factory) -> Path:
    """A generated paired noisy/clean restoration dataset."""
    root = tmp_path_factory.mktemp("cuda_graph_train_restore")
    rng = np.random.default_rng(3)
    for split, count in (("train", 12), ("val", 4)):
        (root / split / "inputs").mkdir(parents=True)
        (root / split / "targets").mkdir(parents=True)
        for i in range(count):
            clean = rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)
            noisy = np.clip(
                clean.astype(np.float32) + rng.normal(0, 12, clean.shape), 0, 255
            ).astype(np.uint8)
            Image.fromarray(noisy).save(root / split / "inputs" / f"{i:03d}.png")
            Image.fromarray(clean).save(root / split / "targets" / f"{i:03d}.png")
    (root / "data.yaml").write_text(
        yaml.dump(
            {
                "path": str(root),
                "train": "train/inputs",
                "val": "val/inputs",
                "input_dir": "inputs",
                "target_dir": "targets",
                "nc": 1,
                "names": ["restore"],
            }
        )
    )
    return root / "data.yaml"


# (family, class name, size, imgsz, task, dataset fixture, rel_tol, extra)
NON_DETECT_CASES = [
    ("resnet", "LibreResNet", "18", 224, "classify", "classify_dataset", 0.0, {}),
    ("convnext", "LibreConvNeXt", "t", 224, "classify", "classify_dataset", 0.0, {}),
    (
        "mobilenetv4",
        "LibreMobileNetV4",
        "s",
        224,
        "classify",
        "classify_dataset",
        0.0,
        {},
    ),
    (
        "efficientnetv2",
        "LibreEfficientNetV2",
        "b0",
        224,
        "classify",
        "classify_dataset",
        0.0,
        {},
    ),
    # NAFNet captures bit-identically on a fixed batch (graph_bench and the
    # unit suite gate that); its coupled crop/flip augmentation is what makes
    # two eager train() runs differ.
    ("nafnet", "LibreNAFNet", "s", 256, "restore", "restore_dataset", 0.01, {}),
    ("fomo", "LibreFOMO", "s", 96, "point", "detect_dataset", 0.01, {}),
    (
        "lingbotvision",
        "LibreLingBotVision",
        "s",
        224,
        "semantic",
        "semantic_dataset",
        0.01,
        {},
    ),
    # SegFormer's MiT encoder has stochastic depth inside the captured region,
    # so replay draws its own random stream (see TrainGraphManager's capture
    # log). Statistically equivalent, not reproducible against eager.
    (
        "segformer",
        "LibreSegformer",
        "b0",
        256,
        "semantic",
        "semantic_dataset",
        0.05,
        {},
    ),
]


@requires_cuda
@pytest.mark.external_data
@pytest.mark.parametrize(
    "family,class_name,size,imgsz,task,dataset_fixture,rel_tol,extra",
    _cases(NON_DETECT_CASES),
    ids=[case[0] for case in NON_DETECT_CASES],
)
def test_non_detect_capture_engages_and_holds(
    family,
    class_name,
    size,
    imgsz,
    task,
    dataset_fixture,
    rel_tol,
    extra,
    request,
    tmp_path,
    caplog,
):
    """Classification, restoration, point and semantic families capture too."""
    data = request.getfixturevalue(dataset_fixture)
    with caplog.at_level(logging.INFO, logger=CAPTURE_LOGGER):
        result = _train(class_name, size, imgsz, task, data, tmp_path, True, extra)

    messages = [record.getMessage() for record in caplog.records]
    assert any("captured training forward/backward" in m for m in messages), (
        f"{family}: capture never engaged; log was {messages}"
    )
    disabled = [m for m in messages if "capture disabled" in m]
    assert not disabled, f"{family}: capture fell back to eager: {disabled}"
    assert np.isfinite(result["final_loss"]), f"{family}: {result['final_loss']}"


@requires_cuda
@pytest.mark.external_data
@pytest.mark.parametrize(
    "family,class_name,size,imgsz,task,dataset_fixture,rel_tol,extra",
    _cases(NON_DETECT_CASES),
    ids=[case[0] for case in NON_DETECT_CASES],
)
def test_non_detect_loss_trajectory_stays_within_eager_noise(
    family,
    class_name,
    size,
    imgsz,
    task,
    dataset_fixture,
    rel_tol,
    extra,
    request,
    tmp_path,
):
    data = request.getfixturevalue(dataset_fixture)
    eager_a = _train(
        class_name, size, imgsz, task, data, tmp_path / "e1", False, extra
    )["epoch_losses"]
    eager_b = _train(
        class_name, size, imgsz, task, data, tmp_path / "e2", False, extra
    )["epoch_losses"]
    graphed = _train(class_name, size, imgsz, task, data, tmp_path / "g", True, extra)[
        "epoch_losses"
    ]

    assert all(np.isfinite(graphed)), f"{family}: {graphed}"
    noise = max(abs(a - b) for a, b in zip(eager_a, eager_b))
    gap = max(abs(a - g) for a, g in zip(eager_a, graphed))
    scale = max(abs(x) for x in eager_a)
    if rel_tol == 0.0:
        assert noise == 0.0, (
            f"{family} is listed as exactly reproducible but two eager runs "
            f"differ by {noise:.3e}: {eager_a} vs {eager_b}"
        )
    assert gap <= max(noise * 4.0, rel_tol * scale), (
        f"{family}: graph moved the loss by {gap:.3e}; eager's own spread is "
        f"{noise:.3e} and the allowance is {rel_tol:.3g} x {scale:.3g}"
        f"\n  eager  {eager_a}\n  eager2 {eager_b}\n  graph  {graphed}"
    )


@requires_cuda
@pytest.mark.training_nightly
def test_capture_engages_without_amp(detect_dataset, tmp_path, caplog):
    """``amp=False`` is a supported path and must still capture.

    Everything else in this file runs under AMP, the default. Without it
    ``trainer.scaler`` is None and the autocast context is never entered, so
    the capture path takes different branches in the trainer.

    No loss assertion here on purpose: at fp32 this hardware does not
    reproduce its own eager run (cuDNN picks a nondeterministic weight-
    gradient algorithm for some shapes), so equality would test that noise
    rather than the capture. See docs/training_cuda_graphs.md.
    """
    import libreyolo

    torch.manual_seed(0)
    model = libreyolo.LibreYOLO9(None, "t")
    with caplog.at_level(logging.INFO, logger=CAPTURE_LOGGER):
        result = model.train(
            data=str(detect_dataset),
            epochs=2,
            batch=4,
            imgsz=320,
            device=0,
            workers=0,
            seed=0,
            amp=False,
            project=str(tmp_path),
            name="fp32",
            exist_ok=True,
            cuda_graph=True,
            eval_interval=100,
        )
    messages = [record.getMessage() for record in caplog.records]
    assert any("captured training forward/backward" in m for m in messages), messages
    assert not [m for m in messages if "capture disabled" in m], messages
    assert all(np.isfinite(result["epoch_losses"])), result["epoch_losses"]


@requires_cuda
@pytest.mark.training_nightly
def test_layer_freezing_matches_eager(detect_dataset, tmp_path):
    """Frozen layers change which parameters get gradients.

    ``make_graphed_callables`` is handed the whole adapter, so a frozen
    backbone means most of its parameters want no gradient. That path runs
    on ``allow_unused_input=True``; this checks it produces the eager answer
    rather than silently dropping live gradients too.
    """
    import libreyolo

    def run(cuda_graph):
        torch.manual_seed(0)
        model = libreyolo.LibreYOLO9(None, "t")
        return model.train(
            data=str(detect_dataset),
            epochs=2,
            batch=4,
            imgsz=320,
            device=0,
            workers=0,
            seed=0,
            freeze="10",
            project=str(tmp_path),
            name="graph" if cuda_graph else "eager",
            exist_ok=True,
            cuda_graph=cuda_graph,
            eval_interval=100,
        )["epoch_losses"]

    assert run(False) == run(True)


@requires_cuda
@pytest.mark.training_nightly
def test_resume_from_checkpoint_recaptures(detect_dataset, tmp_path):
    """A resumed run must capture again and keep training.

    Capture records parameter addresses, so a checkpoint load that replaced
    tensors instead of copying into them would leave the graph reading freed
    storage. Resume loads before ``setup()``, so capture always sees the
    final tensors, but that ordering is worth a test rather than a comment.
    """
    import libreyolo

    torch.manual_seed(0)
    first = libreyolo.LibreYOLO9(None, "t").train(
        data=str(detect_dataset),
        epochs=2,
        batch=4,
        imgsz=320,
        device=0,
        workers=0,
        seed=0,
        project=str(tmp_path),
        name="first",
        exist_ok=True,
        cuda_graph=True,
        eval_interval=100,
    )

    torch.manual_seed(0)
    resumed = libreyolo.LibreYOLO9(first["last_checkpoint"], "t").train(
        data=str(detect_dataset),
        epochs=4,
        batch=4,
        imgsz=320,
        device=0,
        workers=0,
        seed=0,
        project=str(tmp_path),
        name="second",
        exist_ok=True,
        cuda_graph=True,
        resume=True,
        eval_interval=100,
    )
    assert resumed["epoch_losses"], "resume produced no epochs"
    assert all(np.isfinite(resumed["epoch_losses"])), resumed["epoch_losses"]


@requires_cuda
@pytest.mark.training_nightly
def test_capture_failure_does_not_kill_the_run(detect_dataset, tmp_path, caplog):
    """Capture is opt-in and best-effort: failing it must cost speed, not the run.

    The realistic trigger is an out-of-memory while allocating the graph's
    static buffers, which is exactly when a user least wants their training
    to die.
    """
    import libreyolo

    def exploding(*args, **kwargs):
        raise torch.cuda.OutOfMemoryError("simulated OOM during capture")

    original = torch.cuda.make_graphed_callables
    torch.cuda.make_graphed_callables = exploding
    try:
        with caplog.at_level(logging.INFO, logger=CAPTURE_LOGGER):
            torch.manual_seed(0)
            result = libreyolo.LibreYOLO9(None, "t").train(
                data=str(detect_dataset),
                epochs=2,
                batch=4,
                imgsz=320,
                device=0,
                workers=0,
                seed=0,
                project=str(tmp_path),
                name="oom",
                exist_ok=True,
                cuda_graph=True,
                eval_interval=100,
            )
    finally:
        torch.cuda.make_graphed_callables = original

    messages = [record.getMessage() for record in caplog.records]
    disabled = [m for m in messages if "capture disabled" in m]
    assert len(disabled) == 1, f"expected exactly one warning, got {disabled}"
    assert all(np.isfinite(result["epoch_losses"])), result["epoch_losses"]


@requires_cuda
@pytest.mark.training_nightly
def test_gradient_accumulation_matches_eager(detect_dataset, tmp_path):
    """Capture must be safe when ``zero_grad`` runs once per window.

    The plain loop zeroes gradients after every forward, so capture warm-up
    could not pollute them even if it wrote ``.grad``. The accumulation loop
    zeroes only at the start of a window, so the safety here rests on
    ``make_graphed_callables`` warming up through ``torch.autograd.grad``
    (which returns gradients rather than accumulating them). This asserts
    that rather than trusting it.

    Replay is the second hazard: the graphed backward hands its static
    gradient buffers to autograd, which adopts them as ``.grad`` at the start
    of a window, so an unprotected replay turns the window's gradient into
    twice the last micro-batch's. 16 images at batch 4 make one window per
    epoch, captured mid-window in epoch 1; epoch 2 is the first fully replayed
    window, and only epoch 3's losses see the optimizer step it produced, so
    three epochs are the minimum that can detect it.
    """
    import libreyolo

    def run(cuda_graph):
        torch.manual_seed(0)
        model = libreyolo.LibreYOLO9(None, "t")
        return model.train(
            data=str(detect_dataset),
            epochs=3,
            batch=4,
            nbs=16,  # 4 micro-batches per optimizer step
            imgsz=320,
            device=0,
            workers=0,
            seed=0,
            project=str(tmp_path),
            name="graph" if cuda_graph else "eager",
            exist_ok=True,
            cuda_graph=cuda_graph,
            eval_interval=100,
        )["epoch_losses"]

    assert run(False) == run(True)


@requires_cuda
@pytest.mark.training_nightly
def test_in_training_validation_runs_with_a_live_graph(detect_dataset, tmp_path):
    """Per-epoch validation must still work while a graph is captured.

    Capture rebinds the forward of the *adapter* module, never the family
    model's, precisely so validation, EMA evaluation and checkpointing keep
    running the eager forward. This asserts that end to end rather than
    trusting the design: the other cases in this file all disable validation
    to stay fast.
    """
    import libreyolo

    torch.manual_seed(0)
    model = libreyolo.LibreYOLO9(None, "t")
    result = model.train(
        data=str(detect_dataset),
        epochs=2,
        batch=4,
        imgsz=320,
        device=0,
        workers=0,
        seed=0,
        project=str(tmp_path),
        name="valgraph",
        exist_ok=True,
        cuda_graph=True,
        eval_interval=1,
    )
    assert len(result["val_metrics"]) == 2, result["val_metrics"]
    assert Path(result["last_checkpoint"]).exists()


@requires_cuda
@pytest.mark.training_nightly
def test_yolox_capture_survives_the_mosaic_close_switch(detect_dataset, tmp_path):
    """YOLOX turns on its L1 branch mid-run; the graph must be rebuilt there.

    ``head.use_l1`` adds the origin_preds tensors to what the captured
    region returns. Replaying the pre-switch graph would keep training the
    pre-switch network, and re-capturing without restoring the eager forward
    fails with "Cannot prepare for replay during capturing stage" and
    silently disables capture for the rest of the run.
    """
    import libreyolo

    def run(cuda_graph):
        torch.manual_seed(0)
        model = libreyolo.LibreYOLOX(None, "t")
        return model.train(
            data=str(detect_dataset),
            epochs=4,
            no_aug_epochs=2,  # mosaic closes at epoch 2 -> use_l1 flips
            batch=4,
            imgsz=320,
            device=0,
            workers=0,
            seed=0,
            project=str(tmp_path),
            name="graph" if cuda_graph else "eager",
            exist_ok=True,
            cuda_graph=cuda_graph,
            eval_interval=100,
        )

    caplog_logger = logging.getLogger(CAPTURE_LOGGER)
    records: list[str] = []

    class _Collect(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    handler = _Collect()
    caplog_logger.addHandler(handler)
    previous_level = caplog_logger.level
    caplog_logger.setLevel(logging.INFO)
    try:
        eager = run(False)
        graphed = run(True)
    finally:
        caplog_logger.removeHandler(handler)
        caplog_logger.setLevel(previous_level)

    assert sum("captured training forward/backward" in m for m in records) >= 2, (
        f"expected a re-capture after the switch; log was {records}"
    )
    assert any("capture invalidated" in m for m in records), records
    assert not [m for m in records if "capture disabled" in m], records
    assert eager["epoch_losses"] == graphed["epoch_losses"], (
        f"eager {eager['epoch_losses']} != graphed {graphed['epoch_losses']}"
    )
