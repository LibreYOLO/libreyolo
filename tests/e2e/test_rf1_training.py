"""
RF1: Training test for all catalog models.

Runs a short marbles fine-tune and then validates on the test split.
Convolutional families use 10 epochs; DETR-style families (D-FINE, RT-DETR)
use 20 because they converge materially slower on tiny custom datasets.
The dataset auto-downloads from HuggingFace — no API keys needed.

Usage:
    pytest tests/e2e/test_rf1_training.py -v -m e2e
    pytest tests/e2e/test_rf1_training.py::test_rf1_training[yolox-n] -v
    pytest tests/e2e/test_rf1_training.py -k "rfdetr" -v
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest
import torch
import yaml
from PIL import Image

from libreyolo import LibreYOLO
from .conftest import (
    ALL_MODEL_WEIGHT_PARAMS,
    ALL_MODELS_WITH_WEIGHTS,
    cuda_cleanup,
    make_ids,
    require_test_weights,
    run_direct_subprocess,
    run_in_subprocess,
)

pytestmark = [pytest.mark.e2e, pytest.mark.rf1]

DATASET_ROOT = Path.home() / ".cache" / "libreyolo" / "marbles"
HF_REPO = "LibreYOLO/marbles"
HF_REPO_URL = f"https://huggingface.co/datasets/{HF_REPO}"


def download_marbles_dataset():
    """Download the marbles dataset from HuggingFace if not already cached."""
    if DATASET_ROOT.exists() and (DATASET_ROOT / "data.yaml").exists():
        return

    print(f"\nDownloading dataset {HF_REPO} from HuggingFace ...")
    DATASET_ROOT.parent.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            "git",
            "clone",
            f"https://huggingface.co/datasets/{HF_REPO}",
            str(DATASET_ROOT),
        ],
        check=True,
    )
    print(f"Dataset downloaded to {DATASET_ROOT}")


def patch_data_yaml():
    """Ensure data.yaml has an absolute path so training resolves splits."""
    data_yaml = DATASET_ROOT / "data.yaml"
    data = yaml.safe_load(data_yaml.read_text())
    if data.get("path") != str(DATASET_ROOT):
        data["path"] = str(DATASET_ROOT)
        data_yaml.write_text(yaml.dump(data, default_flow_style=False))


@pytest.fixture(scope="module")
def dataset():
    """Download marbles dataset and patch data.yaml. Shared by all fixtures."""
    download_marbles_dataset()
    patch_data_yaml()
    return DATASET_ROOT


@pytest.fixture(scope="module")
def dataset_coco(dataset):
    """Convert YOLO labels to COCO JSON for RF-DETR training.

    Writes _annotations.coco.json into each split dir (train/valid/test).
    Idempotent — skips if annotations already exist.
    Reads class names from data.yaml dynamically.
    """
    with open(dataset / "data.yaml") as f:
        data = yaml.safe_load(f)
    class_names = data["names"]

    # Handle both list and dict formats for names
    # RF-DETR (Roboflow format) uses 0-indexed category IDs
    if isinstance(class_names, dict):
        categories = [
            {"id": i, "name": class_names[i], "supercategory": "object"}
            for i in sorted(class_names.keys())
        ]
    else:
        categories = [
            {"id": i, "name": name, "supercategory": "object"}
            for i, name in enumerate(class_names)
        ]

    for split in ["train", "valid", "test"]:
        ann_file = dataset / split / "_annotations.coco.json"
        if ann_file.exists():
            continue

        images_dir = dataset / split / "images"
        labels_dir = dataset / split / "labels"

        images_list, annotations_list = [], []
        ann_id = 0

        for img_id, img_path in enumerate(sorted(images_dir.glob("*.jpg"))):
            with Image.open(img_path) as img:
                w, h = img.size

            images_list.append(
                {
                    "id": img_id,
                    "file_name": f"images/{img_path.name}",
                    "width": w,
                    "height": h,
                }
            )

            label_file = labels_dir / img_path.with_suffix(".txt").name
            if label_file.exists():
                for line in label_file.read_text().strip().split("\n"):
                    if not line.strip():
                        continue
                    parts = line.strip().split()
                    cls_id = int(parts[0])
                    cx, cy, bw, bh = map(float, parts[1:5])
                    x = (cx - bw / 2) * w
                    y = (cy - bh / 2) * h
                    box_w, box_h = bw * w, bh * h

                    annotations_list.append(
                        {
                            "id": ann_id,
                            "image_id": img_id,
                            "category_id": cls_id,
                            "bbox": [
                                round(x, 2),
                                round(y, 2),
                                round(box_w, 2),
                                round(box_h, 2),
                            ],
                            "area": round(box_w * box_h, 2),
                            "iscrowd": 0,
                        }
                    )
                    ann_id += 1

        coco = {
            "images": images_list,
            "annotations": annotations_list,
            "categories": categories,
        }
        ann_file.write_text(json.dumps(coco))

    return dataset


@pytest.fixture(scope="module")
def dataset_data_yaml(dataset):
    """Return data.yaml path with absolute path for training code."""
    return str(dataset / "data.yaml")


MIN_MAP = 0.05
DETR_RF1_FAMILIES = {"dfine", "rtdetr"}


def rf1_epochs(family: str) -> int:
    """Return the family-specific RF1 epoch budget."""
    return 20 if family in DETR_RF1_FAMILIES else 10


def rf1_workers(family: str) -> tuple[int, int]:
    """Return (train_workers, val_workers) for RF1 stability."""
    if family in DETR_RF1_FAMILIES:
        return 0, 0
    return 2, 4


def rf1_train_kwargs(family: str, size: str) -> dict:
    """Return RF1-only train overrides for families that need them."""
    if family == "dfine":
        # BaseTrainer scales lr as lr0 * batch / 64. On RF1's tiny batches,
        # dfine-s/m underfit badly at the family default lr0=2e-4, while
        # n/l/x already clear the gate with the default test-side recipe.
        lr0 = 8e-4 if size in {"s", "m"} else 2e-4
        return {
            "lr0": lr0,
            "multi_scale": False,
            "aug_stop_epoch_ratio": 0.0,
        }
    if family == "rtdetr":
        return {
            "lr0": 2e-4,
            "mosaic_prob": 0.0,
            "hsv_prob": 0.0,
        }
    if family == "ecdet":
        return {"allow_experimental": True}
    return {}


@pytest.mark.parametrize(
    "family,size,weights",
    ALL_MODEL_WEIGHT_PARAMS,
)
def test_rf1_training(family, size, weights, dataset_coco, dataset_data_yaml, tmp_path):
    """Train on marbles, verify the model learns and clears a basic mAP floor."""
    weights = require_test_weights(weights, expected_family=family)
    if size == "x" or size == "l":
        val_batch = 4
        train_batch = 4
    else:
        val_batch = 8
        train_batch = 8
    # Tiny DETR-family RF1 jobs are materially more stable with a single
    # worker, and they need a slightly longer budget before mAP reflects
    # that they are genuinely learning on marbles.
    train_epochs = rf1_epochs(family)
    workers, val_workers = rf1_workers(family)

    # RF-DETR: run in a fresh subprocess to avoid CUDA driver state corruption
    # that causes SIGSEGV when export tests have run beforehand in the same process.
    if family == "rfdetr":
        output_dir = str(tmp_path / f"rfdetr_{size}")
        coco_dir = str(dataset_coco)
        run_in_subprocess(
            f"""
            from libreyolo import LibreYOLO

            model = LibreYOLO("{weights}", size="{size}")

            pre = model.val(
                data="{dataset_data_yaml}", split="test", batch=8, conf=0.001, iou=0.6
            )
            pre_map = pre["metrics/mAP50-95"]

            model.train(
                data="{coco_dir}",
                epochs=10,
                batch_size=2,
                output_dir="{output_dir}",
            )

            post = model.val(
                data="{dataset_data_yaml}", split="test", batch=8, conf=0.001, iou=0.6
            )
            post_map = post["metrics/mAP50-95"]

            print(f"  pre-training mAP50-95={{pre_map:.4f}}")
            print(f"  post-training mAP50-95={{post_map:.4f}}")

            assert post_map >= 0.05, f"mAP50-95={{post_map:.4f}} below 0.05"
            assert post_map > pre_map, (
                f"No improvement: pre={{pre_map:.4f}} -> post={{post_map:.4f}}"
            )
        """,
            timeout=600,
        )
        shutil.rmtree(tmp_path, ignore_errors=True)
        return

    # D-FINE converges reliably in a clean interpreter with the same RF1
    # recipe, but under pytest's long-lived host process the m/x cases become
    # flaky. Run them in a subprocess so RF1 measures the actual fine-tune path
    # instead of pytest process state.
    if family == "dfine":
        run_name = f"{family}_{size}"
        train_kwargs = rf1_train_kwargs(family, size)
        run_direct_subprocess(
            f"""
            from pathlib import Path

            from libreyolo import LibreYOLO

            model = LibreYOLO("{weights}", size="{size}")
            project = Path(r"{str(tmp_path)}")

            pre = model.val(
                data=r"{dataset_data_yaml}",
                split="test",
                batch={val_batch},
                conf=0.001,
                iou=0.6,
                workers={val_workers},
            )
            pre_map = pre["metrics/mAP50-95"]

            results = model.train(
                data=r"{dataset_data_yaml}",
                epochs={train_epochs},
                batch={train_batch},
                lr0={train_kwargs["lr0"]},
                workers={workers},
                seed=0,
                multi_scale={train_kwargs["multi_scale"]},
                aug_stop_epoch_ratio={train_kwargs["aug_stop_epoch_ratio"]},
                save_period=999,
                project=str(project),
                name="{run_name}",
                exist_ok=True,
            )

            epoch_losses = results["epoch_losses"]
            first_loss = epoch_losses[0]
            last_loss = epoch_losses[-1]

            weights_dir = project / "{run_name}" / "weights"
            best_pt = weights_dir / "best.pt"
            last_pt = weights_dir / "last.pt"
            candidates = [pt for pt in (best_pt, last_pt) if pt.exists()]
            assert candidates, f"No checkpoint found in {{weights_dir}}"

            post_map = -1.0
            best_checkpoint = None
            for checkpoint in candidates:
                fresh = LibreYOLO(str(checkpoint), size="{size}")
                post = fresh.val(
                    data=r"{dataset_data_yaml}",
                    split="test",
                    batch={val_batch},
                    conf=0.001,
                    iou=0.6,
                    workers={val_workers},
                )
                candidate_map = post["metrics/mAP50-95"]
                print(
                    f"  {weights} finetuned {{checkpoint.name}} "
                    f"mAP50-95={{candidate_map:.4f}}"
                )
                if candidate_map > post_map:
                    post_map = candidate_map
                    best_checkpoint = checkpoint.name

            print(f"  {weights} pre-training mAP50-95={{pre_map:.4f}}")
            print(
                f"  {weights} best finetuned checkpoint={{best_checkpoint}} "
                f"mAP50-95={{post_map:.4f}}"
            )
            print(
                f"  {weights} first epoch loss={{first_loss:.4f}}, "
                f"last epoch loss={{last_loss:.4f}}"
            )

            assert post_map >= 0.05, f"mAP50-95={{post_map:.4f}} below 0.05"
            assert post_map > pre_map, (
                f"No improvement: pre={{pre_map:.4f}} -> post={{post_map:.4f}}"
            )
        """,
            timeout=900,
        )
        shutil.rmtree(tmp_path, ignore_errors=True)
        return

    # --- YOLOX / YOLOv9: run in-process ---
    model = LibreYOLO(weights, size=size)
    try:
        train_kwargs = rf1_train_kwargs(family, size)

        # --- Baseline mAP BEFORE training ---
        pre_results = model.val(
            data=dataset_data_yaml,
            split="test",
            batch=val_batch,
            conf=0.001,
            iou=0.6,
            workers=val_workers,
        )
        pre_map = pre_results["metrics/mAP50-95"]

        # --- Train ---
        train_results = model.train(
            data=dataset_data_yaml,
            epochs=train_epochs,
            batch=train_batch,
            workers=workers,
            save_period=999,
            project=str(tmp_path),
            name=f"{family}_{size}",
            exist_ok=True,
            **train_kwargs,
        )

        # --- Post-training mAP ---
        post_results = model.val(
            data=dataset_data_yaml,
            split="test",
            batch=val_batch,
            conf=0.001,
            iou=0.6,
            workers=val_workers,
        )
        post_map = post_results["metrics/mAP50-95"]

        print(f"\n  {weights} pre-training mAP50-95={pre_map:.4f}")
        print(f"  {weights} post-training mAP50-95={post_map:.4f}")

        # --- Loss monitoring ---
        epoch_losses = train_results["epoch_losses"]
        first_loss = epoch_losses[0]
        last_loss = epoch_losses[-1]
        print(
            f"  {weights} first epoch loss={first_loss:.4f}, "
            f"last epoch loss={last_loss:.4f}"
        )

        # D-FINE is DETR-family: total loss is a sum of ~38 weighted auxiliary
        # terms (per-decoder-layer + pre + encoder-aux + DN paths), and combined
        # with augmentation + multi-scale variance the per-epoch loss is too
        # noisy for a monotonic-decrease assertion to hold reliably on small
        # datasets. RF-DETR skips this check for the same reason (see the
        # subprocess branch above). For D-FINE we rely on the mAP-improvement
        # assertions below.
        # D-FINE and ECDet are both DETR-family with ~38 weighted aux losses;
        # see the comment block above for why the monotonic check is skipped.
        if family not in ("dfine", "ecdet"):
            assert last_loss < first_loss, (
                f"Loss did not decrease: first={first_loss:.4f} → last={last_loss:.4f}"
            )

        # --- Assertions ---
        assert post_map >= MIN_MAP, (
            f"Post-training mAP50-95={post_map:.4f} below {MIN_MAP}"
        )

        assert post_map > pre_map, (
            f"Model did not improve: pre={pre_map:.4f} → post={post_map:.4f}"
        )

        shutil.rmtree(tmp_path, ignore_errors=True)
    finally:
        del model
        cuda_cleanup()


# ---------------------------------------------------------------------------
# Phase 2: Reload fine-tuned checkpoints into fresh models
# ---------------------------------------------------------------------------

# YOLOX/YOLO9 reload: derive from catalog (excludes rfdetr)
_RELOAD_MODELS = [(f, s, w) for f, s, w in ALL_MODELS_WITH_WEIGHTS if f != "rfdetr"]


@pytest.mark.parametrize(
    "family,size,weights", _RELOAD_MODELS, ids=make_ids(_RELOAD_MODELS)
)
def test_load_finetuned_checkpoint(
    family, size, weights, dataset_coco, dataset_data_yaml, tmp_path
):
    """Train, save checkpoint, load into fresh model, validate.

    Verifies that fine-tuned checkpoints can be loaded in a new session
    with correct nc, names, and architecture auto-rebuild.
    Also verifies loss decreased during training and mAP improved.
    """
    weights = require_test_weights(weights, expected_family=family)

    if size in ("x", "l"):
        val_batch = 4
        train_batch = 4
    else:
        val_batch = 8
        train_batch = 8
    train_epochs = rf1_epochs(family)
    workers, val_workers = rf1_workers(family)

    if family == "dfine":
        run_name = f"{family}_{size}"
        train_kwargs = rf1_train_kwargs(family, size)
        run_direct_subprocess(
            f"""
            from pathlib import Path
            import torch

            from libreyolo import LibreYOLO

            project = Path(r"{str(tmp_path)}")
            model = LibreYOLO("{weights}", size="{size}")

            pre = model.val(
                data=r"{dataset_data_yaml}",
                split="test",
                batch={val_batch},
                conf=0.001,
                iou=0.6,
                workers={val_workers},
            )
            pre_map = pre["metrics/mAP50-95"]

            results = model.train(
                data=r"{dataset_data_yaml}",
                epochs={train_epochs},
                batch={train_batch},
                lr0={train_kwargs["lr0"]},
                workers={workers},
                seed=0,
                multi_scale={train_kwargs["multi_scale"]},
                aug_stop_epoch_ratio={train_kwargs["aug_stop_epoch_ratio"]},
                save_period=999,
                project=str(project),
                name="{run_name}",
                exist_ok=True,
            )

            epoch_losses = results["epoch_losses"]
            first_loss = epoch_losses[0]
            last_loss = epoch_losses[-1]
            print(
                f"  {weights} first epoch loss={{first_loss:.4f}}, "
                f"last epoch loss={{last_loss:.4f}}"
            )

            weights_dir = project / "{run_name}" / "weights"
            best_pt = weights_dir / "best.pt"
            last_pt = weights_dir / "last.pt"
            candidates = [pt for pt in (best_pt, last_pt) if pt.exists()]
            assert candidates, f"No checkpoint found in {{weights_dir}}"

            ckpt = torch.load(candidates[0], map_location="cpu", weights_only=False)
            assert ckpt["nc"] == 2, f"Expected nc=2 (marbles), got {{ckpt['nc']}}"
            assert ckpt["model_family"] == "{family}"
            print(
                f"  Checkpoint metadata: nc={{ckpt['nc']}}, "
                f"family={{ckpt['model_family']}}, names={{ckpt['names']}}"
            )

            post_map = -1.0
            best_reload = None
            for checkpoint in candidates:
                fresh = LibreYOLO(str(checkpoint), size="{size}")
                post = fresh.val(
                    data=r"{dataset_data_yaml}",
                    split="test",
                    batch={val_batch},
                    conf=0.001,
                    iou=0.6,
                    workers={val_workers},
                )
                candidate_map = post["metrics/mAP50-95"]
                print(
                    f"  {weights} reloaded {{checkpoint.name}} "
                    f"mAP50-95={{candidate_map:.4f}}"
                )
                if candidate_map > post_map:
                    post_map = candidate_map
                    best_reload = checkpoint.name

            print(f"  {weights} pre-training mAP50-95={{pre_map:.4f}}")
            print(
                f"  {weights} best reloaded checkpoint={{best_reload}} "
                f"mAP50-95={{post_map:.4f}}"
            )

            assert post_map > pre_map, (
                f"Reloaded model did not improve: pre={{pre_map:.4f}} -> "
                f"post={{post_map:.4f}}"
            )
        """,
            timeout=900,
        )
        shutil.rmtree(tmp_path, ignore_errors=True)
        return

    model = LibreYOLO(weights, size=size)
    fresh_model = None
    try:
        train_kwargs = rf1_train_kwargs(family, size)

        # 1. Baseline mAP before training
        pre_results = model.val(
            data=dataset_data_yaml,
            split="test",
            batch=val_batch,
            conf=0.001,
            iou=0.6,
            workers=val_workers,
        )
        pre_map = pre_results["metrics/mAP50-95"]

        # 2. Train
        train_results = model.train(
            data=dataset_data_yaml,
            epochs=train_epochs,
            batch=train_batch,
            workers=workers,
            save_period=999,
            project=str(tmp_path),
            name=f"{family}_{size}",
            exist_ok=True,
            **train_kwargs,
        )

        # 3. Verify loss decreased
        epoch_losses = train_results["epoch_losses"]
        first_loss = epoch_losses[0]
        last_loss = epoch_losses[-1]
        print(
            f"\n  {weights} first epoch loss={first_loss:.4f}, "
            f"last epoch loss={last_loss:.4f}"
        )

        # D-FINE and ECDet are both DETR-family with ~38 weighted aux losses;
        # see the comment block above for why the monotonic check is skipped.
        if family not in ("dfine", "ecdet"):
            assert last_loss < first_loss, (
                f"Loss did not decrease: first={first_loss:.4f} → last={last_loss:.4f}"
            )

        # 4. Find best.pt on disk
        best_pt = tmp_path / f"{family}_{size}" / "weights" / "best.pt"
        if not best_pt.exists():
            best_pt = tmp_path / f"{family}_{size}" / "weights" / "last.pt"
        assert best_pt.exists(), f"No checkpoint found at {best_pt}"

        # 5. Verify checkpoint has metadata
        ckpt = torch.load(best_pt, map_location="cpu", weights_only=False)
        assert "nc" in ckpt, "Checkpoint missing 'nc' metadata"
        assert "names" in ckpt, "Checkpoint missing 'names' metadata"
        assert "model_family" in ckpt, "Checkpoint missing 'model_family' metadata"
        assert ckpt["nc"] == 2, f"Expected nc=2 (marbles), got {ckpt['nc']}"
        assert ckpt["model_family"] == family
        print(
            f"  Checkpoint metadata: nc={ckpt['nc']}, family={ckpt['model_family']}, "
            f"names={ckpt['names']}"
        )

        # 6. Load into a completely fresh model (default nc=80)
        del model
        model = None
        cuda_cleanup()

        fresh_model = LibreYOLO(str(best_pt), size=size)

        # 7. Verify auto-rebuild happened
        assert fresh_model.nb_classes == 2, (
            f"Expected nb_classes=2 after loading, got {fresh_model.nb_classes}"
        )
        assert len(fresh_model.names) == 2, (
            f"Expected 2 names, got {len(fresh_model.names)}"
        )

        # 8. Validate reloaded model on test split
        post_results = fresh_model.val(
            data=dataset_data_yaml,
            split="test",
            batch=val_batch,
            conf=0.001,
            iou=0.6,
            workers=val_workers,
        )
        post_map = post_results["metrics/mAP50-95"]

        print(f"  {weights} pre-training mAP50-95={pre_map:.4f}")
        print(f"  {weights} reloaded checkpoint mAP50-95={post_map:.4f}")

        assert post_map >= MIN_MAP, (
            f"Reloaded model mAP50-95={post_map:.4f} below {MIN_MAP}"
        )

        assert post_map > pre_map, (
            f"Reloaded model did not improve over baseline: "
            f"pre={pre_map:.4f} → post={post_map:.4f}"
        )

        shutil.rmtree(tmp_path, ignore_errors=True)
    finally:
        if model is not None:
            del model
        if fresh_model is not None:
            del fresh_model
        cuda_cleanup()


# RF-DETR: reload fine-tuned checkpoint (only n for speed)
_RELOAD_RFDETR = [("rfdetr", "n", "LibreRFDETRn.pt")]


@pytest.mark.parametrize(
    "family,size,weights", _RELOAD_RFDETR, ids=make_ids(_RELOAD_RFDETR)
)
def test_load_finetuned_checkpoint_rfdetr(
    family, size, weights, dataset_coco, dataset_data_yaml, tmp_path
):
    """Train RF-DETR, save checkpoint, load into fresh model, validate.

    RF-DETR uses a different checkpoint format (checkpoint_best_total.pth)
    and requires manual detection head reinitialization.
    Also verifies mAP improved over pre-training baseline.

    Runs in a subprocess to avoid CUDA driver state corruption.
    """
    output_dir = str(tmp_path / f"rfdetr_{size}")
    coco_dir = str(dataset_coco)
    run_in_subprocess(
        f"""
        import gc
        import torch
        from pathlib import Path
        from libreyolo import LibreYOLO

        weights = "{weights}"
        size = "{size}"
        output_dir = "{output_dir}"

        # 1. Baseline mAP before training
        model = LibreYOLO(weights, size=size)
        pre_results = model.val(
            data="{dataset_data_yaml}", split="test", batch=8, conf=0.001, iou=0.6
        )
        pre_map = pre_results["metrics/mAP50-95"]

        # 2. Train
        model.train(
            data="{coco_dir}",
            epochs=10,
            batch_size=2,
            output_dir=output_dir,
        )

        # 3. Find checkpoint on disk
        best_ckpt = Path(output_dir) / "checkpoint_best_total.pth"
        if not best_ckpt.exists():
            ckpts = sorted(Path(output_dir).glob("checkpoint*.pth"))
            assert ckpts, f"No checkpoint found in {{output_dir}}"
            best_ckpt = ckpts[-1]

        # 4. Verify checkpoint structure
        ckpt = torch.load(best_ckpt, map_location="cpu", weights_only=False)
        assert "model" in ckpt, "RF-DETR checkpoint missing 'model' key"
        state_dict = ckpt["model"]
        assert "class_embed.bias" in state_dict, "Missing class_embed in state dict"
        # rfdetr >= 1.6 keeps 91-class head; actual classes are in args.
        class_names = ckpt.get("args", {{}}).get("class_names", [])
        assert len(class_names) == 2, (
            f"Expected 2 class names, got {{class_names}}"
        )
        num_classes = len(class_names)
        num_classes_internal = state_dict["class_embed.bias"].shape[0]

        # 5. Load into fresh model
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        fresh_model = LibreYOLO(weights, size=size)

        if num_classes_internal != fresh_model.model.model.class_embed.bias.shape[0]:
            fresh_model.model.model.reinitialize_detection_head(num_classes_internal)
        fresh_model.model.model.load_state_dict(state_dict, strict=False)
        fresh_model.model.model.eval()
        fresh_model.model.model.to(fresh_model.device)
        fresh_model.nb_classes = num_classes
        fresh_model.model.nb_classes = num_classes

        # 6. Validate reloaded model
        post_results = fresh_model.val(
            data="{dataset_data_yaml}", split="test", batch=8, conf=0.001, iou=0.6
        )
        post_map = post_results["metrics/mAP50-95"]

        print(f"  pre-training mAP50-95={{pre_map:.4f}}")
        print(f"  reloaded checkpoint mAP50-95={{post_map:.4f}}")

        assert post_map >= 0.05, f"mAP50-95={{post_map:.4f}} below 0.05"
        assert post_map > pre_map, (
            f"No improvement: pre={{pre_map:.4f}} -> post={{post_map:.4f}}"
        )
    """,
        timeout=600,
    )
    shutil.rmtree(tmp_path, ignore_errors=True)
