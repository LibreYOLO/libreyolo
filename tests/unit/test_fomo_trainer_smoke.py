"""Unit tests for LibreFOMO training orchestration and configurations."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
import torch

pytestmark = [pytest.mark.unit, pytest.mark.fomo]

if TYPE_CHECKING:
    from libreyolo.models.fomo.model import LibreFOMO


def _make_random_fomo(size: str = "s", nc: int = 1) -> "LibreFOMO":
    from libreyolo.models.fomo.model import LibreFOMO

    return LibreFOMO(model_path=None, size=size, nb_classes=nc, device="cpu")


class TestLibreFOMOTrainerSmoke:
    """Verify trainer setup, checkpoint fallback, resume mechanics, and dataset configurations."""

    def test_training_checkpoint_reload_fallback(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Verify that train() correctly falls back to last_checkpoint if best_checkpoint is missing/empty."""
        dummy_ckpt = tmp_path / "last.pt"
        model = _make_random_fomo(size="s", nc=1)
        from libreyolo.utils.serialization import wrap_libreyolo_checkpoint
        ckpt_dict = wrap_libreyolo_checkpoint(
            model.model.state_dict(),
            model_family="fomo",
            size="s",
            task="point",
            nc=1,
        )
        torch.save(ckpt_dict, dummy_ckpt)
        
        class DummyTrainer:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                pass
            def train(self) -> dict[str, Any]:
                return {
                    "epoch_losses": [0.5],
                    "final_loss": 0.5,
                    "best_checkpoint": "",
                    "last_checkpoint": str(dummy_ckpt),
                }
                
        monkeypatch.setattr("libreyolo.models.fomo.trainer.FOMOTrainer", DummyTrainer)
        
        loaded_path = None
        def mock_load_weights(path: str | Path) -> None:
            nonlocal loaded_path
            loaded_path = path
            
        monkeypatch.setattr(model, "_load_weights", mock_load_weights)
        
        model.train("dummy.yaml")
        assert loaded_path == str(dummy_ckpt)

    def test_train_resume_raises_when_no_checkpoint(self) -> None:
        model = _make_random_fomo(size="s", nc=1)
        with pytest.raises(ValueError, match="resume=True requires a checkpoint"):
            model.train("dummy.yaml", resume=True)

    def test_train_resume_calls_trainer_resume(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        model = _make_random_fomo(size="s", nc=1)
        dummy_ckpt = tmp_path / "last.pt"
        torch.save({"epoch": 0, "config": {"epochs": 3}}, dummy_ckpt)
        model.model_path = dummy_ckpt

        setup_called = False
        resume_called_with = None

        class DummyTrainer:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                pass
            def setup(self) -> None:
                nonlocal setup_called
                setup_called = True
            def resume(self, path: str) -> None:
                nonlocal resume_called_with
                resume_called_with = path
            def train(self) -> dict[str, Any]:
                return {"epoch_losses": [0.5], "final_loss": 0.5}

        monkeypatch.setattr("libreyolo.models.fomo.trainer.FOMOTrainer", DummyTrainer)

        model.train("dummy.yaml", resume=True)
        assert setup_called is True
        assert resume_called_with == str(dummy_ckpt)

    def test_train_seeds_rngs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify that model.train seeds random, numpy, and torch before trainer construction."""
        model = _make_random_fomo(size="s", nc=1)
        
        seeds_recorded = []
        
        class DummyTrainer:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                import random
                import numpy as np
                import torch
                seeds_recorded.append((
                    random.randint(0, 1000000),
                    np.random.randint(0, 1000000),
                    torch.randint(0, 1000000, (1,)).item()
                ))

            def train(self) -> dict[str, Any]:
                return {"epoch_losses": [0.5], "final_loss": 0.5}

        monkeypatch.setattr("libreyolo.models.fomo.trainer.FOMOTrainer", DummyTrainer)

        model.train("dummy.yaml", seed=0)
        state_0_first = seeds_recorded[0]

        seeds_recorded.clear()
        model.train("dummy.yaml", seed=0)
        state_0_second = seeds_recorded[0]

        seeds_recorded.clear()
        model.train("dummy.yaml", seed=100)
        state_100 = seeds_recorded[0]

        assert state_0_first == state_0_second
        assert state_0_first != state_100

    def test_trainer_build_yolo_datasets_omitted_label_files(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Verify that _build_yolo_datasets correctly resolves label files when image files are pre-resolved but label files are None/omitted."""
        from libreyolo.models.fomo.trainer import FOMOTrainer
        
        # Mock load_data_config
        def mock_load_data_config(*args, **kwargs):
            return {
                "root": str(tmp_path),
                "train_img_files": [tmp_path / "images" / "train" / "img1.jpg"],
                "train_label_files": None,
                "val_img_files": [tmp_path / "images" / "val" / "img2.jpg"],
                "val_label_files": None,
                "nc": 1,
                "names": {0: "object"},
            }
        
        img1 = tmp_path / "images" / "train" / "img1.jpg"
        img2 = tmp_path / "images" / "val" / "img2.jpg"
        img1.parent.mkdir(parents=True, exist_ok=True)
        img2.parent.mkdir(parents=True, exist_ok=True)
        img1.touch()
        img2.touch()
        
        lbl1 = tmp_path / "labels" / "train" / "img1.txt"
        lbl2 = tmp_path / "labels" / "val" / "img2.txt"
        lbl1.parent.mkdir(parents=True, exist_ok=True)
        lbl2.parent.mkdir(parents=True, exist_ok=True)
        lbl1.touch()
        lbl2.touch()
        
        import libreyolo.data as data_mod
        monkeypatch.setattr(data_mod, "load_data_config", mock_load_data_config)
        
        model = _make_random_fomo(size="s", nc=1)
        trainer = FOMOTrainer(
            model=model.model,
            wrapper_model=model,
            data="dummy.yaml",
            epochs=1,
            batch=2,
            imgsz=96,
            device="cpu",
            project=str(tmp_path / "runs"),
            workers=0,
        )
        
        from libreyolo.models.fomo.dataset import build_fomo_datasets
        train_ds, val_ds, _, _ = build_fomo_datasets(
            config=trainer.config,
            model=trainer.model,
            wrapper_model=trainer.wrapper_model,
            device=trainer.device,
            input_size=96,
            grid_size=12,
        )
        assert train_ds.label_files is not None
        assert len(train_ds.label_files) == 1
        assert val_ds.label_files is not None
        assert len(val_ds.label_files) == 1

    def test_trainer_class_count_mismatch_fails_fast(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Verify that _build_yolo_datasets raises a ValueError if dataset nc != model nc and wrapper_model is None."""
        from libreyolo.models.fomo.trainer import FOMOTrainer
        
        # Mock load_data_config
        def mock_load_data_config(*args, **kwargs):
            return {
                "root": str(tmp_path),
                "train_img_files": [tmp_path / "images" / "train" / "img1.jpg"],
                "train_label_files": None,
                "val_img_files": [tmp_path / "images" / "val" / "img2.jpg"],
                "val_label_files": None,
                "nc": 2,  # Mismatch with model nc=1
                "names": {0: "class1", 1: "class2"},
            }
            
        img1 = tmp_path / "images" / "train" / "img1.jpg"
        img2 = tmp_path / "images" / "val" / "img2.jpg"
        img1.parent.mkdir(parents=True, exist_ok=True)
        img2.parent.mkdir(parents=True, exist_ok=True)
        img1.touch()
        img2.touch()
        
        lbl1 = tmp_path / "labels" / "train" / "img1.txt"
        lbl2 = tmp_path / "labels" / "val" / "img2.txt"
        lbl1.parent.mkdir(parents=True, exist_ok=True)
        lbl2.parent.mkdir(parents=True, exist_ok=True)
        lbl1.touch()
        lbl2.touch()
        
        import libreyolo.data as data_mod
        monkeypatch.setattr(data_mod, "load_data_config", mock_load_data_config)
        
        model = _make_random_fomo(size="s", nc=1)
        trainer = FOMOTrainer(
            model=model.model,
            wrapper_model=None,  # None to trigger the error path
            data="dummy.yaml",
            epochs=1,
            batch=2,
            imgsz=96,
            device="cpu",
            project=str(tmp_path / "runs"),
            workers=0,
        )
        
        from libreyolo.models.fomo.dataset import build_fomo_datasets
        with pytest.raises(ValueError, match="wrapper_model is None — cannot rebuild head safely"):
            build_fomo_datasets(
                config=trainer.config,
                model=trainer.model,
                wrapper_model=trainer.wrapper_model,
                device=trainer.device,
                input_size=96,
                grid_size=12,
            )

    def test_trainer_derives_nc_from_names_when_nc_absent(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Verify that FOMOTrainer derives dataset_nc from names when nc is absent in the data YAML."""
        from libreyolo.models.fomo.trainer import FOMOTrainer

        # Mock load_data_config
        def mock_load_data_config(*args, **kwargs):
            return {
                "root": str(tmp_path),
                "train_img_files": [tmp_path / "images" / "train" / "img1.jpg"],
                "train_label_files": None,
                "val_img_files": [tmp_path / "images" / "val" / "img2.jpg"],
                "val_label_files": None,
                # "nc" is intentionally absent
                "names": {0: "class1", 1: "class2", 2: "class3"},
            }

        img1 = tmp_path / "images" / "train" / "img1.jpg"
        img2 = tmp_path / "images" / "val" / "img2.jpg"
        img1.parent.mkdir(parents=True, exist_ok=True)
        img2.parent.mkdir(parents=True, exist_ok=True)
        img1.touch()
        img2.touch()

        lbl1 = tmp_path / "labels" / "train" / "img1.txt"
        lbl2 = tmp_path / "labels" / "val" / "img2.txt"
        lbl1.parent.mkdir(parents=True, exist_ok=True)
        lbl2.parent.mkdir(parents=True, exist_ok=True)
        lbl1.touch()
        lbl2.touch()

        import libreyolo.data as data_mod
        monkeypatch.setattr(data_mod, "load_data_config", mock_load_data_config)

        model = _make_random_fomo(size="s", nc=1)
        trainer = FOMOTrainer(
            model=model.model,
            wrapper_model=model,
            data="dummy.yaml",
            epochs=1,
            batch=2,
            imgsz=96,
            device="cpu",
            project=str(tmp_path / "runs"),
            workers=0,
        )

        from libreyolo.models.fomo.dataset import build_fomo_datasets
        train_ds, val_ds, _, _ = build_fomo_datasets(
            config=trainer.config,
            model=trainer.model,
            wrapper_model=trainer.wrapper_model,
            device=trainer.device,
            input_size=96,
            grid_size=12,
        )
        assert trainer.config.num_classes == 3

    def test_val_dispatch_uses_point_validator(self) -> None:
        """BaseModel.val() must route task='point' to PointValidator."""
        from libreyolo.models.base.model import BaseModel

        model = _make_random_fomo(size="s", nc=1)
        assert model.task == "point"

        # Augmented validation must be blocked for point models
        with pytest.raises(ValueError, match="Augmented validation"):
            BaseModel.val(model, data="unused.yaml", imgsz=96, augment=True)

    def test_train_with_invalid_imgsz_raises(self) -> None:
        model = _make_random_fomo(size="s")
        with pytest.raises(ValueError, match="only supports imgsz=96"):
            model.train("dummy.yaml", imgsz=128)
