"""GTR detection and pose wrapper with strict checkpoint loading."""

from pathlib import Path
from typing import ClassVar

from libreyolo.training.ddp_spawn import ddp_aware

from ...tasks import normalize_task, task_to_suffix
from ...training.callbacks import TrainCallbacks
from ...validation.preprocessors import DEIMv2DINOValPreprocessor
from ..dfine.model import LibreDFINE
from ..ec.postprocess import preprocess_image
from .config import GTRConfig
from .nn import LibreGTRModel
from .pose import LibreGTRPoseModel

_POSE_HEAD_KEY = "decoder.keypoint_embedding.weight"


class LibreGTR(LibreDFINE):
    """GTR detection and pose (S/M/L/X). GPU and convergence evidence is pending."""

    FAMILY = "gtr"
    FILENAME_PREFIX = "LibreGTR"
    INPUT_SIZES: ClassVar[dict[str, int]] = {s: 640 for s in ("s", "m", "l", "x")}
    SUPPORTED_TASKS = ("detect", "pose")
    TASK_INPUT_SIZES: ClassVar[dict] = {}
    DEFAULT_TASK = "detect"
    TRAIN_CONFIG = GTRConfig
    SUPPORTS_CUDA_GRAPH = False
    val_preprocessor_class = DEIMv2DINOValPreprocessor
    POSE_NUM_KEYPOINTS = 17
    KEYPOINT_DIM = 3

    @classmethod
    def can_load(cls, sd):
        if (
            "backbone.backbone._model.blocks.0.attn.gk_proj.0.weight" in sd
            and "encoder.stages.0.1.weight" in sd
            and _POSE_HEAD_KEY in sd
        ):
            return True
        return (
            "backbone.backbone._model.blocks.0.attn.gk_proj.0.weight" in sd
            and "decoder.dec_score_head.0.weight" in sd
            and "encoder.stages.0.1.weight" in sd
            and "decoder.pre_bbox_head.layers.2.weight" in sd
            and sd["decoder.pre_bbox_head.layers.2.weight"].shape[0] == 4
            and not any("segmentation_head" in k for k in sd)
        )

    @staticmethod
    def _weight(sd, key):
        # LoRA checkpoints keep adapted Linear weights under ``.base_layer.``.
        value = sd.get(f"{key}.weight")
        return value if value is not None else sd.get(f"{key}.base_layer.weight")

    @classmethod
    def detect_size(cls, sd):
        weight = cls._weight(sd, "backbone.backbone._model.blocks.0.attn.q_proj")
        if weight is None:
            return None
        width = weight.shape[1]
        if width in (192, 256):
            return {192: "s", 256: "m"}[width]
        if width == 384:
            weight = cls._weight(sd, "decoder.decoder.layers.0.linear1")
            return (
                {1024: "l", 2048: "x"}.get(weight.shape[0])
                if weight is not None
                else None
            )
        return None

    @classmethod
    def detect_checkpoint_task(cls, sd):
        if _POSE_HEAD_KEY in sd:
            return "pose"
        return super().detect_checkpoint_task(sd)

    @classmethod
    def detect_nb_classes(cls, sd):
        # Pose is person-only; the head's two logits are not user classes.
        if _POSE_HEAD_KEY in sd:
            return 1
        return super().detect_nb_classes(sd)

    @classmethod
    def detect_size_from_filename(cls, filename):
        from ..base import BaseModel

        return BaseModel.detect_size_from_filename.__func__(cls, filename)

    HF_REVISIONS: ClassVar[dict[str, str]] = {
        "s": "74193dc356e07f51893579211ffdecf0ee2e560a",
        "m": "9b1c76a16dc09bbd0504a01aeadff03d2ecf642d",
        "l": "b828ad6d42dd8dd287244086c2543fb45e60f94d",
        "x": "b029aa3335222ffaba58f8533cc2cd007b79bc3e",
    }
    # Non-detect weight repos, keyed by (size, task). None is a placeholder
    # that downloads from ``main`` until the repo's upload commit is pinned.
    HF_TASK_REVISIONS: ClassVar[dict[tuple[str, str], str | None]] = {
        # PLACEHOLDER: pin the LibreYOLO/LibreGTR{size}-pose commits after upload.
        ("s", "pose"): None,
        ("m", "pose"): None,
        ("l", "pose"): None,
        ("x", "pose"): None,
    }

    @classmethod
    def get_download_url(cls, filename):
        size = cls.detect_size_from_filename(filename)
        if size is None:
            return None
        task = cls.detect_task_from_filename(filename) or "detect"
        name = f"LibreGTR{size}"
        if task == "detect":
            revision = cls.HF_REVISIONS[size]
        else:
            if (size, task) not in cls.HF_TASK_REVISIONS:
                return None
            name = f"{name}-{task_to_suffix(task)}"
            revision = cls.HF_TASK_REVISIONS[(size, task)] or "main"
        if Path(filename).stem != name:
            return None
        return f"https://huggingface.co/LibreYOLO/{name}/resolve/{revision}/{name}.pt"

    def __init__(
        self, model_path, size, nb_classes=80, device="auto", task=None, **kwargs
    ):
        if task is not None and normalize_task(task) == "pose":
            nb_classes = 1
        self.num_keypoints = self.POSE_NUM_KEYPOINTS
        self.keypoint_dim = self.KEYPOINT_DIM
        super().__init__(
            model_path, size, nb_classes=nb_classes, device=device, task=task, **kwargs
        )

    def _init_model(self):
        if self.task == "pose":
            if self.names == {0: "class_0"}:
                self.names = {0: "person"}
            return LibreGTRPoseModel(
                self.size, eval_spatial_size=(self.input_size, self.input_size)
            )
        return LibreGTRModel(self.size, self.nb_classes)

    def _validate_loaded_state_dict_for_task(self, state_dict, checkpoint=None):
        is_pose = self.detect_checkpoint_task(state_dict) == "pose"
        if is_pose != (self.task == "pose"):
            raise RuntimeError(
                f"Checkpoint is a GTR-{'pose' if is_pose else 'detect'} model but "
                f"this instance was initialized for task='{self.task}'. Pass the "
                "matching task or use a -pose filename suffix."
            )
        if not is_pose:
            super()._validate_loaded_state_dict_for_task(state_dict, checkpoint)

    def _rebuild_for_new_classes(self, new_nb_classes):
        if self.task == "pose":
            if new_nb_classes != 1:
                raise ValueError("GTR pose is single-class (person)")
            return
        super()._rebuild_for_new_classes(new_nb_classes)

    def _postprocess(
        self, output, conf_thres, iou_thres, original_size, max_det=300, **kwargs
    ):
        if self.task == "pose":
            from ...postprocess.ec import postprocess_pose

            return postprocess_pose(
                output,
                conf_thres=conf_thres,
                iou_thres=iou_thres,
                original_size=original_size,
                max_det=max_det,
                num_keypoints=self.POSE_NUM_KEYPOINTS,
            )
        return super()._postprocess(
            output, conf_thres, iou_thres, original_size, max_det=max_det, **kwargs
        )

    @staticmethod
    def _apply_lora(model):
        from ...training.lora import apply_lora_to_gtr

        apply_lora_to_gtr(model)

    def _strict_loading(self):
        return True

    def _load_weights(self, model_path):
        super()._load_weights(model_path)
        # Loading a custom-class checkpoint may rebuild the module after the
        # base constructor has switched the original module to evaluation.
        self.model.eval()

    def _get_available_layers(self):
        return {
            "backbone": self.model.backbone,
            "encoder": self.model.encoder,
            "decoder": self.model.decoder,
        }

    @staticmethod
    def _get_preprocess_numpy():
        from ...preprocess.ec import preprocess_numpy

        return preprocess_numpy

    @classmethod
    def _validate_imgsz(cls, imgsz, **kwargs):
        if not isinstance(imgsz, int) or imgsz < 160 or imgsz % 32:
            raise ValueError("GTR imgsz must be a multiple of 32 and at least 160")
        return imgsz

    def _preprocess(self, image, color_format="auto", input_size=None):
        size = self._validate_imgsz(input_size or self.input_size)
        return preprocess_image(image, input_size=size, color_format=color_format)

    @ddp_aware()
    def train(
        self,
        data: str | None = None,
        *,
        epochs: int | None = None,
        batch: int | None = None,
        imgsz: int | None = None,
        lr0: float | None = None,
        device: str = "",
        workers: int | None = None,
        seed: int | None = None,
        project: str | None = None,
        name: str | None = None,
        exist_ok: bool | None = None,
        resume: bool | str = False,
        amp: bool | None = None,
        patience: int | None = None,
        callbacks: TrainCallbacks = None,
        loggers=None,
        **kwargs,
    ) -> dict:
        """Fine-tune GTR; resume restores saved settings before explicit overrides."""
        from dataclasses import fields

        from libreyolo.data import load_data_config

        if self.task == "pose":
            explicit = {
                "epochs": epochs,
                "batch": batch,
                "imgsz": imgsz,
                "lr0": lr0,
                "workers": workers,
                "seed": seed,
                "project": project,
                "name": name,
                "exist_ok": exist_ok,
                "amp": amp,
                "patience": patience,
            }
            kwargs.update({k: v for k, v in explicit.items() if v is not None})
            return self._train_pose(
                data,
                device=device,
                resume=resume,
                callbacks=callbacks,
                loggers=loggers,
                **kwargs,
            )

        from .trainer import GTRTrainer

        kwargs.pop("pretrained", None)
        resume_path = None
        settings = {}
        if resume:
            resume_path = (
                str(resume) if isinstance(resume, (str, Path)) else self.model_path
            )
            if not resume_path:
                raise ValueError("resume=True requires a loaded training checkpoint")
            settings = self._checkpoint_train_config(resume_path)
            if not settings:
                raise ValueError(
                    "GTR resume requires a checkpoint with saved training configuration"
                )
            self._load_weights(str(resume_path))

        valid = {field.name for field in fields(GTRConfig)}
        settings = {
            key: value
            for key, value in settings.items()
            if key in valid and key not in {"size", "num_classes", "resume"}
        }
        explicit = {
            "data": data,
            "epochs": epochs,
            "batch": batch,
            "imgsz": imgsz,
            "lr0": lr0,
            "workers": workers,
            "seed": seed,
            "project": project,
            "name": name,
            "exist_ok": exist_ok,
            "amp": amp,
            "patience": patience,
        }
        settings.update(
            {key: value for key, value in explicit.items() if value is not None}
        )
        settings.update(kwargs)
        if device:
            settings["device"] = device
        settings.setdefault("device", "auto")
        settings.setdefault("seed", 0)
        if not settings.get("data"):
            raise ValueError(
                "GTR training requires data or a resume checkpoint containing data"
            )
        if settings.get("imgsz") is not None:
            settings["imgsz"] = self._validate_imgsz(settings["imgsz"])

        # Seed before replacing the classification heads for a custom dataset.
        self._seed_scratch_initialization(settings["seed"])
        data_config = load_data_config(
            settings["data"],
            autodownload=True,
            single_cls=bool(settings.get("single_cls", False)),
        )
        settings["data"] = data_config.get("yaml_file", settings["data"])
        names = data_config.get("names")
        nc = data_config.get("nc", len(names) if names is not None else self.nb_classes)
        if nc != self.nb_classes:
            if resume_path:
                raise ValueError(
                    "Cannot resume GTR with a different dataset class count"
                )
            self._rebuild_for_new_classes(nc)
        if isinstance(names, list):
            names = dict(enumerate(names))
        if names is not None:
            self.names = self._sanitize_names(names, self.nb_classes)

        trainer = GTRTrainer(
            model=self.model,
            wrapper_model=self,
            size=self.size,
            num_classes=self.nb_classes,
            resume=bool(resume_path),
            callbacks=callbacks,
            loggers=loggers,
            **settings,
        )
        if resume_path:
            trainer.setup()
            trainer.resume(str(resume_path))
        results = trainer.train()
        best = results.get("best_checkpoint")
        if best and Path(best).exists():
            self.model_path = best
            self._load_weights(best)
        self.model.to(self.device)
        return results

    def _train_pose(
        self, data, *, device="", resume=False, callbacks=None, loggers=None, **kwargs
    ):
        """Fine-tune GTR pose on a YOLO keypoint dataset (17 COCO keypoints)."""
        from libreyolo.data import load_data_config

        from .pose_trainer import GTRPoseTrainer

        if kwargs.get("lora"):
            raise ValueError("GTR pose training does not support lora=True yet")
        kwargs.pop("pretrained", None)
        imgsz = int(kwargs.setdefault("imgsz", self.input_size))
        if imgsz != int(self.input_size):
            raise ValueError(
                f"GTR pose fine-tuning requires imgsz={self.input_size}; the "
                "decoder anchor grid is built for the native input size."
            )
        resume_path = None
        if resume:
            resume_path = (
                str(resume) if isinstance(resume, (str, Path)) else self.model_path
            )
            if not resume_path:
                raise ValueError("resume=True requires a loaded training checkpoint")
            saved = self._checkpoint_train_config(resume_path) or {}
            data = data or saved.get("data")
        if not data:
            raise ValueError("GTR pose training requires data")

        data_config = load_data_config(data, autodownload=True)
        data = data_config.get("yaml_file", data)
        kpt_shape = data_config.get("kpt_shape")
        if not kpt_shape or int(kpt_shape[0]) != self.POSE_NUM_KEYPOINTS:
            raise ValueError(
                "GTR pose fine-tuning needs 'kpt_shape: [17, 2|3]' in the dataset "
                "yaml; the keypoint head is fixed at 17 COCO keypoints."
            )
        keypoint_dim = int(kpt_shape[1]) if len(kpt_shape) > 1 else 3
        names = data_config.get("names")
        if names is not None and len(names) != 1:
            raise ValueError("GTR pose fine-tuning supports single-class datasets only")
        if isinstance(names, list):
            names = dict(enumerate(names))
        if names:
            self.names = self._sanitize_names(names, 1)

        trainer = GTRPoseTrainer(
            model=self.model,
            wrapper_model=self,
            size=self.size,
            num_classes=1,
            num_keypoints=self.POSE_NUM_KEYPOINTS,
            keypoint_dim=keypoint_dim,
            flip_idx=kwargs.pop("flip_idx", data_config.get("flip_idx")),
            data=data,
            device=device or "auto",
            resume=bool(resume_path),
            callbacks=callbacks,
            loggers=loggers,
            **kwargs,
        )
        if resume_path:
            trainer.setup()
            trainer.resume(resume_path)
        results = trainer.train()
        best = results.get("best_checkpoint")
        if best and Path(best).exists():
            self.model_path = best
            self._load_weights(best)
        self.model.to(self.device)
        return results

    def export(self, format="onnx", **kwargs):
        """Export a fixed-resolution FP32 graph using portable attention."""
        if format == "pt" and getattr(self, "_quant_manifest", None):
            return super().export(format=format, **kwargs)
        if format not in ("onnx", "torchscript"):
            raise NotImplementedError(
                "GTR currently exports to ONNX and TorchScript only"
            )
        if kwargs.get("dynamic", False):
            raise ValueError("GTR export requires dynamic=False")
        if kwargs.get("half", False):
            raise ValueError("GTR portable export currently requires FP32")
        kwargs["dynamic"] = False
        return super().export(format=format, **kwargs)
