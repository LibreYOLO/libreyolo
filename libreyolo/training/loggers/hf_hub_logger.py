"""Push the best checkpoint of a training run to the Hugging Face Hub."""

from __future__ import annotations

import logging
from pathlib import Path

from ..callbacks import TrainEndEvent
from .base import BaseLogger

logger = logging.getLogger("libreyolo")


class HuggingFaceHubLogger(BaseLogger):
    """Upload ``weights/best.pt`` to a Hub repository when training ends.

    Enable it with a configured instance or the ``"hf:owner/repo"`` string
    form::

        model.train(data="data.yaml", loggers="hf:someuser/my-finetune")

    Authentication is resolved the standard huggingface_hub way (``hf auth
    login`` or ``HF_TOKEN``). Write access is verified at construction time,
    which also creates the target repository, so a credential problem surfaces
    before the run instead of discarding hours of training at the end.

    Args:
        repo_id: Target repository as ``"owner/name"``. Created immediately
            when the logger is constructed.
        private: Create the repo as private. Defaults to True, unlike the
            explicit :meth:`~libreyolo.models.base.BaseModel.push_to_hub`:
            this upload happens unattended at the end of a run, so a model
            trained on proprietary data must never become public by surprise.
            Pass ``private=False`` to publish. Existing repos keep whatever
            visibility they already have.
        token: Hub token overriding the ambient login (must have write scope).
        license: License identifier for the generated model card.
    """

    def __init__(
        self,
        repo_id: str,
        *,
        private: bool = True,
        token: str | None = None,
        license: str | None = None,
    ):
        super().__init__()
        from libreyolo.utils.hf_hub import assert_can_push

        # Proves write access now rather than after the run, which also
        # creates the target repo up front.
        assert_can_push(repo_id, private=private, token=token)
        self.repo_id = repo_id
        self.private = private
        self.token = token
        self.license = license

    def _handle_end(self, event: TrainEndEvent) -> None:
        from libreyolo.utils.hf_hub import push_checkpoint_to_hub

        weights_dir = Path(event.save_dir) / "weights"
        checkpoint = None
        for candidate in ("best.pt", "last.pt"):
            if (weights_dir / candidate).is_file():
                checkpoint = weights_dir / candidate
                break
        if checkpoint is None:
            logger.warning(
                "HuggingFaceHubLogger: no best.pt or last.pt under %s; "
                "nothing to push.",
                weights_dir,
            )
            return

        metrics = {
            key: value
            for key, value in event.results.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        url = push_checkpoint_to_hub(
            checkpoint,
            self.repo_id,
            private=self.private,
            token=self.token,
            license_id=self.license,
            metrics=metrics or None,
            commit_message=(
                f"Upload {event.model_family}{event.model_size or ''} "
                f"{event.task} training result ({event.completed_epochs} epochs)"
            ),
        )
        logger.info("Training checkpoint pushed to %s", url)
