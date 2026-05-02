"""LibreVocab1 Phase 2 — trainer smoke tests.

Wires up the full training loop with a *fake* decoder so we exercise every
piece of the harness (data loader, collator, criterion, optimizer, EMA,
gradient clipping) without needing the real decoder which is still pending
the DEIMv2 branch merge.

The fake decoder is a small parameterized module that maps image features
to (pred_logits, pred_boxes) of the right shapes. It has trainable params
so the optimizer has something to update; loss should decrease over a few
steps when overfitting one batch.
"""

from __future__ import annotations

import importlib.util

import pytest


_REQUIRED = ("scipy",)
_MISSING = [m for m in _REQUIRED if importlib.util.find_spec(m) is None]
pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(
        bool(_MISSING),
        reason=f"libreyolo[vocab1] extras missing: {_MISSING!r}",
    ),
]


class _FakeDecoder:
    """Stand-in for the real decoder. Callable with (images, text_emb)."""

    def __init__(self, num_queries: int = 16, hidden: int = 32, text_dim: int = 8):
        import torch
        import torch.nn as nn

        # A tiny CNN to extract a per-image feature, plus an MLP to project to
        # logits per query × prompt and a small head for boxes.
        class _Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.feat = nn.Sequential(
                    nn.Conv2d(3, 8, 3, stride=4, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Linear(8, hidden),
                )
                self.q_proj = nn.Linear(hidden, num_queries * text_dim)
                self.box_head = nn.Linear(hidden, num_queries * 4)
                self.num_queries = num_queries
                self.text_dim = text_dim

            def forward(self, images, text_emb):
                B = images.shape[0]
                z = self.feat(images)               # (B, hidden)
                q = self.q_proj(z).view(B, num_queries, text_dim)  # (B, Q, text_dim)
                # Cosine logits against text embeddings: (B, Q, K).
                t = text_emb / (text_emb.norm(dim=-1, keepdim=True) + 1e-6)
                qn = q / (q.norm(dim=-1, keepdim=True) + 1e-6)
                logits = qn @ t.T
                boxes = self.box_head(z).view(B, num_queries, 4).sigmoid()
                return {"pred_logits": logits, "pred_boxes": boxes}

        self.net = _Net()

    def __call__(self, *args, **kwargs):
        return self.net(*args, **kwargs)

    def parameters(self):
        return self.net.parameters()

    def named_parameters(self):
        return self.net.named_parameters()

    def train(self):
        self.net.train()
        return self

    def eval(self):
        self.net.eval()
        return self


def _build_synthetic_batch(K: int = 4, B: int = 2, img_size: int = 64):
    """Build one synthetic open-vocab batch matching the real collator schema."""
    import torch
    images = torch.rand(B, 3, img_size, img_size)
    text_emb = torch.randn(K, 8)
    targets = [
        {
            "labels": torch.tensor([0, 1]),
            "boxes_cxcywh": torch.tensor([[0.3, 0.3, 0.2, 0.2], [0.7, 0.6, 0.1, 0.1]]),
        },
        {
            "labels": torch.tensor([2]),
            "boxes_cxcywh": torch.tensor([[0.5, 0.5, 0.3, 0.3]]),
        },
    ]
    return {
        "images": images,
        "text_emb": text_emb,
        "targets": targets,
        "prompt_names": [f"cls{i}" for i in range(K)],
        "image_ids": list(range(B)),
        "image_size": (img_size, img_size),
    }


def test_trainer_step_runs_and_updates_params():
    import torch
    from libreyolo.models.librevocab1.phase2.matcher import OpenVocabHungarianMatcher
    from libreyolo.models.librevocab1.phase2.loss import OpenVocabSetCriterion
    from libreyolo.models.librevocab1.phase2.trainer import Phase2Trainer

    torch.manual_seed(0)
    fake = _FakeDecoder(num_queries=8, text_dim=8)
    criterion = OpenVocabSetCriterion(OpenVocabHungarianMatcher())
    batch = _build_synthetic_batch()

    # Snapshot initial param.
    p0 = next(fake.parameters()).detach().clone()

    trainer = Phase2Trainer(
        model=fake,
        criterion=criterion,
        train_loader=[batch],  # one-batch loader
        optimizer=torch.optim.Adam(fake.parameters(), lr=1e-2),
        device="cpu",
        clip_max_norm=0.1,
        amp=False,
        ema_decay=None,
    )
    losses = trainer.step(batch)
    assert "loss_total" in losses
    assert all(v == v for v in losses.values())  # no NaN
    p1 = next(fake.parameters()).detach()
    assert not torch.allclose(p0, p1), "param did not move after one step"


def test_trainer_overfit_single_batch_decreases_loss():
    import torch
    from libreyolo.models.librevocab1.phase2.matcher import OpenVocabHungarianMatcher
    from libreyolo.models.librevocab1.phase2.loss import OpenVocabSetCriterion
    from libreyolo.models.librevocab1.phase2.trainer import Phase2Trainer

    torch.manual_seed(0)
    fake = _FakeDecoder(num_queries=12, text_dim=8)
    criterion = OpenVocabSetCriterion(OpenVocabHungarianMatcher())
    batch = _build_synthetic_batch(K=4, B=2)
    loader = [batch] * 8  # 8 steps per "epoch", same batch

    trainer = Phase2Trainer(
        model=fake,
        criterion=criterion,
        train_loader=loader,
        optimizer=torch.optim.Adam(fake.parameters(), lr=5e-3),
        device="cpu",
        clip_max_norm=0.1,
        amp=False,
        ema_decay=0.99,
    )
    result = trainer.fit(epochs=8)
    assert result.epochs_completed == 8
    assert result.last_step == 64
    # Mean loss in last epoch noticeably below first epoch.
    first = result.epoch_losses[0]
    last = min(result.epoch_losses[-3:])
    assert last < first * 0.7, f"overfit failed: first={first:.4f}, last={last:.4f}"


def test_trainer_raises_on_nonfinite_loss():
    """Sanity: NaN/Inf loss raises a clear RuntimeError, not a silent step."""
    import torch
    from libreyolo.models.librevocab1.phase2.trainer import Phase2Trainer

    fake = _FakeDecoder(num_queries=4, text_dim=8)
    batch = _build_synthetic_batch(K=3, B=1)

    class _NaNCriterion(torch.nn.Module):
        def forward(self, *args, **kwargs):
            return {
                "loss_cls": torch.tensor(float("nan"), requires_grad=True),
                "loss_bbox": torch.tensor(0.0, requires_grad=True),
                "loss_giou": torch.tensor(0.0, requires_grad=True),
            }

    trainer = Phase2Trainer(
        model=fake,
        criterion=_NaNCriterion(),
        train_loader=[batch],
        optimizer=torch.optim.Adam(fake.parameters(), lr=1e-3),
        device="cpu",
        ema_decay=None,
    )
    with pytest.raises(RuntimeError, match="non-finite"):
        trainer.step(batch)


def test_trainer_ema_tracks_params():
    import torch
    from libreyolo.models.librevocab1.phase2.matcher import OpenVocabHungarianMatcher
    from libreyolo.models.librevocab1.phase2.loss import OpenVocabSetCriterion
    from libreyolo.models.librevocab1.phase2.trainer import Phase2Trainer

    fake = _FakeDecoder(num_queries=4, text_dim=8)
    criterion = OpenVocabSetCriterion(OpenVocabHungarianMatcher())
    batch = _build_synthetic_batch()

    trainer = Phase2Trainer(
        model=fake,
        criterion=criterion,
        train_loader=[batch],
        optimizer=torch.optim.Adam(fake.parameters(), lr=5e-2),
        device="cpu",
        ema_decay=0.5,  # very loose for visible drift in 1 step
    )
    p0 = next(fake.parameters()).detach().clone()
    trainer.step(batch)
    p1 = next(fake.parameters()).detach()
    # Find the ema buffer for the first parameter and check it's between p0 and p1.
    name0 = next(fake.named_parameters())[0]
    ema_buf = trainer.ema.shadow[name0]
    # decay=0.5 -> ema = 0.5*p0 + 0.5*p1
    expected = 0.5 * p0 + 0.5 * p1
    assert torch.allclose(ema_buf, expected, atol=1e-5), (
        "EMA shadow does not match decay update formula"
    )


def test_phase2_config_defaults_match_proposal():
    """Config field check — Phase 2's defaults are the DEIMv2-style recipe."""
    from libreyolo.training.config import LibreVocab1Phase2Config

    cfg = LibreVocab1Phase2Config()
    assert cfg.optimizer == "adamw"
    assert cfg.lr0 == 1e-4
    assert cfg.weight_decay == 1e-4
    assert cfg.scheduler == "flat_cosine"
    assert cfg.clip_max_norm == 0.1
    assert cfg.ema is True
    assert cfg.ema_decay == 0.9999
    # Open-vocab specifics.
    assert cfg.radio_version == "c-radio_v4-so400m"
    assert cfg.text_prompt_template == "a photo of a {}"
    assert cfg.num_negatives == 32
    # Loss weights match plan.
    assert cfg.loss_weight_class == 1.0
    assert cfg.loss_weight_bbox == 5.0
    assert cfg.loss_weight_giou == 2.0
