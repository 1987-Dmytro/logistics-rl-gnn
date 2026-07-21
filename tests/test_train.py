"""Smoke-тесты обучения Phase 6 (НЕ полный прогон). train-cost падает, детерминизм по seed,
энтропия не коллапсирует, no NaN. Требует снапшот + torch/torch-geometric/scipy.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
pytest.importorskip("scipy")

from logistics_rl_gnn.config import instance as im  # noqa: E402
from logistics_rl_gnn.config.train import TrainConfig  # noqa: E402
from logistics_rl_gnn.models.policy import VRPPolicy  # noqa: E402
from logistics_rl_gnn.train.reinforce import Trainer  # noqa: E402

_NEED_SNAP = pytest.mark.skipif(im._latest_snapshot_dir() is None, reason="нет снапшота")


def _smoke_cfg(**kw):
    base = dict(
        epochs=2,
        steps_per_epoch=15,
        batch=6,
        n_range=(12, 16),
        val_range=(1_000_000, 1_000_004),
        ckpt=None,
        seed=0,
    )
    return TrainConfig(**{**base, **kw})


@_NEED_SNAP
def test_smoke_learns_entropy_alive_no_nan():
    torch.manual_seed(0)
    trainer = Trainer(VRPPolicy(), _smoke_cfg())
    before = float(
        np.mean([-trainer.policy.rollout(e, "greedy")[2]["reward"] for e in trainer.val_envs])
    )
    hist = trainer.fit()
    after = hist[-1]["val_cost"]
    assert after < 0.6 * before, (
        f"train-cost не упал: {before:.0f} → {after:.0f}"
    )  # обучение работает
    ents = [h["entropy"] for h in hist]
    # энтропия жива (не детерминизм-коллапс) и не застряла в uniform (< log k, k≤~17)
    assert all(np.isfinite(e) and 0.01 < e < math.log(20) for e in ents)
    for h in hist:  # без NaN в логах
        assert np.isfinite([h["train_cost"], h["val_cost"], h["grad_norm"], h["entropy"]]).all()


@_NEED_SNAP
def test_determinism_by_seed():
    def run():
        torch.manual_seed(0)
        return Trainer(VRPPolicy(), _smoke_cfg(steps_per_epoch=6, batch=4)).fit()

    h1, h2 = run(), run()
    assert [round(r["train_cost"], 4) for r in h1] == [round(r["train_cost"], 4) for r in h2]
    assert [round(r["val_cost"], 4) for r in h1] == [round(r["val_cost"], 4) for r in h2]
