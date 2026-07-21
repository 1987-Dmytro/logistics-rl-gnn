"""Smoke-тесты POMO (Phase 6b Шаг 1, НЕ полный прогон). Стражи против коллапса и вырождения
shared baseline: разброс стартов >0 (advantage жив), |g|>0, cost↓, детерминизм, энтропия не
коллапс, no-NaN. Требует снапшот + torch/torch-geometric.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from logistics_rl_gnn.config import instance as im  # noqa: E402
from logistics_rl_gnn.config.pomo import POMOConfig  # noqa: E402
from logistics_rl_gnn.models.policy import VRPPolicy  # noqa: E402
from logistics_rl_gnn.train.pomo import POMOTrainer, multistart_greedy  # noqa: E402

_NEED_SNAP = pytest.mark.skipif(im._latest_snapshot_dir() is None, reason="нет снапшота")


def _smoke_cfg(**kw):
    base = dict(
        epochs=4,
        steps_per_epoch=6,
        batch=3,
        max_starts=4,
        n_range=(12, 16),
        val_range=(1_000_000, 1_000_003),
        ckpt=None,
        seed=0,
    )
    return POMOConfig(**{**base, **kw})


@_NEED_SNAP
def test_shared_baseline_advantage_alive_and_grad():
    torch.manual_seed(0)
    tr = POMOTrainer(VRPPolicy(), _smoke_cfg())
    rng = np.random.default_rng(0)
    rec = tr.train_batch(rng.integers(0, 100_000, size=3))
    # shared baseline НЕ вырожден: старты одного инстанса дают РАЗНЫЙ cost (raw std>0) → adv жив
    assert any(np.std(v) > 0 for v in rec["cost_vecs"]), "старты равны (baseline вырожден)"
    assert rec["start_std"] > 0.0
    assert rec["grad_norm"] > 1e-6, "нулевой градиент (насыщение)"
    assert np.isfinite(rec["grad_norm"]) and np.isfinite(rec["cost"])  # no-NaN на реальном инстансе


@_NEED_SNAP
def test_cost_drops_and_no_collapse():
    torch.manual_seed(0)
    tr = POMOTrainer(VRPPolicy(), _smoke_cfg())
    hist = tr.fit()
    # cost↓: обученные эпохи бьют untrained epoch 0 (train sample-cost — чистый learning-сигнал;
    # val на малых инстансах насыщен ≈эвристикой). Порог мягкий против RL-шума.
    assert hist[-1]["train_cost"] < hist[0]["train_cost"], (
        f"train_cost не упал: {hist[0]['train_cost']:.0f} → {hist[-1]['train_cost']:.0f}"
    )
    assert all(h["grad_norm"] > 1e-6 for h in hist), "grad занулился (коллапс насыщения)"
    assert all(h["start_std"] > 0 for h in hist), "разброс стартов исчез (shared baseline вырожден)"
    assert all(0.05 < h["entropy"] < math.log(60) for h in hist), "энтропия коллапс/uniform"
    assert hist[-1]["val_cost"] < 1.3 * tr.val_heur, "политика разошлась (val ≫ эвристики)"
    for h in hist:  # без NaN в логах
        assert np.isfinite([h["train_cost"], h["val_cost"], h["grad_norm"], h["entropy"]]).all()


@_NEED_SNAP
def test_multistart_greedy_deterministic_and_feasible():
    torch.manual_seed(0)
    tr = POMOTrainer(VRPPolicy(), _smoke_cfg())
    env = tr.val_envs[0]
    c1, r1 = multistart_greedy(tr.policy, env, tr.cfg.max_starts)
    c2, r2 = multistart_greedy(tr.policy, env, tr.cfg.max_starts)
    assert c1 == pytest.approx(c2) and r1 == r2  # greedy-инференс детерминирован
    assert np.isfinite(c1)
    for rt in r1:  # маршруты старт/финиш в депо
        assert rt[0] == 0 and rt[-1] == 0


@_NEED_SNAP
def test_determinism_by_seed():
    def run():
        torch.manual_seed(0)
        return POMOTrainer(VRPPolicy(), _smoke_cfg(epochs=2, steps_per_epoch=4)).fit()

    h1, h2 = run(), run()
    assert [round(r["train_cost"], 4) for r in h1] == [round(r["train_cost"], 4) for r in h2]
    assert [round(r["val_cost"], 4) for r in h1] == [round(r["val_cost"], 4) for r in h2]
