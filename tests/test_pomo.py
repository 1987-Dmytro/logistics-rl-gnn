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


@_NEED_SNAP
def test_instance_freshness():
    # свежесть: 3 подряд train_batch → РАЗНЫЕ node-id наборы (RNG продвигается, не memorize)
    torch.manual_seed(0)
    tr = POMOTrainer(VRPPolicy(), _smoke_cfg())
    rng = np.random.default_rng(0)
    hashes = [tr.train_batch(rng.integers(0, 100_000, size=3))["inst_hash"] for _ in range(3)]
    assert len(set(hashes)) == 3, f"инстансы не обновляются между шагами: {hashes}"


@_NEED_SNAP
def test_early_stop_fires(monkeypatch):
    # early-stop: val только растёт после epoch0 → patience=2 обрывает на epoch2 (history=3)
    torch.manual_seed(0)
    tr = POMOTrainer(VRPPolicy(), _smoke_cfg(epochs=20, patience=2))
    seq = iter([100.0, 101.0, 102.0, 103.0, 104.0])  # монотонно вверх → нет улучшения
    monkeypatch.setattr(tr, "_validate", lambda: {"val_cost": next(seq), "gap_greedy": 0.0})
    hist = tr.fit()
    assert len(hist) == 3, f"early-stop не сработал на patience=2: {len(hist)} эпох (ждём 3)"
    assert hist[-1]["since_improve"] == 2


# ---------- Шаг 2: обучение под congestion ----------


def _cong_cfg(**kw):
    base = dict(
        warm_start=None, epochs=2, steps_per_epoch=4, batch=3, max_starts=4,
        n_range=(12, 16), val_range=(1_000_000, 1_000_003),
        test_range=(2_000_000, 2_000_003), patience=2, ckpt=None,
    )  # fmt: skip
    return POMOConfig.for_congestion(**{**base, **kw})


@_NEED_SNAP
def test_congestion_train_and_coverage():
    from logistics_rl_gnn.train.pomo import congestion_coverage

    torch.manual_seed(0)
    tr = POMOTrainer(VRPPolicy(), _cong_cfg())
    cov = congestion_coverage(tr.probe_envs)
    assert cov["inc_node_cov"] > 0.5, f"инциденты не задевают узлы (сигнал слаб): {cov}"
    hist = tr.fit()  # congestion-обучение крутится без NaN, shared baseline жив
    for h in hist:
        assert np.isfinite([h["train_cost"], h["val_cost"], h["grad_norm"], h["entropy"]]).all(), h
    assert all(h["start_std"] > 0 for h in hist), "shared baseline вырожден под congestion"


@_NEED_SNAP
def test_congestion_sampler_deterministic():
    from logistics_rl_gnn.train.instance_sampler import InstanceSampler
    from logistics_rl_gnn.train.pomo import sample_congestion_travel

    cfg = _cong_cfg()
    inst = InstanceSampler(n_range=(15, 20)).sample(3)
    a = sample_congestion_travel(inst, 3, cfg)
    b = sample_congestion_travel(inst, 3, cfg)
    assert a.offset_min == b.offset_min and len(a.incidents) == len(b.incidents)  # детерм. по seed
    c = sample_congestion_travel(inst, 9, cfg)
    assert a.offset_min != c.offset_min or len(a.incidents) != len(c.incidents)  # seed меняет


@_NEED_SNAP
def test_warm_start_floor_not_overwritten(tmp_path, monkeypatch):
    # floor: warm_start сохранён как ckpt; если эпохи не бьют стартовый val → ckpt = warm-start
    ckpt = tmp_path / "floor.pt"
    cfg = _cong_cfg(warm_start="dummy", ckpt=str(ckpt), epochs=3, patience=3)
    torch.manual_seed(0)
    tr = POMOTrainer(VRPPolicy(), cfg)
    floor = {k: v.clone() for k, v in tr.policy.state_dict().items()}  # веса ДО обучения
    vals = iter([100.0] + [200.0] * 10)  # floor=100; trained-эпохи хуже (200) → не перезапишут
    monkeypatch.setattr(tr, "_validate", lambda: {"val_cost": next(vals), "gap_greedy": 0.0})
    tr.fit()  # обучение меняет tr.policy, но ckpt остаётся floor (эпохи не били)
    saved = torch.load(ckpt, weights_only=True)
    assert all(torch.equal(saved[k], floor[k]) for k in floor), "floor перезаписан"
