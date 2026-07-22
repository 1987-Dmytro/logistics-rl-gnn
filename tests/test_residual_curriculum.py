"""Path B residual-curriculum — стражи предрегистрации 0011.

Проверяем: (1) seed-дизъюнкция held-out (train/val-residual ∩ {0–9} = ∅, реализации различны);
(2) residual не вырожден (≥3 pending, ≥2 старта, тег real+синтетика); (3) детерминизм по сиду
(воспроизводимый val-пул); (4) POMO работает на residual БЕЗ изменений; (5) _validate даёт ОБЕ оси
и отбирает ПО residual single-decode (== метрика гейта); (6) микс ~50/50. torch обязателен (pomo
тянет torch); тренер/POMO — ещё torch_geometric.
"""

from __future__ import annotations

import numpy as np
import pytest

from logistics_rl_gnn.config import congestion as cg
from logistics_rl_gnn.config import instance as im

pytest.importorskip("torch")
import torch  # noqa: E402

from logistics_rl_gnn.config.pomo import POMOConfig  # noqa: E402
from logistics_rl_gnn.env.events import make_dynamic_env  # noqa: E402
from logistics_rl_gnn.train.instance_sampler import InstanceSampler  # noqa: E402
from logistics_rl_gnn.train.pomo import feasible_starts  # noqa: E402
from logistics_rl_gnn.train.residual_curriculum import (  # noqa: E402
    _now_for_progress,
    greedy_cost,
    make_residual,
    single_decode_cost,
)

_DOW = im.DELIVERY_WEEKDAY
_K = im.FLEET_SIZE
_FR = (0.2, 0.8)


def test_residual_seed_disjoint():
    """0011 held-out: сид-диапазоны residual ∩ {0–9} = ∅; demand-реализации различны."""
    cfg = POMOConfig.for_residual()
    gate = set(range(10))
    assert cfg.res_train_base > 9 and cfg.res_train_base >= 3_000_000
    assert set(cfg.res_val_seeds()).isdisjoint(gate)
    assert min(cfg.res_val_seeds()) >= 4_000_000
    # train ∩ val-residual по сидам: train база 3M+seed(<1M) ⊂ [3M,4M); val ⊂ [4M,4M+48) → ∅
    assert cfg.res_train_base + 1_000_000 <= min(cfg.res_val_seeds())
    # реализации спроса held-out (та же геометрия — реальный город, разный спрос по сиду)
    s = InstanceSampler(n_range=(62, 62))
    assert not np.array_equal(
        s.sample(cfg.res_train_base).demand, s.sample(min(cfg.res_val_seeds())).demand
    )


def test_make_residual_feasible_and_pending():
    """residual не вырожден: ≥3 pending, ≥2 допустимых старта, тег real+синтетика (запрет №5)."""
    s = InstanceSampler(n_range=(62, 62))
    r = make_residual(s, 3_000_000, dow=_DOW, base_k=_K, frac_range=_FR)
    assert len(r.inst.demand) - 1 >= 3  # pending
    assert r.fleet >= 1 and 0.2 <= r.frac <= 0.8
    assert r.kind in ("traffic", "urgent", "breakdown")
    env = make_dynamic_env(r.inst, travel=r.travel, fleet_size=r.fleet)
    obs, _ = env.reset(seed=0)
    assert len(feasible_starts(env, obs, 8)) >= 2  # POMO shared baseline
    assert r.inst.meta.get("tag") == cg.CALIBRATION_TAG and r.inst.meta.get("dynamic")


def test_make_residual_deterministic():
    """Тот же сид → тот же residual (воспроизводимый held-out val-пул)."""
    s = InstanceSampler(n_range=(62, 62))
    a = make_residual(s, 4_000_000, dow=_DOW, base_k=_K, frac_range=_FR)
    b = make_residual(s, 4_000_000, dow=_DOW, base_k=_K, frac_range=_FR)
    assert np.array_equal(a.inst.demand, b.inst.demand)
    assert a.kind == b.kind and a.frac == b.frac and a.fleet == b.fleet


def test_now_for_progress_monotone():
    """now растёт с frac (больше прогресса → позже точка среза)."""
    fin = {i: float(i) for i in range(1, 11)}
    assert _now_for_progress(fin, 0.2) < _now_for_progress(fin, 0.8)


def test_pomo_multistart_on_residual():
    """POMO на residual БЕЗ изменений: feasible_starts = K следующих узлов → cost конечн."""
    pytest.importorskip("torch_geometric")
    from logistics_rl_gnn.models.policy import VRPPolicy
    from logistics_rl_gnn.train.pomo import multistart_greedy

    torch.manual_seed(0)
    s = InstanceSampler(n_range=(62, 62))
    r = make_residual(s, 3_000_001, dow=_DOW, base_k=_K, frac_range=_FR)
    env = make_dynamic_env(r.inst, travel=r.travel, fleet_size=r.fleet)
    c, routes = multistart_greedy(VRPPolicy(), env, 6)
    assert np.isfinite(c) and routes is not None


def _tiny_trainer():
    """Лёгкий ResidualPOMOTrainer (2 full-val + 2 res-val) для проверки проводки."""
    pytest.importorskip("torch_geometric")
    from logistics_rl_gnn.models.policy import VRPPolicy
    from logistics_rl_gnn.train.residual_curriculum import ResidualPOMOTrainer

    torch.manual_seed(0)
    cfg = POMOConfig.for_residual(
        epochs=1, steps_per_epoch=2, batch=4, max_starts=4, patience=2,
        n_range=(15, 20), val_range=(1_000_000, 1_000_002),
        test_range=(2_000_000, 2_000_002), res_val_range=(4_000_000, 4_000_002),
        warm_start=None, ckpt=None,
    )  # fmt: skip
    return ResidualPOMOTrainer(VRPPolicy(), cfg), cfg


def test_validate_both_axes_and_residual_selection():
    """_validate: обе оси; val_cost == residual single-decode (метрика отбора == метрика гейта)."""
    trainer, _ = _tiny_trainer()
    rec = trainer._validate()
    assert "val_res_cost" in rec and "val_full_cost" in rec  # обе оси
    assert rec["val_cost"] == rec["val_res_cost"]  # ОТБОР ПО residual (0011)
    man = float(np.mean([single_decode_cost(trainer.policy, e) for e in trainer.val_res_envs]))
    assert abs(rec["val_res_cost"] - man) < 1e-6  # == single-decode (== rl_raw гейта)
    assert trainer.val_res_heur > 0  # greedy-референс (== greedy_raw гейта) построен


def test_batch_mix_ratio():
    """Микс ~50/50 (seeded): доля residual-эпизодов (meta.dynamic) близка к residual_frac."""
    trainer, _ = _tiny_trainer()
    dyn = []
    for s in range(30):
        _, env = trainer._batch_instance_env(s)
        dyn.append(bool(env._inst.meta.get("dynamic")))
    assert 0.35 <= sum(dyn) / len(dyn) <= 0.65  # ~0.5


def test_greedy_cost_matches_scorer():
    """greedy_cost — тот же evaluate_solution под travel (== greedy_raw гейта)."""
    from logistics_rl_gnn.baselines.greedy import greedy_routes
    from logistics_rl_gnn.env.scoring import evaluate_solution

    s = InstanceSampler(n_range=(62, 62))
    r = make_residual(s, 4_000_005, dow=_DOW, base_k=_K, frac_range=_FR)
    env = make_dynamic_env(r.inst, travel=r.travel, fleet_size=r.fleet)
    gr = greedy_routes(env=env)
    assert greedy_cost(env) == -evaluate_solution(gr, env._inst, env._cost_cfg, travel=env.travel)[
        "reward"
    ]
