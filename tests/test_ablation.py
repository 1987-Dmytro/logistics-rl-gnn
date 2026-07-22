"""Phase 6b Ablation — стражи латентной ниши RL.

Проверяем: (1) end-to-end бюджет соблюдён — латентность ≤ budget+ε, причём дедлайн РЕАЛЬНО
достигается (иначе тест ложен); (2) единый скорер; (3) детерминизм у СХОДИМОСТИ (щедрый бюджет
на маленьком инстансе — не budget-bound); (4) measure_event эмитит все системы на каждый бюджет
и raw_cost консистентен со стартом. torch/torch_geometric обязательны (run_ablation тянет policy).
"""

from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from logistics_rl_gnn.config import instance as im
from logistics_rl_gnn.env.scoring import CostConfig, evaluate_solution

pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import run_ablation as ab  # noqa: E402


def _inst(n, *, spread=0.08, seed=0):
    """Метрический инстанс (времена в СЕК, как реальный): широкие окна, feasible под greedy."""
    rng = np.random.default_rng(seed)
    coords = np.column_stack([10.0 + rng.uniform(0, spread, n), 48.0 + rng.uniform(0, spread, n)])
    d = np.hypot(coords[:, None, 0] - coords[None, :, 0], coords[:, None, 1] - coords[None, :, 1])
    tm = d * 6000.0  # сек
    dm = d * 100000.0
    np.fill_diagonal(tm, 0.0)
    np.fill_diagonal(dm, 0.0)
    return im.Instance(
        node_ids=list(range(n)),
        snapshot_stops=list(range(n)),
        kinds=["depot"] + ["pharmacy"] * (n - 1),
        time_matrix=tm,
        dist_matrix=dm,
        coords=coords,
        windows=np.array([[0, 100000]] * n, dtype=float),
        demand=np.array([0.0] + [5.0] * (n - 1)),
        service=np.array([0.0] + [30.0] * (n - 1)),
        tw_source=["DEPOT"] + ["REAL"] * (n - 1),
        excluded_stops=[],
        start_datetime=datetime(2024, 1, 2, 8),
        horizon_s=100000,
        meta={"seed": seed},
    )


def test_run_budget_respects_deadline():
    """Латентность ≤ budget+ε, и дедлайн РЕАЛЬНО бьётся (unbounded polish > budget — advisor #4)."""
    inst = _inst(40)
    fleet = 8
    gr = ab._greedy(inst, None, fleet)
    # премиса: без бюджета polish на n=40 НЕ сходится мгновенно (иначе тест не трогает дедлайн)
    t0 = time.perf_counter()
    ab.polish(gr, inst, None, budget_ms=10000.0, fleet_size=fleet)
    conv_ms = (time.perf_counter() - t0) * 1000.0
    assert conv_ms > 50.0, f"инстанс сходится за {conv_ms:.0f}мс ≤50 — тест не проверяет дедлайн"
    # ε ≈ несколько full-eval (polish init eval/feas идут ДО дедлайн-часов → ограниченный overshoot)
    t0 = time.perf_counter()
    ab._cost(gr, inst, None)
    eval_ms = (time.perf_counter() - t0) * 1000.0
    budget = 50.0
    out = ab._run_budget(lambda: ab._greedy(inst, None, fleet), inst, None, fleet, budget)
    eps = 6 * eval_ms + 20.0
    assert out["latency_ms"] <= budget + eps, f"{out['latency_ms']:.0f}мс > {budget + eps:.0f}мс"
    assert out["polish_ms"] > 0.0, "polish не запустился в остаток бюджета"


def test_single_scorer_identity():
    """_cost — ровно evaluate_solution (единый скорер всех систем, запрет №3)."""
    inst = _inst(12)
    gr = ab._greedy(inst, None, 8)
    assert ab._cost(gr, inst, None) == -evaluate_solution(gr, inst, CostConfig(), travel=None)[
        "reward"
    ]


def test_run_budget_raw_cost_matches_and_polish_not_worse():
    """raw_cost = скорер(старта); polish НЕ хуже входа (инвариант) — единый путь стоимости."""
    inst = _inst(15)
    out = ab._run_budget(lambda: ab._greedy(inst, None, 8), inst, None, 8, 100.0)
    assert out["raw_cost"] == ab._cost(ab._greedy(inst, None, 8), inst, None)
    assert out["cost"] <= out["raw_cost"] + 1e-9


def test_polish_system_deterministic_at_convergence():
    """У СХОДИМОСТИ (маленький инстанс, щедрый бюджет) cost детерминирован (не budget-bound)."""
    inst = _inst(7)
    a = ab._run_budget(lambda: ab._greedy(inst, None, 8), inst, None, 8, 5000.0)
    b = ab._run_budget(lambda: ab._greedy(inst, None, 8), inst, None, 8, 5000.0)
    assert abs(a["cost"] - b["cost"]) < 1e-9


def _decode_inst(n_pharm=4):
    """Decode-совместимый инстанс (депо + аптеки, широкие окна) для measure_event."""
    return _inst(n_pharm + 1, spread=0.2)


def test_measure_event_emits_all_systems_and_raw_consistent():
    """measure_event: 4 системы × бюджет; rl_polish.raw_cost == rl_raw.cost (advisor #5/#2)."""
    from types import SimpleNamespace

    import torch

    from logistics_rl_gnn.models.policy import VRPPolicy

    torch.manual_seed(0)
    pol = VRPPolicy()
    inst = _decode_inst(4)
    ev = SimpleNamespace(kind="test", at_min=30.0)
    rows = ab.measure_event(pol, 0, ev, inst, None, 8)
    assert len(rows) == 4 * len(ab._BUDGETS_MS)
    for b in ab._BUDGETS_MS:
        systems = {r["system"] for r in rows if r["budget_ms"] == b}
        assert systems == {"rl_raw", "greedy_raw", "greedy_polish", "rl_polish"}
        rl_raw = next(r for r in rows if r["system"] == "rl_raw" and r["budget_ms"] == b)
        rl_pol = next(r for r in rows if r["system"] == "rl_polish" and r["budget_ms"] == b)
        assert abs(rl_pol["raw_cost"] - rl_raw["cost"]) < 1e-9  # тот же decode, тот же скорер
