"""Задача #15 — стражи time-matched. Детерминир. ядро (instance+scorer) — всегда; парити 30с=611.1
(0002), монотонность anytime-кривой и НЕ-конфляция система/динамика — под skipif (durable json вне
git, запрет №1). OR-Tools time-limited GLS = best-so-far → cost wall-clock-bound (не бит-детермин.):
монотонность-с-допуском — верный anytime-инвариант, не бит-равенство (30с в pytest НЕ гоняем).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from logistics_rl_gnn.config import instance as im
from logistics_rl_gnn.env.scoring import CostConfig, evaluate_solution

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "scripts"))
import final_metrics as fm  # noqa: E402

_TM = _ROOT / "results" / "timematch.json"
_SM = _ROOT / "results" / "system_metrics.json"
_ANCHOR_0002_ORTOOLS = 611.14
_ANCHOR_0009_SYSTEM = 631.62


def _snap_ok() -> bool:
    return im._latest_snapshot_dir() is not None


def _has_ortools() -> bool:
    try:
        import ortools  # noqa: F401

        return True
    except ImportError:
        return False


_NEED_SNAP = pytest.mark.skipif(not _snap_ok(), reason="нет снапшота")
_NEED_OR = pytest.mark.skipif(not (_snap_ok() and _has_ortools()), reason="нет снапшота/ortools")
_NEED_TM = pytest.mark.skipif(not _TM.exists(), reason="timematch.json вне git (запрет №1)")


# ---------- детерминир. ядро harness (всегда): instance + scorer ----------


@_NEED_SNAP
def test_harness_core_deterministic():
    """Инстанс и оценщик воспроизводимы → вся недетерминированность кривой = OR-Tools wall-clock."""
    a, b = im.generate_instance(seed=2), im.generate_instance(seed=2)
    assert np.array_equal(a.demand, b.demand)
    assert np.array_equal(a.windows, b.windows)
    r = [[0, 1, 0]]
    assert evaluate_solution(r, a, CostConfig()) == evaluate_solution(r, b, CostConfig())


# ---------- суб-секундный бюджет (OR-Tools) ----------


@_NEED_OR
def test_ortools_subsecond_budget_feasible():
    """0.7с-бюджет (task #15) реально принимается (seconds/nanos) и даёт feasible-решение."""
    from logistics_rl_gnn.baselines import ortools_vrptw

    cfg = CostConfig()
    inst = im.generate_instance(seed=0)
    routes = ortools_vrptw.ortools_routes(inst, cfg, time_limit_s=0.7)
    r = evaluate_solution(routes, inst, cfg)
    assert routes and r["on_time_pct"] == pytest.approx(100.0)  # окна соблюдены (консерв. ceil)


# ---------- durable-кривая (skipif json вне git) ----------


@_NEED_TM
def test_timematch_parity_30s():
    """30с-точка == 611.1€ (0002) в пределах wall-clock-джиттера GLS (best-so-far, tol=2€)."""
    tm = json.loads(_TM.read_text())
    assert abs(tm["ortools_best_30s_eur"] - _ANCHOR_0002_ORTOOLS) < 2.0  # не бит-детерм.
    assert tm["parity"]["ok"] is True


@_NEED_TM
def test_timematch_curve_monotone():
    """Anytime-инвариант: cost_mean не растёт с бюджетом (бюджеты 3× врозь, jitter не собьёт)."""
    curve = json.loads(_TM.read_text())["curve"]
    costs = [pt["cost_mean"] for pt in curve]
    budgets = [pt["budget_s"] for pt in curve]
    assert budgets == sorted(budgets), "кривая должна быть по возрастанию бюджета"
    for lo, hi in zip(costs, costs[1:], strict=False):
        assert hi <= lo + 0.5, f"немонотонно: {hi:.1f} > {lo:.1f} при большем бюджете"


@_NEED_TM
def test_timematch_no_static_dynamic_conflation():
    """Честность (Phase 8): 631.6€ (статика) и 827€/689мс (динамика) — РАЗНЫЕ поля."""
    sr = json.loads(_TM.read_text())["system_ref"]
    assert abs(sr["cost_eur"] - _ANCHOR_0009_SYSTEM) < 0.5  # статик-качество
    assert sr["static_wallclock_s"] >= 30.0  # реальный статик-x ≥30с, НЕ 0.689с
    # динамическая re-plan стоимость (residual) — заметно иная, чем статика → не конфляция
    assert abs(sr["dynamic_replan_cost_eur"] - sr["cost_eur"]) > 100.0


# ---------- парная дисциплина (0010): median-Δ + wins, не unpaired σ ----------


def test_paired_stats_logic():
    """paired_stats: wins = сколько OR<система, median per-seed Δ (OR−система). Детерминир."""
    w, md = fm.paired_stats([10.0, 20.0, 30.0], [15.0, 15.0, 15.0])  # Δ=[-5,+5,+15]
    assert w == 1 and md == 5.0


@pytest.mark.skipif(not (_TM.exists() and _SM.exists()), reason="нужны timematch+system json")
def test_paired_30s_median_negative():
    """На макс. бюджете (30с) OR по МЕДИАНЕ бьёт систему (парно, σ инстанса сокращается)."""
    tm = json.loads(_TM.read_text())
    sys_ps = json.loads(_SM.read_text()).get("per_seed_cost_eur")
    if not sys_ps:
        pytest.skip("per_seed системы нет (старый system_metrics.json)")
    hi = max(tm["curve"], key=lambda c: c["budget_s"])  # 30с
    w, md = fm.paired_stats(hi["per_seed"], sys_ps)
    assert 0 <= w <= len(sys_ps)
    assert md < 0.0, f"OR@30с по медиане не бьёт систему: {md:+.1f}€"
