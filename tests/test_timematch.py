"""Task #15 — time-matched guards. The deterministic core (instance+scorer) always; parity
30s=611.1 (0002), monotonicity of the anytime curve and NON-conflation system/dynamics — under
skipif (durable json outside git, #1). Time-limited OR-Tools GLS = best-so-far → cost is
wall-clock-bound: monotonicity with tolerance is the anytime invariant (30s is NOT run in pytest).
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


_NEED_SNAP = pytest.mark.skipif(not _snap_ok(), reason="no snapshot")
_NEED_OR = pytest.mark.skipif(not (_snap_ok() and _has_ortools()), reason="no snapshot/ortools")
_NEED_TM = pytest.mark.skipif(not _TM.exists(), reason="timematch.json outside git (#1)")


# ---------- the deterministic harness core (always): instance + scorer ----------


@_NEED_SNAP
def test_harness_core_deterministic():
    """Instance and scorer are reproducible → all curve nondeterminism is OR-Tools wall-clock."""
    a, b = im.generate_instance(seed=2), im.generate_instance(seed=2)
    assert np.array_equal(a.demand, b.demand)
    assert np.array_equal(a.windows, b.windows)
    r = [[0, 1, 0]]
    assert evaluate_solution(r, a, CostConfig()) == evaluate_solution(r, b, CostConfig())


# ---------- sub-second budget (OR-Tools) ----------


@_NEED_OR
def test_ortools_subsecond_budget_feasible():
    """The 0.7s budget (task #15) is really accepted (seconds/nanos) and yields a feasible plan."""
    from logistics_rl_gnn.baselines import ortools_vrptw

    cfg = CostConfig()
    inst = im.generate_instance(seed=0)
    routes = ortools_vrptw.ortools_routes(inst, cfg, time_limit_s=0.7)
    r = evaluate_solution(routes, inst, cfg)
    assert routes and r["on_time_pct"] == pytest.approx(100.0)  # windows met (conservative ceil)


# ---------- the durable curve (skipif the json is outside git) ----------


@_NEED_TM
def test_timematch_parity_30s():
    """The 30s point == 611.1€ (0002) within the GLS wall-clock jitter (best-so-far, tol=2€)."""
    tm = json.loads(_TM.read_text())
    assert abs(tm["ortools_best_30s_eur"] - _ANCHOR_0002_ORTOOLS) < 2.0  # not bit-deterministic
    assert tm["parity"]["ok"] is True


@_NEED_TM
def test_timematch_curve_monotone():
    """Anytime invariant: cost_mean never grows with the budget (budgets 3× apart, jitter-safe)."""
    curve = json.loads(_TM.read_text())["curve"]
    costs = [pt["cost_mean"] for pt in curve]
    budgets = [pt["budget_s"] for pt in curve]
    assert budgets == sorted(budgets), "the curve must be ordered by increasing budget"
    for lo, hi in zip(costs, costs[1:], strict=False):
        assert hi <= lo + 0.5, f"non-monotonic: {hi:.1f} > {lo:.1f} at a larger budget"


@_NEED_TM
def test_timematch_no_static_dynamic_conflation():
    """Honesty (Phase 8): 631.6€ (statics) and 827€/689ms (dynamics) are DIFFERENT fields."""
    sr = json.loads(_TM.read_text())["system_ref"]
    assert abs(sr["cost_eur"] - _ANCHOR_0009_SYSTEM) < 0.5  # static quality
    assert sr["static_wallclock_s"] >= 30.0  # the real static x is ≥30s, NOT 0.689s
    # the dynamic re-plan cost (residual) is markedly different from statics → no conflation
    assert abs(sr["dynamic_replan_cost_eur"] - sr["cost_eur"]) > 100.0


# ---------- paired discipline (0010): median-Δ + wins, not unpaired σ ----------


def test_paired_stats_logic():
    """paired_stats: wins = how often OR<system, median per-seed Δ (OR−system). Deterministic."""
    w, md = fm.paired_stats([10.0, 20.0, 30.0], [15.0, 15.0, 15.0])  # Δ=[-5,+5,+15]
    assert w == 1 and md == 5.0


@pytest.mark.skipif(not (_TM.exists() and _SM.exists()), reason="needs timematch+system json")
def test_paired_30s_median_negative():
    """At the max budget (30s) OR beats the system by MEDIAN (paired, the instance σ cancels)."""
    tm = json.loads(_TM.read_text())
    sys_ps = json.loads(_SM.read_text()).get("per_seed_cost_eur")
    if not sys_ps:
        pytest.skip("no per_seed for the system (an old system_metrics.json)")
    hi = max(tm["curve"], key=lambda c: c["budget_s"])  # 30s
    w, md = fm.paired_stats(hi["per_seed"], sys_ps)
    assert 0 <= w <= len(sys_ps)
    assert md < 0.0, f"OR@30s does not beat the system by median: {md:+.1f}€"
