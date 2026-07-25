"""Phase 4 baseline tests. The single scorer on tiny — no snapshot/ortools needed (only the env
runtime deps: `.[env]`); greedy/OR-Tools on the real snapshot — skipped without them.
"""

from __future__ import annotations

import pytest
from test_env import _tiny_instance  # reuse the Phase 3 tiny (depot + 2 pharmacies, known cost)

from logistics_rl_gnn.baselines.greedy import greedy_routes
from logistics_rl_gnn.config import instance as im
from logistics_rl_gnn.env.scoring import CostConfig, evaluate_solution


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


# ---------- the single scorer on tiny (always) ----------


def test_evaluate_tiny_known_cost():
    # route 0→1→2→0: time=10+4+15+4+20=53min; distance=5+7.5+10=22.5km; vehicles=1 (see test_env)
    r = evaluate_solution([[0, 1, 2, 0]], _tiny_instance(), CostConfig())
    expected = -(50.0 * 1 + 0.5 * 22.5 + 20.0 * 53.0 / 60.0)
    assert r["reward"] == pytest.approx(expected)
    assert r["distance_km"] == pytest.approx(22.5)
    assert r["time_min"] == pytest.approx(53.0)
    assert r["vehicles_used"] == 1
    assert r["unserved"] == 0
    assert r["on_time_pct"] == pytest.approx(100.0)


def test_evaluate_counts_unserved():
    # only pharmacy 1 is served → 1 unserved; an empty trailing route is not counted as a vehicle
    r = evaluate_solution([[0, 1, 0], [0]], _tiny_instance(), CostConfig())
    assert r["unserved"] == 1
    assert r["vehicles_used"] == 1


# ---------- greedy (real snapshot) ----------


@_NEED_SNAP
def test_greedy_serves_all_on_time():
    for seed in (0, 1):
        inst = im.generate_instance(seed=seed)
        r = evaluate_solution(greedy_routes(seed=seed), inst, CostConfig())
        assert r["unserved"] == 0, f"greedy left customers unserved (seed={seed})"
        assert r["on_time_pct"] == pytest.approx(100.0)  # late=0 under TW masking


@_NEED_SNAP
def test_greedy_deterministic():
    assert greedy_routes(seed=3) == greedy_routes(seed=3)


# ---------- OR-Tools (real snapshot + ortools) ----------


@_NEED_OR
def test_ortools_serves_all_on_time():
    from logistics_rl_gnn.baselines import ortools_vrptw

    cfg = CostConfig()
    for seed in (0, 1):
        inst = im.generate_instance(seed=seed)
        routes = ortools_vrptw.ortools_routes(inst, cfg, time_limit_s=8)
        r = evaluate_solution(routes, inst, cfg)
        assert r["unserved"] == 0, f"OR-Tools left customers unserved (seed={seed})"
        assert r["on_time_pct"] == pytest.approx(100.0)


@_NEED_OR
def test_ortools_not_worse_than_greedy():
    from logistics_rl_gnn.baselines import ortools_vrptw

    inst = im.generate_instance(seed=0)
    cfg = CostConfig()
    r_g = evaluate_solution(greedy_routes(seed=0), inst, cfg)
    r_o = evaluate_solution(ortools_vrptw.ortools_routes(inst, cfg, time_limit_s=10), inst, cfg)
    assert r_o["reward"] >= r_g["reward"] - 1e-6, "OR-Tools worse than greedy — check objective"
