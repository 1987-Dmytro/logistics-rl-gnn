"""Phase 6b Step 3.5 — local-search polish of decoded routes.

Guards (order = build order): the check_feasible oracle (parity with env semantics + the
cap/TW/T_max boundaries + agreement with evaluate_solution) → then the polish operators (never
worse, feasibility preserved, determinism, budget, optimum parity).
"""

from __future__ import annotations

import itertools
from datetime import datetime

import numpy as np
import pytest

from logistics_rl_gnn.config import instance as im
from logistics_rl_gnn.env.scoring import CostConfig, check_feasible, evaluate_solution

# tests using VRPEnv()/_real_greedy build an instance from the real snapshot — skipped without it
# (as test_env/test_model), else a bare runner (snapshot outside git, #1) → FileNotFoundError.
_NEED_SNAP = pytest.mark.skipif(im._latest_snapshot_dir() is None, reason="no snapshot")


def _mk(time_min, win_min, demand, svc_min, *, n):
    """A controlled instance: times/windows/service in MINUTES (converted to seconds inside)."""
    tm = np.asarray(time_min, dtype=float) * 60.0
    dm = np.asarray(time_min, dtype=float) * 100.0  # distance is arbitrary (does not affect feas)
    return im.Instance(
        node_ids=list(range(n)),
        snapshot_stops=list(range(n)),
        kinds=["depot"] + ["pharmacy"] * (n - 1),
        time_matrix=tm,
        dist_matrix=dm,
        coords=np.column_stack([np.linspace(10, 10.2, n), np.linspace(48, 48.2, n)]),
        windows=np.asarray(win_min, dtype=float) * 60.0,
        demand=np.asarray(demand, dtype=float),
        service=np.asarray(svc_min, dtype=float) * 60.0,
        tw_source=["DEPOT"] + ["REAL"] * (n - 1),
        excluded_stops=[],
        start_datetime=datetime(2024, 1, 2, 8),
        horizon_s=100000,
        meta={"seed": 0},
    )


def _boundary_inst():
    """3 nodes, symmetric 10 min/edge; c1,c2 demand 40, service 5 min; windows [0,100]."""
    t = [[0, 10, 10], [10, 0, 10], [10, 10, 0]]
    return _mk(t, [[0, 1000], [0, 100], [0, 100]], [0, 40, 40], [0, 5, 5], n=3)


# route [0,1,2,0]: arrive1=10 (+svc5→15), arrive2=25 (+svc5→30), return=40 min; load=80.


def test_check_feasible_boundary_cap():
    inst = _boundary_inst()
    r = [[0, 1, 2, 0]]
    assert check_feasible(r, inst, t_max_min=240, vehicle_cap=80, fleet_size=8)  # load 80 == Q
    assert not check_feasible(r, inst, t_max_min=240, vehicle_cap=79, fleet_size=8)  # 80 > 79


def test_check_feasible_boundary_tmax():
    inst = _boundary_inst()
    r = [[0, 1, 2, 0]]
    assert check_feasible(r, inst, t_max_min=40, vehicle_cap=80, fleet_size=8)  # return exactly 40
    assert not check_feasible(r, inst, t_max_min=39.9, vehicle_cap=80, fleet_size=8)


def test_check_feasible_boundary_tw():
    # window c1 = [0, 9]: arrival 10 > 9 → TW violated (hard as in the env, not via a penalty)
    t = [[0, 10, 10], [10, 0, 10], [10, 10, 0]]
    inst = _mk(t, [[0, 1000], [0, 9], [0, 100]], [0, 40, 40], [0, 5, 5], n=3)
    assert not check_feasible([[0, 1, 2, 0]], inst, t_max_min=240, vehicle_cap=80, fleet_size=8)


def test_check_feasible_fleet_cap():
    inst = _boundary_inst()
    two = [[0, 1, 0], [0, 2, 0]]  # 2 non-empty routes
    assert check_feasible(two, inst, t_max_min=240, vehicle_cap=80, fleet_size=2)
    assert not check_feasible(two, inst, t_max_min=240, vehicle_cap=80, fleet_size=1)


def test_check_feasible_per_customer_return_asymmetric():
    """Asymmetric return (non-metric, like OSM): return≤T_max is checked AFTER EVERY customer.

    Regression on a review finding: [0,1,2,0] finishes in 20 min, BUT a direct return from node 1
    costs 100 → the env would not build it (10+100>50). check_feasible must reject (env-strict).
    """
    t = [[0, 10, 10], [100, 0, 5], [10, 5, 0]]  # t(1,0)=100 — an expensive direct return from 1
    inst = _mk(t, [[0, 1000]] * 3, [0, 10, 10], [0, 0, 0], n=3)
    assert not check_feasible([[0, 1, 2, 0]], inst, t_max_min=50, vehicle_cap=80, fleet_size=8)
    assert check_feasible([[0, 1, 2, 0]], inst, t_max_min=120, vehicle_cap=80, fleet_size=8)


@_NEED_SNAP
def test_check_feasible_agrees_with_env_on_real_greedy():
    """A real greedy plan from the env is feasible per check_feasible (same semantics source)."""
    from logistics_rl_gnn.baselines.greedy import greedy_routes
    from logistics_rl_gnn.env.vrp_env import VRPEnv

    env = VRPEnv()
    routes = greedy_routes(env=env, seed=0)
    assert check_feasible(
        routes, env._inst, t_max_min=im.T_MAX_MIN, vehicle_cap=im.VEHICLE_CAP, fleet_size=env.K
    )


def test_check_feasible_agreement_with_evaluate_tw():
    """Agreement: with cap/T_max/fleet ok, the check_feasible TW verdict == on_time 100%."""
    t = [[0, 10, 10], [10, 0, 10], [10, 10, 0]]
    ok = _mk(t, [[0, 1000], [0, 100], [0, 100]], [0, 10, 10], [0, 5, 5], n=3)
    late = _mk(t, [[0, 1000], [0, 9], [0, 100]], [0, 10, 10], [0, 5, 5], n=3)
    r = [[0, 1, 2, 0]]
    cfg = CostConfig()
    assert check_feasible(r, ok, t_max_min=240, vehicle_cap=80, fleet_size=8)
    assert evaluate_solution(r, ok, cfg)["on_time_pct"] == pytest.approx(100.0)
    assert not check_feasible(r, late, t_max_min=240, vehicle_cap=80, fleet_size=8)
    assert evaluate_solution(r, late, cfg)["on_time_pct"] < 100.0  # both see the lateness


# ---------- polish operators ----------

import time  # noqa: E402

from logistics_rl_gnn.replan.local_search import (  # noqa: E402
    _or_opt,
    _relocate,
    _swap,
    _two_opt,
    polish,
)


def _custs(routes):
    """Multiset of (non-depot) customers over all routes — to check conservation."""
    return sorted(n for r in routes for n in r if n != 0)


@pytest.mark.parametrize("op", [_two_opt, _or_opt, _relocate, _swap])
def test_operator_conserves_customers_and_structure(op):
    """Operator: candidates preserve the customer multiset and the depot ends (no loss/dup)."""
    routes = [[0, 1, 2, 3, 0], [0, 4, 5, 0]]
    base = _custs(routes)
    n = 0
    for cand in op(routes):
        assert _custs(cand) == base, f"{op.__name__}: customers changed"
        for r in cand:
            assert r[0] == 0 and r[-1] == 0, f"{op.__name__}: a depot end is broken"
        n += 1
    assert n > 0, f"{op.__name__}: generated no candidates"


def _real_greedy(seed, travel_none=True):
    from logistics_rl_gnn.baselines.greedy import greedy_routes
    from logistics_rl_gnn.env.vrp_env import VRPEnv

    env = VRPEnv()
    routes = greedy_routes(env=env, seed=seed)
    return routes, env._inst


@_NEED_SNAP
@pytest.mark.parametrize("seed", range(6))
def test_polish_never_worse_and_feasible(seed):
    """Invariant: polish ≤ input AND feasible, over many real seeds (full-62 free-flow)."""
    routes, inst = _real_greedy(seed)
    cfg = CostConfig()
    base = -evaluate_solution(routes, inst, cfg)["reward"]
    out, cost = polish(routes, inst, None, budget_ms=300.0, fleet_size=im.FLEET_SIZE)
    assert cost <= base + 1e-6, f"seed {seed}: polish made it worse ({cost} > {base})"
    assert check_feasible(
        out, inst, t_max_min=im.T_MAX_MIN, vehicle_cap=im.VEHICLE_CAP, fleet_size=im.FLEET_SIZE
    ), f"seed {seed}: polish returned an infeasible solution"
    assert _custs(out) == _custs(routes), f"seed {seed}: pot/polish lost customers"


def test_polish_never_returns_env_infeasible_asymmetric():
    """Money path (verifier repro): a cheap 2-opt neighbour is env-infeasible — polish rejects it.

    4 nodes, T_max=50: the input [0,1,3,2,0] is feasible (return from 1 = 48≤50); the reversal →
    [0,3,1,2,0] is cheaper (20<21) BUT the return from 1 = 52>50 (env-infeasible). polish must
    return a feasible solution (strict check).
    """
    t = [[0, 6, 5, 5], [42, 0, 5, 5], [5, 5, 0, 5], [5, 5, 5, 0]]
    inst = _mk(t, [[0, 1000]] * 4, [0, 10, 10, 10], [0, 0, 0, 0], n=4)
    inp = [[0, 1, 3, 2, 0]]
    assert check_feasible(inp, inst, t_max_min=50, vehicle_cap=80, fleet_size=8)
    out, _ = polish(inp, inst, None, budget_ms=2000.0, t_max_min=50, fleet_size=8)
    ok = check_feasible(out, inst, t_max_min=50, vehicle_cap=80, fleet_size=8)
    assert ok, "polish returned an env-infeasible solution"


def test_polish_deterministic_at_convergence():
    """Determinism at convergence (no set/dict iteration). A SMALL instance → convergence in ms,
    the budget never binds even under load (at n=62 it could cut in → wall-clock nondeterminism)."""
    rng = np.random.default_rng(3)
    n = 9
    coords = rng.uniform(0, 8, size=(n, 2))
    d = np.hypot(coords[:, None, 0] - coords[None, :, 0], coords[:, None, 1] - coords[None, :, 1])
    inst = _mk(d.tolist(), [[0, 100000]] * n, [0.0] + [10.0] * (n - 1), [0] + [2] * (n - 1), n=n)
    routes = [[0, 1, 2, 3, 4, 0], [0, 5, 6, 7, 8, 0]]
    a = polish(routes, inst, None, budget_ms=5000.0, t_max_min=100000, fleet_size=4)
    b = polish(routes, inst, None, budget_ms=5000.0, t_max_min=100000, fleet_size=4)
    assert a[0] == b[0] and a[1] == pytest.approx(b[1])
    raw = -evaluate_solution(routes, inst, CostConfig())["reward"]
    assert a[1] <= raw + 1e-9  # genuinely converged (no worse than the input)


@_NEED_SNAP
def test_polish_respects_budget():
    """A small budget → return within the budget + slack (deadline checked before every eval)."""
    routes, inst = _real_greedy(0)
    t0 = time.perf_counter()
    polish(routes, inst, None, budget_ms=40.0, fleet_size=im.FLEET_SIZE)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert elapsed_ms < 40.0 + 200.0, f"budget 40ms, spent {elapsed_ms:.0f}ms"


def test_polish_leaves_optimum_untouched():
    """Parity: on a brute-force tiny optimum polish does NOT break it (it returns the optimum)."""
    # 3 customers, 1 vehicle; asymmetric times → a single cheap order
    t = [[0, 5, 9, 20], [5, 0, 6, 15], [9, 6, 0, 7], [20, 15, 7, 0]]
    inst = _mk(t, [[0, 1000]] * 4, [0, 10, 10, 10], [0, 2, 2, 2], n=4)
    cfg = CostConfig()

    best_r, best_c = None, float("inf")
    for perm in itertools.permutations([1, 2, 3]):
        r = [[0, *perm, 0]]
        if not check_feasible(r, inst, t_max_min=240, vehicle_cap=80, fleet_size=1):
            continue
        c = -evaluate_solution(r, inst, cfg)["reward"]
        if c < best_c:
            best_r, best_c = r, c
    out, cost = polish(best_r, inst, None, budget_ms=3000.0, fleet_size=1)
    assert cost == pytest.approx(best_c), "polish changed the optimum cost"
    assert _custs(out) == _custs(best_r)
