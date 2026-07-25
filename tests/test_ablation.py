"""Phase 6b Ablation — guards for the RL latency niche.

We check: (1) the end-to-end budget holds — latency ≤ budget+ε, and the deadline is REALLY hit
(otherwise the test is vacuous); (2) one scorer; (3) determinism at CONVERGENCE (a generous budget
on a small instance — not budget-bound); (4) measure_event emits every system at every budget and
raw_cost is consistent with the start. torch/torch_geometric required (run_ablation pulls policy).
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
    """A metric instance (times in SECONDS, like the real one): wide windows, greedy-feasible."""
    rng = np.random.default_rng(seed)
    coords = np.column_stack([10.0 + rng.uniform(0, spread, n), 48.0 + rng.uniform(0, spread, n)])
    d = np.hypot(coords[:, None, 0] - coords[None, :, 0], coords[:, None, 1] - coords[None, :, 1])
    tm = d * 6000.0  # seconds
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
    """Latency ≤ budget+ε, and the deadline is REALLY hit (unbounded polish > budget, adv #4)."""
    inst = _inst(40)
    fleet = 8
    gr = ab._greedy(inst, None, fleet)
    # premise: without a budget polish on n=40 does NOT converge instantly (else vacuous test)
    t0 = time.perf_counter()
    ab.polish(gr, inst, None, budget_ms=10000.0, fleet_size=fleet)
    conv_ms = (time.perf_counter() - t0) * 1000.0
    assert conv_ms > 50.0, f"the instance converges in {conv_ms:.0f}ms ≤50 — no deadline tested"
    # ε ≈ a few full evals (polish init eval/feas run BEFORE the deadline clock → bounded overshoot)
    t0 = time.perf_counter()
    ab._cost(gr, inst, None)
    eval_ms = (time.perf_counter() - t0) * 1000.0
    budget = 50.0
    out = ab._run_budget(lambda: ab._greedy(inst, None, fleet), inst, None, fleet, budget)
    eps = 6 * eval_ms + 20.0
    assert out["latency_ms"] <= budget + eps, f"{out['latency_ms']:.0f}ms > {budget + eps:.0f}ms"
    assert out["polish_ms"] > 0.0, "polish did not run in the remaining budget"


def test_single_scorer_identity():
    """_cost is exactly evaluate_solution (one scorer for all systems, prohibition #3)."""
    inst = _inst(12)
    gr = ab._greedy(inst, None, 8)
    assert ab._cost(gr, inst, None) == -evaluate_solution(gr, inst, CostConfig(), travel=None)[
        "reward"
    ]


def test_run_budget_raw_cost_matches_and_polish_not_worse():
    """raw_cost = scorer(start); polish is never worse than the input — one cost path."""
    inst = _inst(15)
    out = ab._run_budget(lambda: ab._greedy(inst, None, 8), inst, None, 8, 100.0)
    assert out["raw_cost"] == ab._cost(ab._greedy(inst, None, 8), inst, None)
    assert out["cost"] <= out["raw_cost"] + 1e-9


def test_polish_system_deterministic_at_convergence():
    """At CONVERGENCE (small instance, generous budget) cost is deterministic (not budget-bound)."""
    inst = _inst(7)
    a = ab._run_budget(lambda: ab._greedy(inst, None, 8), inst, None, 8, 5000.0)
    b = ab._run_budget(lambda: ab._greedy(inst, None, 8), inst, None, 8, 5000.0)
    assert abs(a["cost"] - b["cost"]) < 1e-9


def _decode_inst(n_pharm=4):
    """A decode-compatible instance (depot + pharmacies, wide windows) for measure_event."""
    return _inst(n_pharm + 1, spread=0.2)


def test_measure_event_emits_all_systems_and_raw_consistent():
    """measure_event: 4 systems × budget; rl_polish.raw_cost == rl_raw.cost (advisor #5/#2)."""
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
        assert abs(rl_pol["raw_cost"] - rl_raw["cost"]) < 1e-9  # same decode, same scorer
