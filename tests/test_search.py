"""Phase 6b Step 3 — inference search (sample-K + PortfolioPlanner). No training, decode only.

Guards: parity of the batched decode vs the single one (einsum transposition), determinism of
sample_k by seed, the portfolio ≤ greedy guarantee BY CONSTRUCTION at every event, latency logged,
the planner does not mutate the input instance. torch/torch_geometric required (else skipped).
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from logistics_rl_gnn.config import instance as im

pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
import torch  # noqa: E402

from logistics_rl_gnn.env.events import congestion_for, make_dynamic_env  # noqa: E402
from logistics_rl_gnn.models.policy import VRPPolicy  # noqa: E402


def _tiny(n_pharm=4):
    """A compact feasible instance (depot + n pharmacies, wide windows) for decode guards."""
    k = n_pharm + 1
    rng = np.random.default_rng(0)
    coords = np.column_stack([10.0 + rng.uniform(0, 0.2, k), 48.0 + rng.uniform(0, 0.2, k)])
    d = np.hypot(coords[:, None, 0] - coords[None, :, 0], coords[:, None, 1] - coords[None, :, 1])
    time_m = d * 6000.0  # ~seconds
    dist_m = d * 100000.0
    np.fill_diagonal(time_m, 0.0)
    np.fill_diagonal(dist_m, 0.0)
    return im.Instance(
        node_ids=list(range(k)),
        snapshot_stops=list(range(k)),
        kinds=["depot"] + ["pharmacy"] * n_pharm,
        time_matrix=time_m,
        dist_matrix=dist_m,
        coords=coords,
        windows=np.array([[0, 36000]] * k, dtype=float),
        demand=np.array([0.0] + [10.0] * n_pharm),
        service=np.array([0.0] + [120.0] * n_pharm),
        tw_source=["DEPOT"] + ["REAL"] * n_pharm,
        excluded_stops=[],
        start_datetime=datetime(2024, 1, 2, 8),
        horizon_s=36000,
        meta={"seed": 0},
    )


def test_logits_batch_parity_vs_single():
    """Batched decode over 1 row == single logits (guards einsum 'bd,nd->bn' + masked_fill)."""
    torch.manual_seed(0)
    pol = VRPPolicy()
    env = make_dynamic_env(_tiny(), fleet_size=2, t_max_min=1000.0)
    obs, _ = env.reset(seed=0)
    enc = pol.encode(env)
    # the initial state + the state after one step (different pos/mask/cur_time)
    for _ in range(2):
        ctx = pol._context(env, enc)
        mask = torch.as_tensor(obs["action_mask"], dtype=torch.float32)
        single = pol.decoder.logits(ctx, enc[2], mask)
        batch = pol.decoder.logits_batch(ctx.unsqueeze(0), enc[2], mask.unsqueeze(0))[0]
        fin = torch.isfinite(single)
        assert torch.allclose(single[fin], batch[fin], atol=1e-5), "batch ≠ single logits"
        assert (torch.isinf(single) == torch.isinf(batch)).all(), "the −inf mask diverged"
        a = int(torch.as_tensor(obs["action_mask"]).nonzero()[0])  # a feasible step
        obs, _, term, trunc, _ = env.step(a)
        if term or trunc:
            break


def test_sample_k_deterministic_by_seed():
    """sample_k(seed) is reproducible (its own generator): the same seed → the same K routes."""
    torch.manual_seed(0)
    pol = VRPPolicy()
    inst = _tiny()

    def run(seed):
        envs = [make_dynamic_env(inst, fleet_size=2, t_max_min=1000.0) for _ in range(8)]
        envs[0].reset(seed=0)
        return pol.sample_k(envs, pol.encode(envs[0]), temperature=0.8, seed=seed)

    assert run(7) == run(7), "the same seed → different rollouts (determinism broken)"


def test_sample_k_rejects_zero_temperature():
    torch.manual_seed(0)
    pol = VRPPolicy()
    envs = [make_dynamic_env(_tiny(), fleet_size=2, t_max_min=1000.0)]
    envs[0].reset(seed=0)
    with pytest.raises(AssertionError):
        pol.sample_k(envs, pol.encode(envs[0]), temperature=0.0)


def _planner_cases():
    """Several instances (free-flow + congestion) as a proxy for 'every event'."""
    cases = []
    for n in (3, 5, 6):
        inst = _tiny(n)
        cases.append((inst, None, 8))  # free-flow
        cases.append((inst, congestion_for(inst, dow=1, offset_min=30.0), 8))  # congestion
    return cases


def test_portfolio_never_worse_than_greedy_every_case():
    """Guarantee BY CONSTRUCTION: portfolio ≤ greedy on EVERY instance (the same scorer)."""
    from logistics_rl_gnn.replan.portfolio import PortfolioPlanner

    torch.manual_seed(0)
    planner = PortfolioPlanner(VRPPolicy(), k_samples=8, temperature=1.0, rl_starts=4)
    for inst, travel, fleet in _planner_cases():
        r = planner.plan(inst, travel, fleet_size=fleet)
        assert r["cost"] <= r["greedy_cost"] + 1e-6, f"portfolio {r['cost']} > {r['greedy_cost']}"
        assert r["source"] in ("greedy", "rl_greedy", "sample")


def test_take_best_skips_none_candidates():
    """multistart may return None (no feasible start) — take_best skips it."""
    from logistics_rl_gnn.env.scoring import CostConfig
    from logistics_rl_gnn.replan.portfolio import take_best

    inst = _tiny(3)
    good = [[0, 1, 2, 3, 0]]  # a valid route
    routes, cost, idx = take_best([None, good, None], inst, None, CostConfig())
    assert idx == 1 and routes is good and cost > 0


def test_portfolio_with_polish_preserves_guarantee():
    """Step 3.5: polishing the top-M in the portfolio does NOT break the ≤ greedy guarantee."""
    from logistics_rl_gnn.replan.portfolio import PortfolioPlanner

    torch.manual_seed(0)
    planner = PortfolioPlanner(
        VRPPolicy(), k_samples=8, rl_starts=4, polish_budget_ms=80.0, polish_top_m=3
    )
    for inst, travel, fleet in _planner_cases():
        r = planner.plan(inst, travel, fleet_size=fleet)
        assert r["cost"] <= r["greedy_cost"] + 1e-6, "polish broke the ≤ greedy guarantee"


def test_portfolio_logs_latency():
    from logistics_rl_gnn.replan.portfolio import PortfolioPlanner

    torch.manual_seed(0)
    planner = PortfolioPlanner(VRPPolicy(), k_samples=8, rl_starts=4)
    r = planner.plan(_tiny(5), None, fleet_size=8)
    assert r["latency_ms"] >= 0.0 and "latency_ms" in r


def test_planner_does_not_mutate_instance():
    """Eval does not mutate the input: instance arrays intact after plan (the planner copies)."""
    from logistics_rl_gnn.replan.portfolio import PortfolioPlanner

    torch.manual_seed(0)
    inst = _tiny(5)
    tm, dem, win = inst.time_matrix.copy(), inst.demand.copy(), inst.windows.copy()
    PortfolioPlanner(VRPPolicy(), k_samples=8, rl_starts=4).plan(inst, None, fleet_size=8)
    assert np.array_equal(inst.time_matrix, tm)
    assert np.array_equal(inst.demand, dem)
    assert np.array_equal(inst.windows, win)
