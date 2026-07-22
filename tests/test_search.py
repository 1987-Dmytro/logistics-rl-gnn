"""Phase 6b Шаг 3 — инференс-поиск (sample-K + PortfolioPlanner). Без обучения, только decode.

Стражи: parity батч-декода vs одиночного (einsum-транспонировка), детерминизм sample_k по seed,
гарантия portfolio ≤ greedy ПО ПОСТРОЕНИЮ на каждом событии, латентность логируется, планер
не мутирует вход-инстанс. torch/torch_geometric обязательны (иначе skip).
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
    """Компактный feasible инстанс (депо + n аптек, широкие окна) для decode-стражей."""
    k = n_pharm + 1
    rng = np.random.default_rng(0)
    coords = np.column_stack([10.0 + rng.uniform(0, 0.2, k), 48.0 + rng.uniform(0, 0.2, k)])
    d = np.hypot(coords[:, None, 0] - coords[None, :, 0], coords[:, None, 1] - coords[None, :, 1])
    time_m = d * 6000.0  # ~сек
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
    """Батч-декод по 1 строке == одиночный logits (страж einsum 'bd,nd->bn' и masked_fill)."""
    torch.manual_seed(0)
    pol = VRPPolicy()
    env = make_dynamic_env(_tiny(), fleet_size=2, t_max_min=1000.0)
    obs, _ = env.reset(seed=0)
    enc = pol.encode(env)
    # начальное состояние + состояние после одного шага (разные pos/mask/cur_time)
    for _ in range(2):
        ctx = pol._context(env, enc)
        mask = torch.as_tensor(obs["action_mask"], dtype=torch.float32)
        single = pol.decoder.logits(ctx, enc[2], mask)
        batch = pol.decoder.logits_batch(ctx.unsqueeze(0), enc[2], mask.unsqueeze(0))[0]
        fin = torch.isfinite(single)
        assert torch.allclose(single[fin], batch[fin], atol=1e-5), "батч ≠ одиночный logits"
        assert (torch.isinf(single) == torch.isinf(batch)).all(), "маска −inf разошлась"
        a = int(torch.as_tensor(obs["action_mask"]).nonzero()[0])  # feasible шаг
        obs, _, term, trunc, _ = env.step(a)
        if term or trunc:
            break


def test_sample_k_deterministic_by_seed():
    """sample_k(seed) воспроизводим (свой генератор): тот же seed → те же K маршрутов."""
    torch.manual_seed(0)
    pol = VRPPolicy()
    inst = _tiny()

    def run(seed):
        envs = [make_dynamic_env(inst, fleet_size=2, t_max_min=1000.0) for _ in range(8)]
        envs[0].reset(seed=0)
        return pol.sample_k(envs, pol.encode(envs[0]), temperature=0.8, seed=seed)

    assert run(7) == run(7), "тот же seed → разные роллауты (детерминизм сломан)"


def test_sample_k_rejects_zero_temperature():
    torch.manual_seed(0)
    pol = VRPPolicy()
    envs = [make_dynamic_env(_tiny(), fleet_size=2, t_max_min=1000.0)]
    envs[0].reset(seed=0)
    with pytest.raises(AssertionError):
        pol.sample_k(envs, pol.encode(envs[0]), temperature=0.0)


def _planner_cases():
    """Несколько инстансов (free-flow + congestion) как прокси 'каждого события'."""
    cases = []
    for n in (3, 5, 6):
        inst = _tiny(n)
        cases.append((inst, None, 8))  # free-flow
        cases.append((inst, congestion_for(inst, dow=1, offset_min=30.0), 8))  # congestion
    return cases


def test_portfolio_never_worse_than_greedy_every_case():
    """Гарантия ПО ПОСТРОЕНИЮ: portfolio ≤ greedy на КАЖДОМ инстансе (тот же scorer)."""
    from logistics_rl_gnn.replan.portfolio import PortfolioPlanner

    torch.manual_seed(0)
    planner = PortfolioPlanner(VRPPolicy(), k_samples=8, temperature=1.0, rl_starts=4)
    for inst, travel, fleet in _planner_cases():
        r = planner.plan(inst, travel, fleet_size=fleet)
        assert r["cost"] <= r["greedy_cost"] + 1e-6, f"portfolio {r['cost']} > {r['greedy_cost']}"
        assert r["source"] in ("greedy", "rl_greedy", "sample")


def test_take_best_skips_none_candidates():
    """multistart может вернуть None (нет feasible-старта) — take_best его пропускает."""
    from logistics_rl_gnn.env.scoring import CostConfig
    from logistics_rl_gnn.replan.portfolio import take_best

    inst = _tiny(3)
    good = [[0, 1, 2, 3, 0]]  # валидный маршрут
    routes, cost, idx = take_best([None, good, None], inst, None, CostConfig())
    assert idx == 1 and routes is good and cost > 0


def test_portfolio_with_polish_preserves_guarantee():
    """Шаг 3.5: polish топ-M в портфеле НЕ ломает гарантию ≤ greedy (исходный greedy в пуле)."""
    from logistics_rl_gnn.replan.portfolio import PortfolioPlanner

    torch.manual_seed(0)
    planner = PortfolioPlanner(
        VRPPolicy(), k_samples=8, rl_starts=4, polish_budget_ms=80.0, polish_top_m=3
    )
    for inst, travel, fleet in _planner_cases():
        r = planner.plan(inst, travel, fleet_size=fleet)
        assert r["cost"] <= r["greedy_cost"] + 1e-6, "polish сломал гарантию ≤ greedy"


def test_portfolio_logs_latency():
    from logistics_rl_gnn.replan.portfolio import PortfolioPlanner

    torch.manual_seed(0)
    planner = PortfolioPlanner(VRPPolicy(), k_samples=8, rl_starts=4)
    r = planner.plan(_tiny(5), None, fleet_size=8)
    assert r["latency_ms"] >= 0.0 and "latency_ms" in r


def test_planner_does_not_mutate_instance():
    """Евал не мутирует вход: массивы инстанса целы после plan (планер строит свои env-копии)."""
    from logistics_rl_gnn.replan.portfolio import PortfolioPlanner

    torch.manual_seed(0)
    inst = _tiny(5)
    tm, dem, win = inst.time_matrix.copy(), inst.demand.copy(), inst.windows.copy()
    PortfolioPlanner(VRPPolicy(), k_samples=8, rl_starts=4).plan(inst, None, fleet_size=8)
    assert np.array_equal(inst.time_matrix, tm)
    assert np.array_equal(inst.demand, dem)
    assert np.array_equal(inst.windows, win)
