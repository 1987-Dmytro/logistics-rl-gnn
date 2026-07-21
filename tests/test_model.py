"""Sanity-тесты политики Phase 5 (БЕЗ полноценного обучения — это Phase 6).

forward-валидность, masking, overfit-tiny (арх способна учиться), детерминизм/NaN.
skip без torch/torch-geometric.
"""

from __future__ import annotations

import numpy as np
import pytest
from test_env import _tiny_env, _tiny_instance  # реюз Phase 3 tiny

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from logistics_rl_gnn.config import instance as im  # noqa: E402
from logistics_rl_gnn.models.policy import VRPPolicy  # noqa: E402

_D = 128


def _policy(seed=0):
    torch.manual_seed(seed)
    return VRPPolicy(d_model=_D)


# ---------- forward ----------


def test_forward_shapes_and_distribution():
    policy = _policy()
    env = _tiny_env()
    obs, _ = env.reset(seed=0)
    node_embs, graph_emb, _ = enc = policy.encode(env)
    assert node_embs.shape == (env.k, _D)
    assert graph_emb.shape == (_D,)
    dist = policy.action_dist(env, obs, enc)
    probs = dist.probs
    assert probs.shape == (env.k,)  # π по N+1 действиям
    assert torch.all(probs >= 0) and torch.allclose(probs.sum(), torch.tensor(1.0), atol=1e-5)
    assert not torch.isnan(probs).any()
    # logπ feasible-действий конечны
    for j in np.flatnonzero(obs["action_mask"]):
        assert torch.isfinite(dist.log_prob(torch.tensor(int(j))))


# ---------- masking (env как single source) ----------


def test_masking_infeasible_zero_prob():
    policy = _policy()
    env = _tiny_env()
    obs, _ = env.reset(seed=0)
    enc = policy.encode(env)  # энкодер статичен → кодируем до шага
    obs, *_ = env.step(1)  # обслужили аптеку 1 → visited → инфеасибл
    dist = policy.action_dist(env, obs, enc)
    probs = dist.probs.detach()
    assert obs["action_mask"][1] == 0  # env сам маскирует
    assert probs[1] < 1e-6  # декодер не даёт вероятность инфеасибл
    assert torch.allclose(probs.sum(), torch.tensor(1.0), atol=1e-5)


# ---------- overfit tiny: арх способна учиться (несколько REINFORCE-шагов) ----------


def test_overfit_tiny_cost_drops():
    policy = _policy(seed=0)
    opt = torch.optim.Adam(
        policy.parameters(), lr=1e-3
    )  # 1e-2 нестабилен: tanh-clip → uniform-коллапс
    env = _tiny_env()  # ТОТ ЖЕ конфиг (fleet=1, t_max=1000) → обе аптеки feasible

    def mean_cost(n=16):
        return float(np.mean([-policy.rollout(env, mode="sample")[2]["reward"] for _ in range(n)]))

    init = mean_cost()
    curve = []
    for _ in range(200):
        logps, rewards = [], []
        for _ in range(8):  # batch>=8 → baseline=mean != reward (иначе нулевой градиент)
            _, lp, m = policy.rollout(env, mode="sample")
            logps.append(lp)
            rewards.append(m["reward"])
        R = torch.tensor(rewards)
        # REINFORCE: максимизируем reward (=−cost) → loss = −((R−baseline)·Σlogπ)
        loss = -((R - R.mean()) * torch.stack(logps)).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)  # стабилизирует REINFORCE
        opt.step()
        curve.append(-float(R.mean()))  # средняя стоимость батча

    final = mean_cost()
    greedy_cost = -policy.rollout(env, mode="greedy")[2]["reward"]
    opt_cost = -evaluate_reward([[0, 1, 2, 0]])  # оптимум tiny (симметричен: обе перестановки)
    assert final < init, f"стоимость не упала: {init:.1f} → {final:.1f}"
    assert greedy_cost < opt_cost + 6.0, f"greedy не сошёлся к оптимуму: {greedy_cost:.1f}"
    assert not np.isnan(curve).any()


def evaluate_reward(routes):
    from logistics_rl_gnn.env.scoring import CostConfig, evaluate_solution

    return evaluate_solution(routes, _tiny_instance(), CostConfig())["reward"]


# ---------- детерминизм / NaN ----------


def test_forward_deterministic_no_nan():
    def fwd():
        policy = _policy(seed=42)
        env = _tiny_env()
        obs, _ = env.reset(seed=0)
        return policy.action_dist(env, obs, policy.encode(env)).probs.detach()

    p1, p2 = fwd(), fwd()
    assert torch.allclose(p1, p2)  # seed весов → воспроизводимый forward
    assert not torch.isnan(p1).any()


# ---------- реальный инстанс, K>1 (путь смены машины) ----------


@pytest.mark.skipif(im._latest_snapshot_dir() is None, reason="нет снапшота")
def test_rollout_real_multivehicle():
    from logistics_rl_gnn.env.vrp_env import VRPEnv

    policy = _policy()
    routes, sum_logp, m = policy.rollout(VRPEnv(), mode="greedy")  # K=8: депо-возврат→новая машина
    assert torch.isfinite(sum_logp) and not torch.isnan(sum_logp)  # эпизод дошёл до конца
    assert len(routes) >= 1
    for rt in routes:  # каждый маршрут старт/финиш в депо
        assert rt[0] == 0 and rt[-1] == 0
    assert m.keys() >= {"reward", "vehicles_used", "unserved"}  # метрики evaluate_solution
