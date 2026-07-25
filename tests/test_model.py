"""Sanity tests for the Phase 5 policy (WITHOUT real training — that is Phase 6).

forward validity, masking, overfit-tiny (the architecture can learn), determinism/NaN.
Skipped without torch/torch-geometric.
"""

from __future__ import annotations

import numpy as np
import pytest
from test_env import _tiny_env, _tiny_instance  # reuse the Phase 3 tiny

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
    assert probs.shape == (env.k,)  # π over N+1 actions
    assert torch.all(probs >= 0) and torch.allclose(probs.sum(), torch.tensor(1.0), atol=1e-5)
    assert not torch.isnan(probs).any()
    # logπ of feasible actions is finite
    for j in np.flatnonzero(obs["action_mask"]):
        assert torch.isfinite(dist.log_prob(torch.tensor(int(j))))


# ---------- masking (the env as the single source) ----------


def test_masking_infeasible_zero_prob():
    policy = _policy()
    env = _tiny_env()
    obs, _ = env.reset(seed=0)
    enc = policy.encode(env)  # the encoder is static → encode before the step
    obs, *_ = env.step(1)  # pharmacy 1 served → visited → infeasible
    dist = policy.action_dist(env, obs, enc)
    probs = dist.probs.detach()
    assert obs["action_mask"][1] == 0  # the env masks it itself
    assert probs[1] < 1e-6  # the decoder gives no probability to an infeasible action
    assert torch.allclose(probs.sum(), torch.tensor(1.0), atol=1e-5)


# ---------- overfit tiny: the architecture can learn (a few REINFORCE steps) ----------


def test_overfit_tiny_cost_drops():
    policy = _policy(seed=0)
    opt = torch.optim.Adam(policy.parameters(), lr=1e-3)  # stable step (Phase 5/6 finding)
    env = _tiny_env()  # THE SAME config (fleet=1, t_max=1000) → both pharmacies feasible

    def mean_cost(n=16):
        return float(np.mean([-policy.rollout(env, mode="sample")[2]["reward"] for _ in range(n)]))

    init = mean_cost()
    curve = []
    for _ in range(200):
        logps, rewards = [], []
        for _ in range(8):  # batch>=8 → baseline=mean != reward (else the gradient is zero)
            _, lp, m = policy.rollout(env, mode="sample")
            logps.append(lp)
            rewards.append(m["reward"])
        R = torch.tensor(rewards)
        # REINFORCE: maximise reward (=−cost) → loss = −((R−baseline)·Σlogπ)
        loss = -((R - R.mean()) * torch.stack(logps)).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)  # stabilises REINFORCE
        opt.step()
        curve.append(-float(R.mean()))  # mean batch cost

    final = mean_cost()
    greedy_cost = -policy.rollout(env, mode="greedy")[2]["reward"]
    opt_cost = -evaluate_reward([[0, 1, 2, 0]])  # tiny optimum (symmetric: both permutations)
    assert final < init, f"cost did not fall: {init:.1f} → {final:.1f}"
    assert greedy_cost < opt_cost + 6.0, f"greedy did not reach the optimum: {greedy_cost:.1f}"
    assert not np.isnan(curve).any()


def evaluate_reward(routes):
    from logistics_rl_gnn.env.scoring import CostConfig, evaluate_solution

    return evaluate_solution(routes, _tiny_instance(), CostConfig())["reward"]


# ---------- determinism / NaN ----------


def test_forward_deterministic_no_nan():
    def fwd():
        policy = _policy(seed=42)
        env = _tiny_env()
        obs, _ = env.reset(seed=0)
        return policy.action_dist(env, obs, policy.encode(env)).probs.detach()

    p1, p2 = fwd(), fwd()
    assert torch.allclose(p1, p2)  # weight seed → reproducible forward
    assert not torch.isnan(p1).any()


# ---------- real instance, K>1 (the vehicle-switch path) ----------


@pytest.mark.skipif(im._latest_snapshot_dir() is None, reason="no snapshot")
def test_rollout_real_multivehicle():
    from logistics_rl_gnn.env.vrp_env import VRPEnv

    policy = _policy()
    routes, sum_logp, m = policy.rollout(VRPEnv(), mode="greedy")  # K=8: depot return→next vehicle
    assert torch.isfinite(sum_logp) and not torch.isnan(sum_logp)  # the episode ran to the end
    assert len(routes) >= 1
    for rt in routes:  # every route starts/finishes at the depot
        assert rt[0] == 0 and rt[-1] == 0
    assert m.keys() >= {"reward", "vehicles_used", "unserved"}  # evaluate_solution metrics
