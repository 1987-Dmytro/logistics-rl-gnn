"""REINFORCE with a rollout baseline (Kool, Phase 6).

baseline = a frozen copy of the policy (greedy on the SAME instances). advantage = cost of the
current sample − greedy baseline (paired), NORMALISED (mean0/std1) — otherwise the enormous scale
(all-depot baseline) drove softmax into saturation within 1 epoch (Phase 6 diag).
loss = mean(adv · Σlogπ) — the sign is +Σlogπ (∇L=E[(cost−b)∇Σlogπ]=∇E[cost], descent ↓cost; the
spec formula −Σlogπ would give ↑cost). Every epoch a paired t-test of greedy-current vs
greedy-baseline on val → significantly better ⇒ update. Runtime guard: grad_norm≈0 for several
epochs in a row (saturation) → abort the run (rather than 100 wasted epochs).
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
from scipy.stats import ttest_rel

from logistics_rl_gnn.baselines.greedy import greedy_routes
from logistics_rl_gnn.env.scoring import evaluate_solution
from logistics_rl_gnn.env.vrp_env import VRPEnv
from logistics_rl_gnn.train.instance_sampler import InstanceSampler

_FREEZE_PATIENCE = 3  # epochs in a row with grad_norm≈0 → abort the run (collapse guard)


def make_env(instance) -> VRPEnv:
    """VRPEnv on a fixed instance (instance_fn ignores seed → reset yields the same instance)."""
    return VRPEnv(instance_fn=lambda s: instance)


def _greedy_cost(policy, env) -> float:
    with torch.no_grad():
        return -policy.rollout(env, mode="greedy")[2]["reward"]


def _heuristic_greedy_cost(env) -> float:
    """Cost of the nearest-feasible heuristic (Phase 4) on the env instance — the gap reference."""
    routes = greedy_routes(env=env)
    return -evaluate_solution(routes, env._inst, env._cost_cfg)["reward"]


class Trainer:
    def __init__(self, policy, cfg):
        self.policy, self.cfg = policy, cfg
        self.opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
        self.sampler = InstanceSampler(n_range=cfg.n_range)
        self.baseline = self._frozen_copy(policy)
        self.val_envs = [make_env(self.sampler.sample(s)) for s in cfg.val_seeds()]
        self.val_heur = float(
            np.mean([_heuristic_greedy_cost(e) for e in self.val_envs])
        )  # fixed reference

    @staticmethod
    def _frozen_copy(policy):
        b = copy.deepcopy(policy).eval()
        for p in b.parameters():
            p.requires_grad_(False)  # the baseline must not track training
        return b

    def train_batch(self, seeds) -> dict:
        envs = [make_env(self.sampler.sample(int(s))) for s in seeds]
        costs, logps, ents = [], [], []
        for env in envs:  # the current sampling policy (with gradient)
            _, slp, m, ent = self.policy.rollout(env, mode="sample", return_entropy=True)
            costs.append(-m["reward"])
            logps.append(slp)
            ents.append(ent)
        bl = torch.tensor(
            [_greedy_cost(self.baseline, e) for e in envs]
        )  # baseline greedy (paired)
        adv = torch.tensor(costs) - bl  # cost_sample − cost_baseline
        adv = ((adv - adv.mean()) / (adv.std() + 1e-8)).detach()  # normalise: bound ‖grad‖
        mean_ent = torch.stack(ents).mean()
        loss = (adv * torch.stack(logps)).mean()  # +Σlogπ → descent ↓cost
        if self.cfg.entropy_beta:  # reserve: −β·H maximises entropy (against softmax sharpening)
            loss = loss - self.cfg.entropy_beta * mean_ent
        self.opt.zero_grad()
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.grad_clip)
        self.opt.step()
        return {
            "cost": float(np.mean(costs)),
            "entropy": float(mean_ent.detach()),
            "grad_norm": float(gnorm),
        }

    def _maybe_update_baseline(self) -> dict:
        cur = np.array([_greedy_cost(self.policy, e) for e in self.val_envs])
        base = np.array([_greedy_cost(self.baseline, e) for e in self.val_envs])
        _, p = ttest_rel(cur, base)  # two-sided → use p/2 for the one-sided "better"
        updated = bool(
            cur.mean() < base.mean() and np.isfinite(p) and (p / 2) < self.cfg.baseline_p
        )
        if updated:
            self.baseline = self._frozen_copy(self.policy)
        return {
            "val_cost": float(cur.mean()),
            "val_base": float(base.mean()),
            "p": float(p),
            "gap_greedy": float(cur.mean() / self.val_heur - 1.0),
            "baseline_updated": updated,
        }

    def fit(self, log_fn=None) -> list[dict]:
        rng = np.random.default_rng(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)
        best, history, frozen = float("inf"), [], 0
        for epoch in range(self.cfg.epochs):
            ep = [
                self.train_batch(rng.integers(*self.cfg.train_range, size=self.cfg.batch))
                for _ in range(self.cfg.steps_per_epoch)
            ]
            rec = {
                "epoch": epoch,
                "train_cost": float(np.mean([e["cost"] for e in ep])),
                "entropy": float(np.mean([e["entropy"] for e in ep])),
                "grad_norm": float(np.mean([e["grad_norm"] for e in ep])),
                **self._maybe_update_baseline(),
            }
            history.append(rec)
            if rec["val_cost"] < best:
                best = rec["val_cost"]
                if self.cfg.ckpt:
                    Path(self.cfg.ckpt).parent.mkdir(parents=True, exist_ok=True)
                    torch.save(self.policy.state_dict(), self.cfg.ckpt)
            if log_fn:
                log_fn(rec)
            # runtime collapse guard: |g|≈0 = saturation (Phase 6). Abort at epoch ~3, NOT 100.
            frozen = frozen + 1 if rec["grad_norm"] < 1e-6 else 0
            if frozen >= _FREEZE_PATIENCE:
                raise RuntimeError(
                    f"training frozen: grad_norm≈0 for {frozen} epochs (saturation/collapse) "
                    f"— aborted at epoch {epoch} (not {self.cfg.epochs})"
                )
        return history
