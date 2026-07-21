"""REINFORCE с rollout-baseline (Kool, Phase 6).

baseline = замороженная копия политики (greedy на ТЕХ ЖЕ инстансах). advantage = стоимость
sample-текущей − greedy-baseline (paired). loss = mean(advantage · Σlogπ) — знак +Σlogπ (вывод:
∇L=E[(cost−b)∇Σlogπ]=∇E[cost], градиентный спуск ↓cost; спек-формула −Σlogπ дала бы ↑cost).
Каждую эпоху paired t-test greedy-current vs greedy-baseline на val → значимо лучше ⇒ апдейт.
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


def make_env(instance) -> VRPEnv:
    """VRPEnv на фиксированном инстансе (instance_fn игнорит seed → reset даёт тот же инстанс)."""
    return VRPEnv(instance_fn=lambda s: instance)


def _greedy_cost(policy, env) -> float:
    with torch.no_grad():
        return -policy.rollout(env, mode="greedy")[2]["reward"]


def _heuristic_greedy_cost(env) -> float:
    """Стоимость эвристики nearest-feasible (Phase 4) на инстансе env — референс gap."""
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
        )  # фикс. референс

    @staticmethod
    def _frozen_copy(policy):
        b = copy.deepcopy(policy).eval()
        for p in b.parameters():
            p.requires_grad_(False)  # baseline не должен отслеживать обучение
        return b

    def train_batch(self, seeds) -> dict:
        envs = [make_env(self.sampler.sample(int(s))) for s in seeds]
        costs, logps, ents = [], [], []
        for env in envs:  # sample-текущая политика (с градиентом)
            _, slp, m, ent = self.policy.rollout(env, mode="sample", return_entropy=True)
            costs.append(-m["reward"])
            logps.append(slp)
            ents.append(ent)
        bl = torch.tensor(
            [_greedy_cost(self.baseline, e) for e in envs]
        )  # baseline greedy (paired)
        adv = (torch.tensor(costs) - bl).detach()  # cost_sample − cost_baseline
        loss = (adv * torch.stack(logps)).mean()  # +Σlogπ → спуск ↓cost
        self.opt.zero_grad()
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.grad_clip)
        self.opt.step()
        return {
            "cost": float(np.mean(costs)),
            "entropy": float(np.mean(ents)),
            "grad_norm": float(gnorm),
        }

    def _maybe_update_baseline(self) -> dict:
        cur = np.array([_greedy_cost(self.policy, e) for e in self.val_envs])
        base = np.array([_greedy_cost(self.baseline, e) for e in self.val_envs])
        _, p = ttest_rel(cur, base)  # двусторонний → делим p/2 для one-sided «лучше»
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
        best, history = float("inf"), []
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
        return history
