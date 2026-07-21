"""Обучение RL (Phase 6). Smoke: python scripts/train.py --smoke  (2 эпохи, cost↓/энтропия).
Полный: python scripts/train.py [--epochs N]  → «Стало» на full-62 vs «Было» (baselines.json).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from logistics_rl_gnn.config.train import TrainConfig
from logistics_rl_gnn.env.vrp_env import VRPEnv
from logistics_rl_gnn.models.policy import VRPPolicy
from logistics_rl_gnn.train.reinforce import Trainer


def _log(rec: dict) -> None:
    print(
        f"epoch {rec['epoch']:3d} | train_cost {rec['train_cost']:7.1f} | "
        f"val {rec['val_cost']:7.1f} (gap_greedy {rec['gap_greedy']:+.1%}) | "
        f"H {rec['entropy']:.3f} | |g| {rec['grad_norm']:.2f} | "
        f"base{'↑' if rec['baseline_updated'] else ' '} (p={rec['p']:.3f})"
    )


def eval_full62(policy, seeds) -> tuple[float, list[float]]:
    """Политика (greedy) на ПОЛНЫХ инстансах (== «Было» Phase 4): валидный Было/Стало."""
    env = VRPEnv()
    costs = [-policy.rollout(env, mode="greedy", seed=int(s))[2]["reward"] for s in seeds]
    return float(np.mean(costs)), costs


def main() -> None:
    ap = argparse.ArgumentParser(description="RL-обучение CVRPTW (Phase 6)")
    ap.add_argument("--smoke", action="store_true", help="2 эпохи, малый batch (демо механизма)")
    ap.add_argument("--epochs", type=int, help="переопределить число эпох")
    args = ap.parse_args()

    cfg = TrainConfig.smoke() if args.smoke else TrainConfig()
    if args.epochs:
        cfg.epochs = args.epochs
    torch.manual_seed(cfg.seed)
    policy = VRPPolicy()
    trainer = Trainer(policy, cfg)

    print(
        f"{'SMOKE' if args.smoke else 'FULL'} | epochs={cfg.epochs} batch={cfg.batch} "
        f"steps/ep={cfg.steps_per_epoch} n_range={cfg.n_range} val_heur={trainer.val_heur:.1f}"
    )
    trainer.fit(log_fn=_log)

    # финал: «Стало» на full-62/seeds 0-9 vs «Было» (baselines.json)
    best = policy
    if cfg.ckpt and Path(cfg.ckpt).exists():
        best = VRPPolicy()
        best.load_state_dict(torch.load(cfg.ckpt, weights_only=True))  # только тензоры
    stalo, _ = eval_full62(best, cfg.eval_seeds)
    print("\n=== Стало vs Было (full-62, seeds 0–9, greedy-decode) ===")
    bl = Path("results/baselines.json")
    if bl.exists():
        ref = json.loads(bl.read_text())
        g = -ref["greedy"]["agg"]["reward"]["mean"]
        o = -ref["ortools"]["agg"]["reward"]["mean"]
        print(f"  RL «Стало» : {stalo:7.1f} €")
        print(f"  greedy      : {g:7.1f} €  (RL gap {stalo / g - 1:+.1%})")
        print(f"  OR-Tools    : {o:7.1f} €  (RL gap {stalo / o - 1:+.1%})")
    else:
        print(f"  RL «Стало» : {stalo:7.1f} €  (нет baselines.json — запусти run_baselines.py)")
    if args.smoke:
        print("  [smoke: короткий прогон — число иллюстративно; реальное «Стало» — полный прогон]")


if __name__ == "__main__":
    main()
