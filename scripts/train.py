"""RL training (Phase 6). Smoke: python scripts/train.py --smoke  (2 epochs, cost↓/entropy).
Full: python scripts/train.py [--epochs N]  → "after" on full-62 vs "before" (baselines.json).
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
    """Policy (greedy) on FULL instances (== the Phase 4 "before"): a valid before/after."""
    env = VRPEnv()
    costs = [-policy.rollout(env, mode="greedy", seed=int(s))[2]["reward"] for s in seeds]
    return float(np.mean(costs)), costs


def main() -> None:
    ap = argparse.ArgumentParser(description="RL training for CVRPTW (Phase 6)")
    ap.add_argument("--smoke", action="store_true", help="2 epochs, small batch (mechanism demo)")
    ap.add_argument("--epochs", type=int, help="override the number of epochs")
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

    # final: "after" on full-62/seeds 0-9 vs "before" (baselines.json)
    best = policy
    if cfg.ckpt and Path(cfg.ckpt).exists():
        best = VRPPolicy()
        best.load_state_dict(torch.load(cfg.ckpt, weights_only=True))  # tensors only
    stalo, _ = eval_full62(best, cfg.eval_seeds)
    print("\n=== After vs Before (full-62, seeds 0–9, greedy decode) ===")
    bl = Path("results/baselines.json")
    if bl.exists():
        ref = json.loads(bl.read_text())
        g = -ref["greedy"]["agg"]["reward"]["mean"]
        o = -ref["ortools"]["agg"]["reward"]["mean"]
        print(f"  RL 'after' : {stalo:7.1f} €")
        print(f"  greedy      : {g:7.1f} €  (RL gap {stalo / g - 1:+.1%})")
        print(f"  OR-Tools    : {o:7.1f} €  (RL gap {stalo / o - 1:+.1%})")
    else:
        print(f"  RL 'after' : {stalo:7.1f} €  (no baselines.json — run run_baselines.py)")
    if args.smoke:
        print("  [smoke: short run — the number is illustrative; a real 'after' needs a full run]")


if __name__ == "__main__":
    main()
