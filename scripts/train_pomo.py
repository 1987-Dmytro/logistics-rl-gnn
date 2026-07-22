"""POMO-обучение CVRPTW на статике (Phase 6b Шаг 1).

Smoke:  python scripts/train_pomo.py --smoke     (5 эпох, малый batch — cost↓/|g|>0/старты живы)
Полный: python scripts/train_pomo.py [--epochs N]  → «Стало» на full-62 (multi-start greedy) vs
        «Было» (baselines.json: greedy/OR-Tools). Веса лучшей по val → results/policy_pomo_best.pt.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from logistics_rl_gnn.config.pomo import POMOConfig
from logistics_rl_gnn.env.vrp_env import VRPEnv
from logistics_rl_gnn.models.policy import VRPPolicy
from logistics_rl_gnn.train.pomo import POMOTrainer, multistart_greedy


def _make_log(cfg: POMOConfig):
    """Лог-строка эпохи. train/val — оба gap-to-greedy (apples-to-apples); mem растёт = memorize;
    es = эпох без улучшения val / patience (видно приближение early-stop)."""

    def _log(rec: dict) -> None:
        gap_ort = f" OR {rec['gap_ort']:+.1%}" if "gap_ort" in rec else ""
        print(
            f"ep {rec['epoch']:3d} | train {rec['train_cost']:7.1f} (g{rec['train_gap']:+.1%}) | "
            f"val {rec['val_cost']:7.1f} (g{rec['gap_greedy']:+.1%}{gap_ort}) | "
            f"mem {rec['mem_gap']:+.1%} | H {rec['entropy']:.3f} | |g| {rec['grad_norm']:.2f} | "
            f"std {rec['start_std']:.1f} | es {rec['since_improve']}/{cfg.patience} | "
            f"fr {rec['inst_hash'][:6]}"  # freshness: меняется каждую эпоху (RNG свеж)
        )

    return _log


def _val_ortools_ref(cfg: POMOConfig) -> float:
    """OR-Tools-стоимость на ТЕХ ЖЕ val-инстансах (один раз) → референс gap в логе."""
    from logistics_rl_gnn.baselines.ortools_vrptw import ortools_routes
    from logistics_rl_gnn.env.scoring import CostConfig, evaluate_solution
    from logistics_rl_gnn.train.instance_sampler import InstanceSampler

    sampler = InstanceSampler(n_range=cfg.val_n_range or cfg.n_range)  # совпасть с eval_sampler
    costs = []
    for s in cfg.val_seeds():
        inst = sampler.sample(s)
        r = ortools_routes(inst, time_limit_s=10)
        costs.append(-evaluate_solution(r, inst, CostConfig())["reward"])
    return float(np.mean(costs))


def eval_full62(policy, seeds, max_starts: int) -> tuple[float, list[float]]:
    """«Стало» = multi-start greedy на ПОЛНЫХ инстансах (== «Было» Phase 4): валидный Было/Стало."""
    env = VRPEnv()
    costs = [multistart_greedy(policy, env, max_starts, reset_seed=int(s))[0] for s in seeds]
    return float(np.mean(costs)), costs


def main() -> None:
    ap = argparse.ArgumentParser(description="POMO-обучение CVRPTW (Phase 6b Шаг 1)")
    ap.add_argument("--smoke", action="store_true", help="5 эпох, малый batch (демо механизма)")
    ap.add_argument("--epochs", type=int, help="переопределить число эпох")
    ap.add_argument("--no-ort-ref", action="store_true", help="не считать OR-Tools val-референс")
    args = ap.parse_args()

    cfg = POMOConfig.smoke() if args.smoke else POMOConfig()
    if args.epochs:
        cfg.epochs = args.epochs
    torch.manual_seed(cfg.seed)
    policy = VRPPolicy()
    # OR-Tools-референс для лога: по умолчанию для full, для smoke пропускаем (скорость)
    val_ort = None if (args.smoke or args.no_ort_ref) else _val_ortools_ref(cfg)
    trainer = POMOTrainer(policy, cfg, val_ort=val_ort)

    ort_s = f" val_OR={val_ort:.1f}" if val_ort else ""
    print(
        f"{'SMOKE' if args.smoke else 'FULL'} POMO | epochs≤{cfg.epochs} patience={cfg.patience} "
        f"batch={cfg.batch} starts={cfg.max_starts} steps/ep={cfg.steps_per_epoch} n={cfg.n_range} "
        f"β={cfg.entropy_beta} val_heur={trainer.val_heur:.1f}{ort_s}"
    )
    hist = trainer.fit(log_fn=_make_log(cfg))
    sel = min(hist, key=lambda h: h["val_cost"])  # эпоха отбора (best-by-val)
    if len(hist) < cfg.epochs:
        print(f"[early-stop на эпохе {hist[-1]['epoch']} — {cfg.patience} эпох без улучшения val]")

    # финал: лучшая по val → TEST (held-out) + «Стало» на full-62 vs «Было» (baselines.json).
    best = policy
    if cfg.ckpt and Path(cfg.ckpt).exists():
        best = VRPPolicy()
        best.load_state_dict(torch.load(cfg.ckpt, weights_only=True))
    test = trainer.test_eval(best)
    stalo, _ = eval_full62(best, cfg.eval_seeds, cfg.max_starts)

    print("\n=== Обобщение (gap-to-greedy; val≈test≈deployment → обобщает, расход → memorize) ===")
    print(f"  train (probe, эпоха отбора {sel['epoch']}): {sel['train_gap']:+.1%}")
    print(f"  val   (отбор):                    {sel['gap_greedy']:+.1%}")
    print(f"  TEST  (held-out):                 {test['test_gap_greedy']:+.1%}")
    print("\n=== Стало vs Было (full-62, seeds 0–9, multi-start greedy) ===")
    bl = Path("results/baselines.json")
    if bl.exists():
        ref = json.loads(bl.read_text())
        g = -ref["greedy"]["agg"]["reward"]["mean"]
        o = -ref["ortools"]["agg"]["reward"]["mean"]
        print(f"  POMO «Стало»: {stalo:7.1f} €")
        print(f"  greedy      : {g:7.1f} €  (POMO gap {stalo / g - 1:+.1%}  ← deployment)")
        print(f"  OR-Tools    : {o:7.1f} €  (POMO gap {stalo / o - 1:+.1%})")
    else:
        print(f"  POMO «Стало»: {stalo:7.1f} €  (нет baselines.json — запусти run_baselines.py)")
    if args.smoke:
        print("  [smoke: короткий прогон — число иллюстративно; реальное «Стало» — полный прогон]")


if __name__ == "__main__":
    main()
