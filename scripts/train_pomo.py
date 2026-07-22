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
from logistics_rl_gnn.train.pomo import (
    POMOTrainer,
    congestion_coverage,
    multistart_greedy,
)


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


def eval_congestion(policy_cong, policy_ff, cfg, seeds, *, deadline_s=10, with_ort=True) -> dict:
    """Static-congestion Было/Стало (Шаг 2, запрет №3): на ИДЕНТИЧНЫХ congestion-инстансах —
    congestion-RL vs free-flow-RL(best.pt) vs greedy vs OR-Tools(snapshot); единый scorer под
    congestion. Гейт OOD: congestion-RL должен бить free-flow-RL (headline, не gap-к-OR)."""
    from dataclasses import replace

    from logistics_rl_gnn.baselines.greedy import greedy_routes
    from logistics_rl_gnn.baselines.ortools_vrptw import ortools_routes
    from logistics_rl_gnn.config import instance as im
    from logistics_rl_gnn.env.events import make_dynamic_env
    from logistics_rl_gnn.env.scoring import CostConfig, evaluate_solution
    from logistics_rl_gnn.replan.compare import _snapshot_matrix_s
    from logistics_rl_gnn.train.instance_sampler import InstanceSampler
    from logistics_rl_gnn.train.pomo import sample_congestion_travel

    sampler = InstanceSampler(n_range=cfg.val_n_range or cfg.n_range)
    ccfg = CostConfig()
    acc: dict = {m: [] for m in ("rl_cong", "rl_ff", "greedy", "ortools")}
    for s in seeds:
        inst = sampler.sample(int(s))
        travel = sample_congestion_travel(inst, int(s), cfg)  # ТОТ ЖЕ congestion, что train-eval
        env = make_dynamic_env(inst, travel=travel)
        acc["rl_cong"].append(multistart_greedy(policy_cong, env, cfg.max_starts)[0])
        acc["rl_ff"].append(multistart_greedy(policy_ff, env, cfg.max_starts)[0])
        gr = greedy_routes(env=make_dynamic_env(inst, travel=travel))
        acc["greedy"].append(-evaluate_solution(gr, inst, ccfg, travel=travel)["reward"])
        if with_ort:  # OR-Tools на static-снимке congestion (пессимистичен — см. 0004)
            snap = replace(inst, time_matrix=_snapshot_matrix_s(travel, len(inst.demand)))
            orr = ortools_routes(snap, fleet_size=im.FLEET_SIZE, time_limit_s=deadline_s)
            acc["ortools"].append(-evaluate_solution(orr, inst, ccfg, travel=travel)["reward"])
    return {m: float(np.mean(v)) for m, v in acc.items() if v}


def _report_static(best, cfg, smoke: bool) -> None:
    stalo, _ = eval_full62(best, cfg.eval_seeds, cfg.max_starts)
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
    if smoke:
        print("  [smoke: число иллюстративно; реальное «Стало» — полный прогон]")


def _report_congestion(best, cfg, smoke: bool, baseline) -> None:
    """Было/Стало под congestion: headline = RL-congestion vs RL-free-flow (бьёт ли OOD)."""
    ff = VRPPolicy()  # free-flow best.pt (warm-start) для RL-vs-RL
    if cfg.warm_start and Path(cfg.warm_start).exists():
        ff.load_state_dict(torch.load(cfg.warm_start, weights_only=True))
    seeds = list(cfg.test_seeds())[: (4 if smoke else 32)]  # held-out
    r = eval_congestion(best, ff, cfg, seeds, deadline_s=(2 if smoke else 10), with_ort=not smoke)
    g_ff = r["rl_cong"] / r["rl_ff"] - 1
    g_gr = r["rl_cong"] / r["greedy"] - 1
    print("\n=== Было/Стало под congestion (held-out test, единый scorer) ===")
    print(f"  RL-congestion : {r['rl_cong']:7.1f} €  ← Стало")
    print(f"  RL-free-flow  : {r['rl_ff']:7.1f} €  (gap {g_ff:+.1%}  ← headline: бьёт ли OOD)")
    print(f"  greedy        : {r['greedy']:7.1f} €  (Стало gap {g_gr:+.1%})")
    if "ortools" in r:
        go = r["rl_cong"] / r["ortools"] - 1
        print(f"  OR-Tools(snap): {r['ortools']:7.1f} €  (gap {go:+.1%}; snapshot-пессимизм 0004)")
    if smoke:
        print("  [smoke: число иллюстративно; OR-Tools пропущен]")


def main() -> None:
    ap = argparse.ArgumentParser(description="POMO-обучение CVRPTW (Phase 6b Шаг 1/Шаг 2)")
    ap.add_argument("--smoke", action="store_true", help="малый прогон (демо механизма)")
    ap.add_argument("--epochs", type=int, help="переопределить число эпох")
    ap.add_argument("--no-ort-ref", action="store_true", help="не считать OR-Tools val-референс")
    ap.add_argument("--congestion", action="store_true", help="Шаг 2: обучение под congestion")
    ap.add_argument("--warm-start", type=str, default=None, help="путь к весам тёплого старта")
    args = ap.parse_args()

    if args.congestion:
        over = (
            dict(
                epochs=5, steps_per_epoch=8, batch=4, max_starts=5, patience=3,
                n_range=(15, 20), val_range=(1_000_000, 1_000_004),
                test_range=(2_000_000, 2_000_004), ckpt=None,
            )
            if args.smoke
            else {}
        )  # fmt: skip
        cfg = POMOConfig.for_congestion(**over)
    else:
        cfg = POMOConfig.smoke() if args.smoke else POMOConfig()
    if args.epochs:
        cfg.epochs = args.epochs
    if args.warm_start:
        cfg.warm_start = args.warm_start

    torch.manual_seed(cfg.seed)
    policy = VRPPolicy()
    if cfg.warm_start and Path(cfg.warm_start).exists():
        policy.load_state_dict(torch.load(cfg.warm_start, weights_only=True))
        print(f"warm-start: {cfg.warm_start}")

    # val_OR-референс в логе: free-flow OR не осмыслен под congestion → там пропускаем
    val_ort = (
        None if (args.smoke or args.no_ort_ref or cfg.congestion) else _val_ortools_ref(cfg)
    )
    trainer = POMOTrainer(policy, cfg, val_ort=val_ort)

    baseline = None
    if cfg.congestion:  # advisor-гейты: coverage + планка free-flow-best под congestion ДО обучения
        cov = congestion_coverage(trainer.probe_envs)
        print(
            f"COVERAGE probe: inc_node {cov['inc_node_cov']:.0%} "
            f"inc_edge {cov['inc_edge_cov']:.0%} ср.рёбер {cov['mean_inc_edge_frac']:.1%}"
        )
        baseline = trainer._validate()  # fit() учит ДО первой валидации → планку берём явно
        print(
            f"ПЛАНКА (free-flow-best под congestion): val {baseline['val_cost']:.1f} "
            f"gap_greedy {baseline['gap_greedy']:+.1%}"
        )

    ort_s = f" val_OR={val_ort:.1f}" if val_ort else ""
    tag = " [congestion]" if cfg.congestion else ""
    print(
        f"{'SMOKE' if args.smoke else 'FULL'} POMO{tag} | epochs≤{cfg.epochs} "
        f"patience={cfg.patience} batch={cfg.batch} starts={cfg.max_starts} "
        f"steps/ep={cfg.steps_per_epoch} n={cfg.n_range} β={cfg.entropy_beta} lr={cfg.lr:.0e} "
        f"val_heur={trainer.val_heur:.1f}{ort_s}"
    )
    hist = trainer.fit(log_fn=_make_log(cfg))
    sel = min(hist, key=lambda h: h["val_cost"])  # эпоха отбора (best-by-val)
    if len(hist) < cfg.epochs:
        print(f"[early-stop на эпохе {hist[-1]['epoch']} — {cfg.patience} эпох без улучшения val]")

    best = policy  # лучшая по val
    if cfg.ckpt and Path(cfg.ckpt).exists():
        best = VRPPolicy()
        best.load_state_dict(torch.load(cfg.ckpt, weights_only=True))

    print("\n=== Обобщение (gap-to-greedy; val≈test → обобщает, расход → memorize) ===")
    print(f"  train (probe, эпоха отбора {sel['epoch']}): {sel['train_gap']:+.1%}")
    print(f"  val   (отбор):                    {sel['gap_greedy']:+.1%}")
    print(f"  TEST  (held-out):                 {trainer.test_eval(best)['test_gap_greedy']:+.1%}")

    if cfg.congestion:
        _report_congestion(best, cfg, args.smoke, baseline)
    else:
        _report_static(best, cfg, args.smoke)


if __name__ == "__main__":
    main()
