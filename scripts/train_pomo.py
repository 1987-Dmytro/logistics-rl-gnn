"""POMO training for CVRPTW on statics (Phase 6b Step 1).

Smoke:  python scripts/train_pomo.py --smoke     (5 epochs, small batch — cost↓/|g|>0/starts alive)
Full: python scripts/train_pomo.py [--epochs N]  → "after" on full-62 (multi-start greedy) vs
        "before" (baselines.json: greedy/OR-Tools). Best-by-val → results/policy_pomo_best.pt.
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
    """Epoch log line. train/val — both gap-to-greedy (apples-to-apples); mem grows = memorising;
    es = epochs without val improvement / patience (early-stop approaching is visible)."""

    def _log(rec: dict) -> None:
        gap_ort = f" OR {rec['gap_ort']:+.1%}" if "gap_ort" in rec else ""
        print(
            f"ep {rec['epoch']:3d} | train {rec['train_cost']:7.1f} (g{rec['train_gap']:+.1%}) | "
            f"val {rec['val_cost']:7.1f} (g{rec['gap_greedy']:+.1%}{gap_ort}) | "
            f"mem {rec['mem_gap']:+.1%} | H {rec['entropy']:.3f} | |g| {rec['grad_norm']:.2f} | "
            f"std {rec['start_std']:.1f} | es {rec['since_improve']}/{cfg.patience} | "
            f"fr {rec['inst_hash'][:6]}"  # freshness: changes every epoch (RNG is fresh)
        )

    return _log


def _val_ortools_ref(cfg: POMOConfig) -> float:
    """OR-Tools cost on THE SAME val instances (once) → the gap reference in the log."""
    from logistics_rl_gnn.baselines.ortools_vrptw import ortools_routes
    from logistics_rl_gnn.env.scoring import CostConfig, evaluate_solution
    from logistics_rl_gnn.train.instance_sampler import InstanceSampler

    sampler = InstanceSampler(n_range=cfg.val_n_range or cfg.n_range)  # match eval_sampler
    costs = []
    for s in cfg.val_seeds():
        inst = sampler.sample(s)
        r = ortools_routes(inst, time_limit_s=10)
        costs.append(-evaluate_solution(r, inst, CostConfig())["reward"])
    return float(np.mean(costs))


def eval_full62(policy, seeds, max_starts: int) -> tuple[float, list[float]]:
    """"after" = multi-start greedy on FULL instances (== Phase 4 "before"): valid before/after."""
    env = VRPEnv()
    costs = [multistart_greedy(policy, env, max_starts, reset_seed=int(s))[0] for s in seeds]
    return float(np.mean(costs)), costs


def eval_congestion(policy_cong, policy_ff, cfg, seeds, *, deadline_s=10, with_ort=True) -> dict:
    """Static congestion before/after (Step 2, prohibition #3): on IDENTICAL congestion instances —
    congestion-RL vs free-flow-RL(best.pt) vs greedy vs OR-Tools(snapshot); a single scorer under
    congestion. OOD gate: congestion-RL must beat free-flow-RL (the headline, not the gap to OR)."""
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
        travel = sample_congestion_travel(inst, int(s), cfg)  # THE SAME congestion as train-eval
        env = make_dynamic_env(inst, travel=travel)
        acc["rl_cong"].append(multistart_greedy(policy_cong, env, cfg.max_starts)[0])
        acc["rl_ff"].append(multistart_greedy(policy_ff, env, cfg.max_starts)[0])
        gr = greedy_routes(env=make_dynamic_env(inst, travel=travel))
        acc["greedy"].append(-evaluate_solution(gr, inst, ccfg, travel=travel)["reward"])
        if with_ort:  # OR-Tools on a static congestion snapshot (pessimistic — see 0004)
            snap = replace(inst, time_matrix=_snapshot_matrix_s(travel, len(inst.demand)))
            orr = ortools_routes(snap, fleet_size=im.FLEET_SIZE, time_limit_s=deadline_s)
            acc["ortools"].append(-evaluate_solution(orr, inst, ccfg, travel=travel)["reward"])
    return {m: float(np.mean(v)) for m, v in acc.items() if v}


def _report_static(best, cfg, smoke: bool) -> None:
    stalo, _ = eval_full62(best, cfg.eval_seeds, cfg.max_starts)
    print("\n=== After vs Before (full-62, seeds 0–9, multi-start greedy) ===")
    bl = Path("results/baselines.json")
    if bl.exists():
        ref = json.loads(bl.read_text())
        g = -ref["greedy"]["agg"]["reward"]["mean"]
        o = -ref["ortools"]["agg"]["reward"]["mean"]
        print(f"  POMO 'after': {stalo:7.1f} €")
        print(f"  greedy      : {g:7.1f} €  (POMO gap {stalo / g - 1:+.1%}  ← deployment)")
        print(f"  OR-Tools    : {o:7.1f} €  (POMO gap {stalo / o - 1:+.1%})")
    else:
        print(f"  POMO 'after': {stalo:7.1f} €  (no baselines.json — run run_baselines.py)")
    if smoke:
        print("  [smoke: the number is illustrative; a real 'after' needs a full run]")


def _report_congestion(best, cfg, smoke: bool, baseline) -> None:
    """Before/after under congestion: headline = RL-congestion vs RL-free-flow (beats OOD?)."""
    ff = VRPPolicy()  # free-flow best.pt (warm start) for RL-vs-RL
    if cfg.warm_start and Path(cfg.warm_start).exists():
        ff.load_state_dict(torch.load(cfg.warm_start, weights_only=True))
    seeds = list(cfg.test_seeds())[: (4 if smoke else 32)]  # held-out
    r = eval_congestion(best, ff, cfg, seeds, deadline_s=(2 if smoke else 10), with_ort=not smoke)
    g_ff = r["rl_cong"] / r["rl_ff"] - 1
    g_gr = r["rl_cong"] / r["greedy"] - 1
    print("\n=== Before/after under congestion (held-out test, single scorer) ===")
    print(f"  RL-congestion : {r['rl_cong']:7.1f} €  ← after")
    print(f"  RL-free-flow  : {r['rl_ff']:7.1f} €  (gap {g_ff:+.1%}  ← headline: does it beat OOD)")
    print(f"  greedy        : {r['greedy']:7.1f} €  ('after' gap {g_gr:+.1%})")
    if "ortools" in r:
        go = r["rl_cong"] / r["ortools"] - 1
        print(f"  OR-Tools(snap): {r['ortools']:7.1f} €  (gap {go:+.1%}; snapshot pessimism 0004)")
    if smoke:
        print("  [smoke: illustrative number; OR-Tools skipped]")


def main() -> None:
    ap = argparse.ArgumentParser(description="POMO training for CVRPTW (Phase 6b Step 1/Step 2)")
    ap.add_argument("--smoke", action="store_true", help="small run (mechanism demo)")
    ap.add_argument("--epochs", type=int, help="override the number of epochs")
    ap.add_argument("--no-ort-ref", action="store_true", help="skip the OR-Tools val reference")
    ap.add_argument("--congestion", action="store_true", help="Step 2: training under congestion")
    ap.add_argument("--warm-start", type=str, default=None, help="path to warm-start weights")
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

    # val OR reference in the log: free-flow OR is meaningless under congestion → skipped there
    val_ort = (
        None if (args.smoke or args.no_ort_ref or cfg.congestion) else _val_ortools_ref(cfg)
    )
    trainer = POMOTrainer(policy, cfg, val_ort=val_ort)

    baseline = None
    if cfg.congestion:  # advisor gates: coverage + the free-flow-best bar under congestion first
        cov = congestion_coverage(trainer.probe_envs)
        print(
            f"COVERAGE probe: inc_node {cov['inc_node_cov']:.0%} "
            f"inc_edge {cov['inc_edge_cov']:.0%} mean edges {cov['mean_inc_edge_frac']:.1%}"
        )
        baseline = trainer._validate()  # fit() trains BEFORE the first validation → take the bar
        print(
            f"BAR (free-flow-best under congestion): val {baseline['val_cost']:.1f} "
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
    sel = min(hist, key=lambda h: h["val_cost"])  # the selection epoch (best-by-val)
    if len(hist) < cfg.epochs:
        print(f"[early-stop at epoch {hist[-1]['epoch']} — {cfg.patience} epochs without val gain]")

    best = policy  # best by val
    if cfg.ckpt and Path(cfg.ckpt).exists():
        best = VRPPolicy()
        best.load_state_dict(torch.load(cfg.ckpt, weights_only=True))

    print("\n=== Generalisation (gap-to-greedy; val≈test → generalises, drift → memorising) ===")
    print(f"  train (probe, selection epoch {sel['epoch']}): {sel['train_gap']:+.1%}")
    print(f"  val   (selection):                    {sel['gap_greedy']:+.1%}")
    print(f"  TEST  (held-out):                 {trainer.test_eval(best)['test_gap_greedy']:+.1%}")

    if cfg.congestion:
        _report_congestion(best, cfg, args.smoke, baseline)
    else:
        _report_static(best, cfg, args.smoke)


if __name__ == "__main__":
    main()
