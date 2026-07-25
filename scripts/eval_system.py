"""Phase 8 — one seeded eval of the SYSTEM (polished portfolio) for the FULL metric vector.

The durable json ([[0009]] polish_summary) stored only the system COST; the before/after table also
needs distance/time/vehicles. We run the polished portfolio ONCE on seeds 0–9 (== baselines/0009,
full-62, free-flow) and write the full vector into results/system_metrics.json with provenance.

**PARITY GUARD:** the aggregate cost MUST match the durable 631.6€ (0009 port_pol) — otherwise this
is NOT that system (an assert error, not a silent drift). It does not replace decision 0009: it
reproduces its number and adds the metrics that were never logged. Deterministic (greedy/multistart/
sample_k seed=0; polish to convergence).
Run: python scripts/eval_system.py [--seeds N] [--budget-ms MS]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import run_dynamic as rd  # noqa: E402

from logistics_rl_gnn.baselines.greedy import greedy_routes  # noqa: E402
from logistics_rl_gnn.config import instance as im  # noqa: E402
from logistics_rl_gnn.env.events import make_dynamic_env  # noqa: E402
from logistics_rl_gnn.env.scoring import CostConfig, evaluate_solution  # noqa: E402
from logistics_rl_gnn.replan.local_search import polish  # noqa: E402
from logistics_rl_gnn.replan.portfolio import (  # noqa: E402
    candidate_rows,
    rl_candidate_mean,
    take_best,
)
from logistics_rl_gnn.train.pomo import multistart_greedy  # noqa: E402

_CKPT = Path("results/policy_pomo_congestion.pt")
_OUT = Path("results/system_metrics.json")
_CFG = CostConfig()
_FLEET = im.FLEET_SIZE
_Q = ("distance_km", "time_min", "vehicles_used", "on_time_pct", "unserved")
_DURABLE_COST_0009 = 631.6212305905019  # port_pol from polish_summary.json — the parity anchor


def _env(inst):
    return make_dynamic_env(inst, travel=None)  # K/Q from the instance (scenario → meta, else 8×80)


def system_routes(pol, inst, *, budget_ms, k_samples, temp, rl_starts, report=None):
    """Polished portfolio: best-by-cost of {greedy, RL-multi, sample-K}, each polished to the end.

    Identical to the port_pol selection in run_polish.static_polish (0009), but it returns the
    winner's ROUTES (for the full metric vector) rather than just the cost.

    pol=None → ablation `--no-model`: RL candidates are not produced (portfolio = greedy+polish).
    report (optional dict) is filled with the candidate table: rows/chosen/cost_model/cost_nomodel/
    rl_mean. Here EVERY candidate gets polish → "without the model" = polished greedy, an exact
    counterfactual of the same run (unlike re-plan, where polish only reaches the top-M)."""
    fleet, cap = im.fleet_of(inst)
    labels, cands = ["greedy"], [greedy_routes(env=_env(inst))]
    if pol is not None:
        _, rl = multistart_greedy(pol, _env(inst), rl_starts)  # None when no feasible start exists
        envs = [_env(inst) for _ in range(k_samples)]
        envs[0].reset(seed=0)
        sk = pol.sample_k(envs, pol.encode(envs[0]), temperature=temp, seed=0)
        sk_scores: list = []
        sk_best = take_best(sk, inst, None, _CFG, scores_out=sk_scores)[0]
        if rl is not None:
            labels.append("rl_greedy")
            cands.append(rl)
        labels.append("sample")
        cands.append(sk_best)
    polished = [polish(c, inst, None, budget_ms=budget_ms, fleet_size=fleet, vehicle_cap=cap)
                for c in cands]
    best_i = min(range(len(polished)), key=lambda i: polished[i][1])  # (routes, cost) → min cost
    if report is not None:
        raw = [(i, -evaluate_solution(c, inst, _CFG)["reward"]) for i, c in enumerate(cands)]
        rows = candidate_rows(labels, raw)
        for r in rows:  # here EVERY candidate gets polish (unlike re-plan)
            r["polished"] = polished[labels.index(r["source"])][1]
        if pol is not None:  # sample-K: n/mean over ALL K rollouts, not over the single winner
            for r in rows:
                if r["source"] == "sample":
                    r.update(n=len(sk_scores), cost=min(c for _, c in sk_scores),
                             mean=sum(c for _, c in sk_scores) / len(sk_scores))
        report.update(
            rows=rows, chosen=labels[best_i] + "+polish", cost_model=polished[best_i][1],
            cost_nomodel=polished[labels.index("greedy")][1],
            rl_mean=rl_candidate_mean(rows), used_model=pol is not None,
        )
    return polished[best_i][0]


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 8 — full system metric vector (0009 parity)")
    ap.add_argument("--ckpt", default=str(_CKPT))
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--budget-ms", type=float, default=30000.0, help="polish to convergence (0009)")
    ap.add_argument("--k-samples", type=int, default=128)
    ap.add_argument("--rl-starts", type=int, default=16)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--tol", type=float, default=2.0, help="parity tolerance to 631.6€ (wallclock)")
    ap.add_argument("--out", default=str(_OUT))
    args = ap.parse_args()

    ckpt = Path(args.ckpt)
    torch.manual_seed(0)
    pol = rd._load_policy(ckpt)
    acc = {k: [] for k in ("reward", *_Q)}
    for s in range(args.seeds):
        inst = im.generate_instance(seed=s)
        routes = system_routes(
            pol, inst, budget_ms=args.budget_ms, k_samples=args.k_samples,
            temp=args.temp, rl_starts=args.rl_starts,
        )
        q = evaluate_solution(routes, inst, _CFG)
        for k in acc:
            acc[k].append(float(q[k]))
        print(f"seed {s}: cost {-q['reward']:7.1f}€ dist {q['distance_km']:6.1f}km "
              f"time {q['time_min']:6.0f}min veh {int(q['vehicles_used'])}")
    mean = {k: float(np.mean(v)) for k, v in acc.items()}
    cost = -mean["reward"]
    # PARITY GUARD: cost must match the durable 631.6 (otherwise it is NOT that system)
    assert abs(cost - _DURABLE_COST_0009) < args.tol, (
        f"PARITY FAIL: cost {cost:.2f}€ != durable 0009 {_DURABLE_COST_0009:.2f}€ "
        f"(|Δ|={abs(cost - _DURABLE_COST_0009):.2f} > {args.tol}) — wrong system or budget-bound"
    )
    out = {
        "phase": "8-system-full-vector",
        "note": "polished portfolio (== [[0009]]), FULL vector; cost parity with 0009 631.6€. "
        "seeds 0-9 full-62 free-flow. One seeded eval (does not replace 0009, adds metrics).",
        "config": {
            "ckpt": str(ckpt), "seeds": list(range(args.seeds)), "budget_ms": args.budget_ms,
            "k_samples": args.k_samples, "rl_starts": args.rl_starts, "temperature": args.temp,
            "fleet_size": _FLEET,
        },
        "provenance": rd._provenance(ckpt),
        "durable_cost_anchor_0009": _DURABLE_COST_0009,
        # per-seed cost for a PAIRED comparison (the same seeds as OR-Tools in timematch → the
        # instance σ cancels; median-Δ + wins, as in 0010). mean(per_seed) == means.cost_eur.
        "per_seed_cost_eur": [float(-r) for r in acc["reward"]],
        "means": {
            "cost_eur": cost, "distance_km": mean["distance_km"], "time_min": mean["time_min"],
            "vehicles_used": mean["vehicles_used"], "on_time_pct": mean["on_time_pct"],
            "unserved": mean["unserved"],
        },
    }
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\nSYSTEM (polished portfolio, seeds 0-9): cost {cost:.1f}€ (0009 parity 631.6 ✓) "
          f"dist {mean['distance_km']:.1f}km time {mean['time_min']:.0f}min "
          f"veh {mean['vehicles_used']:.1f}")
    print(f"→ {args.out}")


if __name__ == "__main__":
    main()
