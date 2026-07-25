"""Task #15 — time-matched comparison: the OR-Tools anytime curve vs the deployed system.

The final benchmark. We give OR-Tools THE SAME wall-clock and measure quality on IDENTICAL
instances (full-62, seeds 0–9, the same `generate_instance` + the single `evaluate_solution`, #3).
Budgets {0.7,2,5,30}s → the curve cost(budget). PARITY GUARD: the 30s point == 611.1€ (0002).

HONESTY (Phase 8 conflation quarantine): the system reaches its static 631.6€ with a polish budget
of 30000ms per candidate ×≤3 (`system_metrics.json`) → its static wall-clock is ≥30s, NOT 689ms.
689ms is the DYNAMIC re-plan latency on a residual (cost 827€ there, `polish_summary.json`) — a
different setting. The system is placed at its REAL static x (≥30s, y=631.6€); 689ms/827€ goes in
a separate field as dynamics. The crossover is computed from the data, not pre-judged.

Writes ONLY results/timematch.json. The "Time-matched" section of docs/final_metrics.md is emitted
by final_metrics.py (one owner per file — otherwise overwrite/duplication). Run:
    python scripts/run_timematch.py [--seeds 10] [--budgets 0.7 2 5 30]
"""

from __future__ import annotations

import argparse
import json
import platform
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import numpy as np

from logistics_rl_gnn.baselines import ortools_vrptw
from logistics_rl_gnn.config import instance as im
from logistics_rl_gnn.env.scoring import CostConfig, evaluate_solution

_RES = Path("results")
_BUDGETS = [0.7, 2.0, 5.0, 30.0]
_ANCHOR_0002_ORTOOLS = 611.14  # the 30s point (decision 0002) — the parity guard
# wall-clock tolerance: time-limited OR-Tools = best-so-far (not bit-det.), as eval_system tol=2.0.
# Catches a DRIFT of path/instances (tens of €), not GLS convergence jitter (<2€).
_PARITY_TOL = 2.0


def _sys_ref() -> dict:
    """The system's honest point: static quality 631.6€ at its REAL static x (polish budget ≥30s).

    689ms/827€ is kept SEPARATE as the dynamic re-plan latency (a different setting, not the static
    wall-clock for 631.6€). Numbers come from the durable json (parity), else decision anchors."""
    smp, psp = _RES / "system_metrics.json", _RES / "polish_summary.json"
    cost = 631.62
    static_budget_ms = 30000.0
    dyn_lat_ms, dyn_cost = 688.7, 827.3
    if smp.exists():
        sm = json.loads(smp.read_text())
        cost = sm["means"]["cost_eur"]
        static_budget_ms = sm["config"]["budget_ms"]
    if psp.exists():
        dyn = json.loads(psp.read_text())["dynamic"]["aggregates"]["rl"]
        dyn_lat_ms, dyn_cost = dyn["latency_ms_median"], dyn["cost_eur_mean"]
    return {
        "cost_eur": cost,
        "static_wallclock_s": static_budget_ms / 1000.0,  # LOWER bound: ×≤3 candidates + decode
        "static_wallclock_note": f"polish {static_budget_ms:.0f}ms/candidate ×≤3 + decode → ≥"
        f"{static_budget_ms / 1000.0:.0f}s (candidates run in sequence; true wall-clock is higher)",
        "dynamic_replan_latency_ms": dyn_lat_ms,  # QUARANTINE: other setting (residual), not 631.6€
        "dynamic_replan_cost_eur": dyn_cost,
        "conflation_avoided": "631.6€ is statics (≥30s polish), NOT '@689ms'; 689ms is dynamics "
        "on a residual (cost 827€). Different settings — not one point.",
    }


def run(seeds: list[int], budgets: list[float]) -> dict:
    cfg = CostConfig()
    per_seed = {b: [] for b in budgets}
    for s in seeds:
        inst = im.generate_instance(seed=s)  # the same path as run_baselines (parity 30s=611.1)
        for b in budgets:  # one instance → all budgets (the same instance, prohibition #3)
            routes = ortools_vrptw.ortools_routes(inst, cfg, time_limit_s=b)
            per_seed[b].append(-evaluate_solution(routes, inst, cfg)["reward"])  # cost €

    curve = []
    for b in budgets:
        costs = per_seed[b]
        curve.append({
            "budget_s": b,
            "cost_mean": float(np.mean(costs)),
            "cost_std": float(np.std(costs)),
            "per_seed": [float(c) for c in costs],
        })
    or_best = curve[-1]["cost_mean"]  # 30s — the strongest OR point
    sys_ref = _sys_ref()

    # crossover: smallest budget where OR-Tools ≤ the system's static quality (631.6€), from data
    xover = next((c["budget_s"] for c in curve if c["cost_mean"] <= sys_ref["cost_eur"]), None)

    # PARITY GUARD: the 30s point == 611.1€ (0002)
    parity_ok = abs(or_best - _ANCHOR_0002_ORTOOLS) < _PARITY_TOL
    assert parity_ok, (
        f"PARITY FAIL: OR-Tools@30s {or_best:.2f}€ != anchor 0002 {_ANCHOR_0002_ORTOOLS}€ "
        f"(|Δ|={abs(or_best - _ANCHOR_0002_ORTOOLS):.2f} > {_PARITY_TOL}) — the path drifted"
    )
    return {
        "phase": "15-time-matched",
        "note": "OR-Tools anytime curve (full-62, seeds 0–9, single scorer) vs the system's static "
        "quality 631.6€. OR-Tools gets the same wall-clock. cost = −reward, €. The system's point "
        "sits at its REAL static x (≥30s polish), NOT at 689ms (dynamics, field dynamic_replan_*).",
        "config": {
            "fleet_size": im.FLEET_SIZE,
            "vehicle_cap": im.VEHICLE_CAP,
            "t_max_min": im.T_MAX_MIN,
            "costs": asdict(cfg),
            "delivery_weekday": im.DELIVERY_WEEKDAY,
            "snapshot": im._latest_snapshot_dir().name,
        },
        "run": {
            "seeds": seeds,
            "budgets_s": budgets,
            "ortools_version": version("ortools"),
            "numpy_version": version("numpy"),
            "python": platform.python_version(),
        },
        "curve": curve,
        "ortools_best_30s_eur": or_best,
        "system_ref": sys_ref,
        "crossover_budget_s": xover,  # None → OR-Tools never reached the system's quality in 30s
        "parity": {"anchor_0002_ortools": _ANCHOR_0002_ORTOOLS, "measured_30s": or_best,
                   "ok": parity_ok},
    }


def print_report(res: dict) -> None:
    c = res["config"]
    sr = res["system_ref"]
    print(f"\nTime-matched — OR-Tools vs the system | K={c['fleet_size']} Q={c['vehicle_cap']} "
          f"| seeds={res['run']['seeds']} | OR-Tools {res['run']['ortools_version']}")
    print(f"{'budget':>10}  {'cost,€ (mean±std)':>22}  {'vs system 631.6':>16}  {'vs OR@30s':>10}")
    print("-" * 66)
    or_best = res["ortools_best_30s_eur"]
    for pt in res["curve"]:
        m, sd = pt["cost_mean"], pt["cost_std"]
        vs_sys = m - sr["cost_eur"]
        vs_or = m - or_best
        print(f"{pt['budget_s']:>9.1f}s  {m:>10.1f} ± {sd:6.1f}       "
              f"{vs_sys:>+8.1f}€        {vs_or:>+7.1f}€")
    print("-" * 66)
    xo = res["crossover_budget_s"]
    xo_s = f"{xo:.1f}s" if xo is not None else ">30s (not reached within 30s)"
    print(f"System: {sr['cost_eur']:.1f}€ at wall-clock ≥{sr['static_wallclock_s']:.0f}s "
          f"({sr['static_wallclock_note']}).")
    print(f"OR-Tools reaches ≤ 631.6€ (the system's quality) at budget: {xo_s}.")
    print(f"QUARANTINE: 689ms is the dynamic re-plan latency (residual, cost "
          f"{sr['dynamic_replan_cost_eur']:.0f}€), NOT the static wall-clock for 631.6€.")
    print(f"\nVERDICT: time-matched in statics OR-Tools@30s = {or_best:.1f}€ "
          f"{'BEATS' if or_best < sr['cost_eur'] else 'does NOT beat'} the system "
          f"{sr['cost_eur']:.1f}€ "
          f"({or_best - sr['cost_eur']:+.1f}€); the system's edge is dynamics only, not statics.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Task #15 — time-matched OR-Tools vs the system")
    ap.add_argument("--seeds", type=int, default=10, help="number of seeds (0..N-1)")
    ap.add_argument("--budgets", type=float, nargs="+", default=_BUDGETS, help="budgets, s")
    ap.add_argument("--out", type=Path, default=_RES / "timematch.json")
    args = ap.parse_args()
    if args.seeds < 1:
        ap.error("--seeds must be >= 1")

    res = run(list(range(args.seeds)), sorted(args.budgets))
    print_report(res)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"\nsummary → {args.out}  (the final_metrics.md section is emitted by final_metrics.py)")


if __name__ == "__main__":
    main()
