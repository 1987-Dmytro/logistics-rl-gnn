"""Phase 6b Step 3.5 — local-search polish (NO training). Checkpoint: congestion-best.

Measurements (the same protocol as Step 3; provenance → results/polish_summary.json):
  a) static full-62 / seeds 0–9 / free-flow: polish EVERY candidate TO CONVERGENCE (a generous
     budget — otherwise "polish gives little" is a budget artefact) → decomposition of the polish
     contribution over greedy vs RL; polished portfolio vs Step-3 766.1€ vs OR-Tools 611€ → new gap;
  b) dynamic 0004 with polish inside the SHARED budget (in-budget, not convergence) → table+latency.

Run: python scripts/run_polish.py [--ckpt P] [--seeds N] [--dyn-seeds N]
        [--static-budget-ms MS] [--polish-budget-ms MS] [--k-dyn K] [--out P]
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
from logistics_rl_gnn.replan.portfolio import PortfolioPlanner, take_best  # noqa: E402
from logistics_rl_gnn.train.pomo import multistart_greedy  # noqa: E402

_CKPT = Path("results/policy_pomo_congestion.pt")
_OUT = Path("results/polish_summary.json")
_CFG = CostConfig()
_FLEET = im.FLEET_SIZE


def _cost(r, inst):
    return -evaluate_solution(r, inst, _CFG)["reward"]


def _pol(r, inst, budget_ms):
    return polish(r, inst, None, budget_ms=budget_ms, fleet_size=_FLEET)[1]


def static_polish(pol, seeds, *, budget_ms, k_samples, temp, rl_starts) -> dict:
    bl = json.loads(Path("results/baselines.json").read_text())
    g_ref = -bl["greedy"]["agg"]["reward"]["mean"]
    o_ref = -bl["ortools"]["agg"]["reward"]["mean"]
    acc = {k: [] for k in ("g_raw", "g_pol", "r_raw", "r_pol", "s_raw", "s_pol", "port_pol")}
    for s in seeds:
        inst = im.generate_instance(seed=int(s))
        gr = greedy_routes(env=make_dynamic_env(inst, travel=None, fleet_size=_FLEET))
        _, rl = multistart_greedy(
            pol, make_dynamic_env(inst, travel=None, fleet_size=_FLEET), rl_starts
        )
        envs = [make_dynamic_env(inst, travel=None, fleet_size=_FLEET) for _ in range(k_samples)]
        envs[0].reset(seed=0)
        sk = pol.sample_k(envs, pol.encode(envs[0]), temperature=temp, seed=0)
        sk_best = take_best(sk, inst, None, _CFG)[0]
        g_raw, r_raw, s_raw = _cost(gr, inst), _cost(rl, inst), _cost(sk_best, inst)
        g_pol, r_pol, s_pol = (_pol(x, inst, budget_ms) for x in (gr, rl, sk_best))
        for k, v in zip(
            acc, (g_raw, g_pol, r_raw, r_pol, s_raw, s_pol, min(g_pol, r_pol, s_pol)), strict=True
        ):
            acc[k].append(v)
    m = {k: float(np.mean(v)) for k, v in acc.items()}

    def _row(name, raw, pol):
        print(f"  {name:9s}: {raw:7.1f} → {pol:7.1f} €  (polish {pol / raw - 1:+.1%})")

    print(f"\n=== Static polish (full-62, seeds 0–{len(seeds) - 1}, free-flow, to convergence) ===")
    _row("greedy", m["g_raw"], m["g_pol"])
    _row("RL-multi", m["r_raw"], m["r_pol"])
    _row("sample-K", m["s_raw"], m["s_pol"])
    print("  " + "─" * 42)
    print(f"  polished-portfolio: {m['port_pol']:7.1f} €")
    print(f"    vs Step-3 portfolio 766.1 : {m['port_pol'] / 766.1 - 1:+.1%}")
    print(f"    vs greedy {g_ref:.1f}         : {m['port_pol'] / g_ref - 1:+.1%}")
    print(f"    vs OR-Tools {o_ref:.1f}       : {m['port_pol'] / o_ref - 1:+.1%}  ← new gap")
    print(f"  RL edge after polish (RL vs greedy polished): {m['r_pol'] / m['g_pol'] - 1:+.1%}")
    return {
        "budget_ms": budget_ms,
        "seeds": [int(s) for s in seeds],
        "means": m,
        "greedy_ref_eur": g_ref,
        "ortools_ref_eur": o_ref,
        "step3_portfolio_eur": 766.1,
        "portfolio_gap_ortools": m["port_pol"] / o_ref - 1,
        "portfolio_vs_step3": m["port_pol"] / 766.1 - 1,
        "polish_delta_greedy": m["g_pol"] / m["g_raw"] - 1,
        "polish_delta_rl": m["r_pol"] / m["r_raw"] - 1,
        "rl_edge_after_polish": m["r_pol"] / m["g_pol"] - 1,  # <0 → RL still ahead of greedy
    }


def dynamic_polish(planner, seeds, *, deadline_s, n_events, ckpt) -> dict:
    res = rd.run(
        list(seeds), deadline_s=deadline_s, n_events=n_events, ckpt=ckpt, rl_planner=planner
    )
    agg = {m: rd._agg(res["records"], m) for m in rd._METHODS}
    rd._print_table(agg)
    rl = {(r["seed"], r["at_min"]): -r["reward"] for r in res["records"] if r["method"] == "rl"}
    gr = {(r["seed"], r["at_min"]): -r["reward"] for r in res["records"] if r["method"] == "greedy"}
    d = [rl[k] - gr[k] for k in rl if k in gr]
    guar = {
        "n_events": len(d),
        "violations": int(sum(x > 1e-6 for x in d)),
        "worst_delta_eur": float(max(d)) if d else 0.0,
        "median_delta_eur": float(np.median(d)) if d else 0.0,
    }
    lat = agg["rl"]["latency_ms_median"]
    print(
        f"GUARANTEE RL≤greedy: violations {guar['violations']}/{guar['n_events']} "
        f"(worst Δ {guar['worst_delta_eur']:+.2f}€) | latency portfolio+polish {lat:.0f}ms "
        f"({'<1s ✓' if lat < 1000 else '≥1s ✗'})"
    )
    return {"aggregates": agg, "guarantee": guar, "n_records": len(res["records"])}


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 6b Step 3.5 — local-search polish")
    ap.add_argument("--ckpt", type=str, default=str(_CKPT))
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--dyn-seeds", type=int, default=5)
    ap.add_argument("--static-budget-ms", type=float, default=8000.0, help="to convergence at n=62")
    ap.add_argument(
        "--polish-budget-ms", type=float, default=400.0, help="dynamic: shared polish budget"
    )
    ap.add_argument("--k-static", type=int, default=128)
    ap.add_argument("--k-dyn", type=int, default=16)
    ap.add_argument("--rl-starts", type=int, default=16)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--events", type=int, default=6)
    ap.add_argument("--deadline", type=int, default=2)
    ap.add_argument("--out", type=str, default=str(_OUT))
    ap.add_argument("--skip-static", action="store_true")
    args = ap.parse_args()

    ckpt = Path(args.ckpt)
    torch.manual_seed(0)
    pol = rd._load_policy(ckpt)

    stat = None
    if not args.skip_static:
        stat = static_polish(
            pol,
            range(args.seeds),
            budget_ms=args.static_budget_ms,
            k_samples=args.k_static,
            temp=args.temp,
            rl_starts=args.rl_starts,
        )
    planner = PortfolioPlanner(
        pol,
        k_samples=args.k_dyn,
        temperature=args.temp,
        rl_starts=max(4, args.rl_starts // 2),
        polish_budget_ms=args.polish_budget_ms,
        polish_top_m=5,
    )
    dyn = dynamic_polish(
        planner, range(args.dyn_seeds), deadline_s=args.deadline, n_events=args.events, ckpt=ckpt
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "phase": "6b-step3.5-local-search-polish",
        "config": {
            "ckpt": str(ckpt),
            "static_seeds": list(range(args.seeds)),
            "dyn_seeds": list(range(args.dyn_seeds)),
            "static_budget_ms": args.static_budget_ms,
            "polish_budget_ms": args.polish_budget_ms,
            "k_static": args.k_static,
            "k_dyn": args.k_dyn,
            "rl_starts": args.rl_starts,
            "temperature": args.temp,
        },
        "provenance": rd._provenance(ckpt),
        "static": stat,
        "dynamic": dyn,
        "note": "polish: 2-opt+Or-opt(1-3) intra, relocate+swap inter; cost/feasibility — the "
        "single evaluate_solution/check_feasible (correct under time-dependence, full eval per "
        "move). Invariant polish ≤ input. Static: convergence; dynamic: in-budget. Outside git #1.",
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"\nsummary → {out}")


if __name__ == "__main__":
    main()
