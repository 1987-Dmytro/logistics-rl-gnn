"""Phase 6b Шаг 3 — инференс-поиск (БЕЗ обучения, только decode). Чекпойнт: congestion-best.

Три замера (все с полным provenance → results/search_summary.json):
  1) K-таблица: sample-K take-best vs (качество, латентность) при K=16/64/128/256 (де-риск
     латентности — она env-bound: батч-декод держит нейронку ~плоской по K, растёт K× env.step);
  2) static: full-62 / seeds 0–9 / free-flow — sample-K take-best И PortfolioPlanner vs OR-Tools
     (611€) и greedy (825€). Congestion-веса на free-flow-инстансах: channel-1 mult≡1,
     node_congestion=0 → congestion-фичи нейтральны, политика работает как обычная;
  3) dynamic: харнесс 0004 (`run_dynamic.run`) с PortfolioPlanner → RL(портфель) vs greedy vs
     OR-Tools. Гарантия ПО ПОСТРОЕНИЮ: RL ≤ greedy на КАЖДОМ событии (счётчик нарушений=0 +
     худшая per-event дельта) + латентность < 1с.

Запуск: python scripts/run_search.py [--ckpt PATH] [--seeds N] [--k-static K] [--k-dyn K]
        [--temp T] [--events E] [--deadline S] [--out PATH] [--skip-ktable]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))  # переиспользуем харнесс run_dynamic
import run_dynamic as rd  # noqa: E402

from logistics_rl_gnn.baselines.greedy import greedy_routes  # noqa: E402
from logistics_rl_gnn.config import instance as im  # noqa: E402
from logistics_rl_gnn.env.events import make_dynamic_env  # noqa: E402
from logistics_rl_gnn.env.scoring import CostConfig, evaluate_solution  # noqa: E402
from logistics_rl_gnn.replan.portfolio import PortfolioPlanner, take_best  # noqa: E402

_CKPT = Path("results/policy_pomo_congestion.pt")
_OUT = Path("results/search_summary.json")
_CFG = CostConfig()


def _greedy_cost(inst, travel, fleet):
    routes = greedy_routes(env=make_dynamic_env(inst, travel=travel, fleet_size=fleet))
    return -evaluate_solution(routes, inst, _CFG, travel=travel)["reward"]


def _sample_take_best(pol, inst, travel, fleet, K, temp, *, seed=0):
    """sample-K take-best (только стохастические роллауты) + латентность батч-декода."""
    envs = [make_dynamic_env(inst, travel=travel, fleet_size=fleet) for _ in range(K)]
    envs[0].reset(seed=0)
    enc = pol.encode(envs[0])
    t0 = time.perf_counter()
    routes = pol.sample_k(envs, enc, temperature=temp, seed=seed)
    ms = (time.perf_counter() - t0) * 1000.0
    _, cost, _ = take_best(routes, inst, travel, _CFG)
    return cost, ms


def k_table(pol, seeds, temp, ks=(16, 64, 128, 256)) -> dict:
    """sample-K take-best на full-62 free-flow: качество (vs greedy) и латентность по K."""
    insts = [im.generate_instance(seed=int(s)) for s in seeds]
    gd = [_greedy_cost(i, None, im.FLEET_SIZE) for i in insts]
    print("\n=== K-таблица (sample-K take-best, full-62 free-flow, ср. по сидам) ===")
    print(f"{'K':>5} | {'best,€':>8} | {'vs greedy':>9} | {'лат,мс (медиана)':>16}")
    print("-" * 48)
    gd_mean = float(np.mean(gd))
    rows = []
    for K in ks:
        costs, lats = [], []
        for i in insts:
            _sample_take_best(pol, i, None, im.FLEET_SIZE, K, temp, seed=1)  # warmup
            c, ms = _sample_take_best(pol, i, None, im.FLEET_SIZE, K, temp)
            costs.append(c)
            lats.append(ms)
        cost, lat = float(np.mean(costs)), statistics.median(lats)
        gap = cost / gd_mean - 1
        print(f"{K:5d} | {cost:8.1f} | {gap:+8.1%} | {lat:16.1f}")
        rows.append({"K": K, "cost_eur": cost, "gap_greedy": gap, "latency_ms_median": lat})
    print("латентность ~линейна по K (env-bound: K× env.step/шаг; нейронка батчится → плоская).")
    return {"greedy_mean_eur": gd_mean, "rows": rows}


def static_gap(pol, planner, seeds, K, temp) -> dict:
    """full-62 free-flow: sample-K take-best И portfolio vs OR-Tools 611€ / greedy 825€."""
    bl = json.loads(Path("results/baselines.json").read_text())
    g_ref = -bl["greedy"]["agg"]["reward"]["mean"]
    o_ref = -bl["ortools"]["agg"]["reward"]["mean"]
    sk, port, gd = [], [], []
    for s in seeds:
        inst = im.generate_instance(seed=int(s))
        gd.append(_greedy_cost(inst, None, im.FLEET_SIZE))
        sk.append(_sample_take_best(pol, inst, None, im.FLEET_SIZE, K, temp)[0])
        port.append(planner.plan(inst, None, fleet_size=im.FLEET_SIZE)["cost"])
    sk_m, port_m, gd_m = (float(np.mean(x)) for x in (sk, port, gd))
    print(f"\n=== Static gap (full-62, seeds 0–{len(seeds) - 1}, free-flow; K={K} temp={temp}) ===")
    print(f"  greedy (baseline)   : {g_ref:7.1f} €")
    print(f"  OR-Tools (baseline) : {o_ref:7.1f} €")
    print(
        f"  sample-K take-best  : {sk_m:7.1f} €  "
        f"(vs greedy {sk_m / g_ref - 1:+.1%}, vs OR {sk_m / o_ref - 1:+.1%})"
    )
    print(
        f"  PortfolioPlanner    : {port_m:7.1f} €  ← лучший  "
        f"(vs greedy {port_m / g_ref - 1:+.1%}, vs OR {port_m / o_ref - 1:+.1%})"
    )
    return {
        "greedy_ref_eur": g_ref,
        "ortools_ref_eur": o_ref,
        "seeds": [int(s) for s in seeds],
        "sample_take_best_eur": sk_m,
        "portfolio_eur": port_m,
        "greedy_local_eur": gd_m,
        "sample_gap_ortools": sk_m / o_ref - 1,
        "portfolio_gap_ortools": port_m / o_ref - 1,
        "portfolio_gap_greedy": port_m / g_ref - 1,
    }


def _guarantee(records, eps=1e-6) -> dict:
    """RL(портфель) vs greedy per-event: нарушений 0 (гарантия), худшая/медиана дельта."""
    rl = {(r["seed"], r["at_min"]): -r["reward"] for r in records if r["method"] == "rl"}
    gr = {(r["seed"], r["at_min"]): -r["reward"] for r in records if r["method"] == "greedy"}
    d = [rl[k] - gr[k] for k in rl if k in gr]  # €; ≤0 = не хуже greedy
    return {
        "n_events": len(d),
        "violations": int(sum(x > eps for x in d)),
        "worst_delta_eur": float(max(d)) if d else 0.0,
        "median_delta_eur": float(np.median(d)) if d else 0.0,
    }


def dynamic_table(planner, seeds, *, deadline_s, n_events, ckpt) -> dict:
    """Харнесс 0004 с портфелем → таблица + гарантия per-event + латентность."""
    res = rd.run(
        list(seeds), deadline_s=deadline_s, n_events=n_events, ckpt=ckpt, rl_planner=planner
    )
    agg = {m: rd._agg(res["records"], m) for m in rd._METHODS}
    rd._print_table(agg)
    g = _guarantee(res["records"])
    print(
        f"ГАРАНТИЯ RL≤greedy: нарушений {g['violations']}/{g['n_events']} "
        f"(худшая Δ {g['worst_delta_eur']:+.2f}€, медиана {g['median_delta_eur']:+.2f}€) | "
        f"латентность портфеля {agg['rl']['latency_ms_median']:.0f}мс "
        f"({'<1с ✓' if agg['rl']['latency_ms_median'] < 1000 else '≥1с ✗'})"
    )
    return {"aggregates": agg, "guarantee": g, "n_records": len(res["records"])}


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 6b Шаг 3 — инференс-поиск (portfolio)")
    ap.add_argument("--ckpt", type=str, default=str(_CKPT))
    ap.add_argument("--seeds", type=int, default=10, help="static: seeds 0..N-1")
    ap.add_argument("--dyn-seeds", type=int, default=5, help="dynamic: seeds 0..N-1")
    ap.add_argument("--k-static", type=int, default=128)
    ap.add_argument("--k-dyn", type=int, default=32, help="малый K → латентность < 1с")
    ap.add_argument("--rl-starts", type=int, default=16)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--events", type=int, default=6)
    ap.add_argument("--deadline", type=int, default=2)
    ap.add_argument("--out", type=str, default=str(_OUT))
    ap.add_argument("--skip-ktable", action="store_true")
    args = ap.parse_args()

    ckpt = Path(args.ckpt)
    torch.manual_seed(0)
    pol = rd._load_policy(ckpt)
    static_seeds = range(args.seeds)

    kt = None if args.skip_ktable else k_table(pol, range(min(3, args.seeds)), args.temp)
    p_static = PortfolioPlanner(
        pol, k_samples=args.k_static, temperature=args.temp, rl_starts=args.rl_starts
    )
    stat = static_gap(pol, p_static, static_seeds, args.k_static, args.temp)
    p_dyn = PortfolioPlanner(
        pol, k_samples=args.k_dyn, temperature=args.temp, rl_starts=max(4, args.rl_starts // 2)
    )
    dyn = dynamic_table(
        p_dyn, range(args.dyn_seeds), deadline_s=args.deadline, n_events=args.events, ckpt=ckpt
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "phase": "6b-step3-inference-search",
        "config": {
            "ckpt": str(ckpt),
            "static_seeds": list(static_seeds),
            "dyn_seeds": list(range(args.dyn_seeds)),
            "k_static": args.k_static,
            "k_dyn": args.k_dyn,
            "rl_starts": args.rl_starts,
            "temperature": args.temp,
            "n_events": args.events,
            "deadline_s": args.deadline,
        },
        "provenance": rd._provenance(ckpt),
        "k_table": kt,
        "static": stat,
        "dynamic": dyn,
        "note": "БЕЗ обучения — только decode. Гарантия RL≤greedy = greedy-кандидат в портфеле "
        "байт-идентичен методу greedy (тот же instance+travel+fleet+scorer). Static — free-flow "
        "(congestion-фичи нейтральны). Веса вне git (запрет №1); число = seed+config+sha+версии.",
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"\nсводка → {out}")


if __name__ == "__main__":
    main()
