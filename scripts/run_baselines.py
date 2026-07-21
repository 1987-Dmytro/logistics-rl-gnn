"""Прогон бейзлайнов «Было»: greedy + OR-Tools на N сидов, единый оценщик, mean±std.

Запуск:  python scripts/run_baselines.py [--seeds 10] [--time-limit 30]
Печатает таблицу «Было» (пробег/время/машины/on-time/unserved/reward) и пишет сводку
results/baselines.json (config+seeds+версии → воспроизводимость, запрет №4).
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
from logistics_rl_gnn.baselines.greedy import greedy_routes
from logistics_rl_gnn.config import instance as im
from logistics_rl_gnn.env.scoring import CostConfig, evaluate_solution

_METRICS = ["distance_km", "time_min", "vehicles_used", "on_time_pct", "unserved", "reward"]
_COLS = [
    ("пробег,км", "distance_km"),
    ("время,мин", "time_min"),
    ("машин", "vehicles_used"),
    ("on-time,%", "on_time_pct"),
    ("unserved", "unserved"),
    ("reward,€", "reward"),
]


def _agg(rows: list[dict]) -> dict:
    return {
        m: {
            "mean": float(np.mean([r[m] for r in rows])),
            "std": float(np.std([r[m] for r in rows])),
        }
        for m in _METRICS
    }


def run(seeds: list[int], time_limit_s: int) -> dict:
    cfg = CostConfig()
    per_seed = {"greedy": [], "ortools": []}
    for s in seeds:
        inst = im.generate_instance(seed=s)
        per_seed["greedy"].append(evaluate_solution(greedy_routes(seed=s), inst, cfg))
        o = ortools_vrptw.ortools_routes(inst, cfg, time_limit_s=time_limit_s)
        per_seed["ortools"].append(evaluate_solution(o, inst, cfg))
    return {
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
            "ortools_time_limit_s": time_limit_s,
            "ortools_version": version("ortools"),
            "numpy_version": version("numpy"),
            "python": platform.python_version(),
        },
        "greedy": {"per_seed": per_seed["greedy"], "agg": _agg(per_seed["greedy"])},
        "ortools": {"per_seed": per_seed["ortools"], "agg": _agg(per_seed["ortools"])},
    }


def _fmt(name: str, agg: dict) -> str:
    cells = [f"{name:<9}"]
    for _, key in _COLS:
        m, sd = agg[key]["mean"], agg[key]["std"]
        cells.append(f"{m:8.2f}±{sd:6.2f}")
    return "  ".join(cells)


def print_table(res: dict) -> None:
    c = res["config"]
    print(
        f"\n«Было» — K={c['fleet_size']} Q={c['vehicle_cap']} T_max={c['t_max_min']:.0f}мин | "
        f"seeds={res['run']['seeds']} | OR-Tools {res['run']['ortools_version']} "
        f"@ {res['run']['ortools_time_limit_s']}с"
    )
    header = f"{'метод':<9}  " + "  ".join(f"{h:>15}" for h, _ in _COLS)
    print(header)
    print("-" * len(header))
    print(_fmt("greedy", res["greedy"]["agg"]))
    print(_fmt("OR-Tools", res["ortools"]["agg"]))


def main() -> None:
    ap = argparse.ArgumentParser(description="Бейзлайны CVRPTW «Было»")
    ap.add_argument("--seeds", type=int, default=10, help="число сидов (0..N-1)")
    ap.add_argument("--time-limit", type=int, default=30, help="бюджет OR-Tools, сек/сид")
    ap.add_argument("--out", type=Path, default=Path("results/baselines.json"))
    args = ap.parse_args()
    if args.seeds < 1 or args.time_limit < 1:
        ap.error("--seeds и --time-limit должны быть >= 1")  # иначе пустая агрегация → NaN в JSON

    res = run(list(range(args.seeds)), args.time_limit)
    print_table(res)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"\nсводка → {args.out}")


if __name__ == "__main__":
    main()
