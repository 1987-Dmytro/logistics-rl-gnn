"""Phase 8 — один seeded eval СИСТЕМЫ (полир. портфель) для ПОЛНОГО вектора метрик.

Durable json ([[0009]] polish_summary) хранил только COST системы; для «Было/Стало»-таблицы нужны
также пробег/время/машины. Прогоняем полир. портфель ОДИН раз на сидах 0–9 (== baselines/0009,
full-62, free-flow), пишем полный вектор в results/system_metrics.json с provenance.

**ПАРИТИ-СТРАЖ:** агрегатный cost ОБЯЗАН совпасть с durable 631.6€ (0009 port_pol) — иначе это НЕ
та система (assert-ошибка, не тихий разъезд). Не заменяет decision 0009: воспроизводит его число +
добирает незалогированные метрики. Детерминирован (greedy/multistart/sample_k seed=0; polish до
сходимости). Запуск: python scripts/eval_system.py [--seeds N] [--budget-ms MS]
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
from logistics_rl_gnn.replan.portfolio import take_best  # noqa: E402
from logistics_rl_gnn.train.pomo import multistart_greedy  # noqa: E402

_CKPT = Path("results/policy_pomo_congestion.pt")
_OUT = Path("results/system_metrics.json")
_CFG = CostConfig()
_FLEET = im.FLEET_SIZE
_Q = ("distance_km", "time_min", "vehicles_used", "on_time_pct", "unserved")
_DURABLE_COST_0009 = 631.6212305905019  # port_pol из polish_summary.json — якорь парити


def _env(inst):
    return make_dynamic_env(inst, travel=None, fleet_size=_FLEET)


def system_routes(pol, inst, *, budget_ms, k_samples, temp, rl_starts):
    """Полир. портфель: best-по-cost из {greedy, RL-multi, sample-K}, каждый polish до сходимости.

    Идентично отбору port_pol в run_polish.static_polish (0009), но возвращает МАРШРУТЫ победителя
    (для полного вектора метрик), а не только cost."""
    gr = greedy_routes(env=_env(inst))
    _, rl = multistart_greedy(pol, _env(inst), rl_starts)  # None при отсутствии feasible старта
    envs = [_env(inst) for _ in range(k_samples)]
    envs[0].reset(seed=0)
    sk = pol.sample_k(envs, pol.encode(envs[0]), temperature=temp, seed=0)
    sk_best = take_best(sk, inst, None, _CFG)[0]
    cands = [c for c in (gr, rl, sk_best) if c is not None]
    polished = [polish(c, inst, None, budget_ms=budget_ms, fleet_size=_FLEET) for c in cands]
    best_routes, _ = min(polished, key=lambda rc: rc[1])  # (routes, cost) → min cost
    return best_routes


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 8 — полный вектор метрик системы (парити 0009)")
    ap.add_argument("--ckpt", default=str(_CKPT))
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--budget-ms", type=float, default=30000.0, help="polish до сходимости (0009)")
    ap.add_argument("--k-samples", type=int, default=128)
    ap.add_argument("--rl-starts", type=int, default=16)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--tol", type=float, default=2.0, help="парити-допуск к 631.6€ (wall-clock)")
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
    # ПАРИТИ-СТРАЖ: cost обязан совпасть с durable 631.6 (иначе НЕ та система)
    assert abs(cost - _DURABLE_COST_0009) < args.tol, (
        f"ПАРИТИ FAIL: cost {cost:.2f}€ != durable 0009 {_DURABLE_COST_0009:.2f}€ "
        f"(|Δ|={abs(cost - _DURABLE_COST_0009):.2f} > {args.tol}) — НЕ та система или budget-bound"
    )
    out = {
        "phase": "8-system-full-vector",
        "note": "полир. портфель (== [[0009]]), ПОЛНЫЙ вектор; cost парити с 0009 631.6€. "
        "seeds 0-9 full-62 free-flow. Один seeded eval (не заменяет 0009, добирает метрики).",
        "config": {
            "ckpt": str(ckpt), "seeds": list(range(args.seeds)), "budget_ms": args.budget_ms,
            "k_samples": args.k_samples, "rl_starts": args.rl_starts, "temperature": args.temp,
            "fleet_size": _FLEET,
        },
        "provenance": rd._provenance(ckpt),
        "durable_cost_anchor_0009": _DURABLE_COST_0009,
        "means": {
            "cost_eur": cost, "distance_km": mean["distance_km"], "time_min": mean["time_min"],
            "vehicles_used": mean["vehicles_used"], "on_time_pct": mean["on_time_pct"],
            "unserved": mean["unserved"],
        },
    }
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\nСИСТЕМА (полир. портфель, seeds 0-9): cost {cost:.1f}€ (парити 0009 631.6 ✓) "
          f"dist {mean['distance_km']:.1f}km time {mean['time_min']:.0f}min "
          f"veh {mean['vehicles_used']:.1f}")
    print(f"→ {args.out}")


if __name__ == "__main__":
    main()
