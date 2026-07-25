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
_DURABLE_COST_0009 = 631.6212305905019  # port_pol из polish_summary.json — якорь парити


def _env(inst):
    return make_dynamic_env(inst, travel=None)  # K/Q из инстанса (сценарий → meta, дефолт → 8×80)


def system_routes(pol, inst, *, budget_ms, k_samples, temp, rl_starts, report=None):
    """Полир. портфель: best-по-cost из {greedy, RL-multi, sample-K}, каждый polish до сходимости.

    Идентично отбору port_pol в run_polish.static_polish (0009), но возвращает МАРШРУТЫ победителя
    (для полного вектора метрик), а не только cost.

    pol=None → ablation `--no-model`: RL-кандидаты не порождаются (портфель = greedy+polish).
    report (опц. dict) наполняется таблицей кандидатов: rows/chosen/cost_model/cost_nomodel/
    rl_mean. Здесь polish получают ВСЕ кандидаты → «без модели» = polished greedy, точный
    контрфактуал того же прогона (в отличие от re-plan, где polish берёт лишь топ-M)."""
    fleet, cap = im.fleet_of(inst)
    labels, cands = ["greedy"], [greedy_routes(env=_env(inst))]
    if pol is not None:
        _, rl = multistart_greedy(pol, _env(inst), rl_starts)  # None при отсутствии feasible старта
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
        for r in rows:  # polish здесь получает КАЖДЫЙ кандидат (в отличие от re-plan)
            r["polished"] = polished[labels.index(r["source"])][1]
        if pol is not None:  # sample-K: n/mean по ВСЕМ K роллаутам, не по одному победителю
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
        # per-seed cost для ПАРНОГО сравнения (те же seeds, что OR-Tools в timematch → σ инстанса
        # сокращается; median-Δ + wins, как 0010). mean(per_seed) == means.cost_eur (парити).
        "per_seed_cost_eur": [float(-r) for r in acc["reward"]],
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
