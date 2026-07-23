"""Задача #15 — time-matched сравнение: anytime-кривая OR-Tools vs deployed-система (БЕЗ обучения).

Финал бенчмарков. Даём OR-Tools ТОТ ЖЕ wall-clock и меряем качество на ИДЕНТИЧНЫХ инстансах
(full-62, seeds 0–9, тот же `generate_instance` + единый `evaluate_solution`, запрет №3). Бюджеты
{0.7,2,5,30}с → кривая cost(budget). ПАРИТИ-СТРАЖ: 30с-точка == 611.1€ (0002), иначе разъезд.

ЧЕСТНОСТЬ (карантин конфляции Phase 8): статические 631.6€ система берёт polish-бюджетом
30000мс/кандидат ×≤3 (`system_metrics.json`) → её статик-wall-clock ≥30с, НЕ 689мс. 689мс —
ДИНАМИЧЕСКАЯ re-plan латентность на residual (cost там 827€, `polish_summary.json`) — другой
сеттинг. Систему кладём на её РЕАЛЬНЫЙ статик-x (≥30с, y=631.6€); 689мс/827€ — отдельным полем как
динамика. Точку пересечения кривой считаем из данных, не пред-судим.

Пишет ТОЛЬКО results/timematch.json. Секцию «Time-matched» в docs/final_metrics.md эмитит
final_metrics.py (единый владелец файла — иначе перезапись/дубль). Запуск:
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
_ANCHOR_0002_ORTOOLS = 611.14  # 30с-точка (decision 0002) — парити-страж
# wall-clock-допуск: OR-Tools time-limited = best-so-far (не бит-детерм.), как eval_system tol=2.0.
# Ловит РАЗЪЕЗД пути/инстансов (десятки €), не джиттер сходимости GLS (<2€).
_PARITY_TOL = 2.0


def _sys_ref() -> dict:
    """Честная точка системы: статик-качество 631.6€ на её РЕАЛЬНОМ статик-x (polish-бюджет ≥30с).

    689мс/827€ — ОТДЕЛЬНО как динамическая re-plan латентность (другой сеттинг, не статик-wall-clock
    для 631.6€). Числа из durable json (парити), при отсутствии — decision-якоря."""
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
        "static_wallclock_s": static_budget_ms / 1000.0,  # НИЖНЯЯ граница: ×≤3 кандидата + decode
        "static_wallclock_note": f"polish {static_budget_ms:.0f}мс/кандидат ×≤3 + decode → ≥"
        f"{static_budget_ms / 1000.0:.0f}с (кандидаты последовательно; истинный wall-clock выше)",
        "dynamic_replan_latency_ms": dyn_lat_ms,  # КАРАНТИН: другой сеттинг (residual), НЕ 631.6€
        "dynamic_replan_cost_eur": dyn_cost,
        "conflation_avoided": "631.6€ — статика (≥30с polish), НЕ «@689мс»; 689мс — динамика "
        "на residual (cost 827€). Разные сеттинги — не одна точка.",
    }


def run(seeds: list[int], budgets: list[float]) -> dict:
    cfg = CostConfig()
    per_seed = {b: [] for b in budgets}
    for s in seeds:
        inst = im.generate_instance(seed=s)  # тот же путь, что run_baselines (парити 30с=611.1)
        for b in budgets:  # один инстанс → все бюджеты (тот же instance, запрет №3)
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
    or_best = curve[-1]["cost_mean"]  # 30с — сильнейшая OR-точка
    sys_ref = _sys_ref()

    # crossover: наименьший бюджет, где OR-Tools ≤ статик-качества системы (631.6€) — из данных
    xover = next((c["budget_s"] for c in curve if c["cost_mean"] <= sys_ref["cost_eur"]), None)

    # ПАРИТИ-СТРАЖ: 30с-точка == 611.1€ (0002)
    parity_ok = abs(or_best - _ANCHOR_0002_ORTOOLS) < _PARITY_TOL
    assert parity_ok, (
        f"ПАРИТИ FAIL: OR-Tools@30с {or_best:.2f}€ != anchor 0002 {_ANCHOR_0002_ORTOOLS}€ "
        f"(|Δ|={abs(or_best - _ANCHOR_0002_ORTOOLS):.2f} > {_PARITY_TOL}) — путь разъехался"
    )
    return {
        "phase": "15-time-matched",
        "note": "anytime-кривая OR-Tools (full-62, seeds 0–9, единый scorer) vs статик-качество "
        "системы 631.6€. Даём OR-Tools тот же wall-clock. cost = −reward, €. Точка системы — на её "
        "РЕАЛЬНОМ статик-x (≥30с polish), НЕ на 689мс (это динамика, поле dynamic_replan_*).",
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
        "crossover_budget_s": xover,  # None → OR-Tools не достиг качества системы в 30с (не ждём)
        "parity": {"anchor_0002_ortools": _ANCHOR_0002_ORTOOLS, "measured_30s": or_best,
                   "ok": parity_ok},
    }


def print_report(res: dict) -> None:
    c = res["config"]
    sr = res["system_ref"]
    print(f"\nTime-matched — OR-Tools vs система | K={c['fleet_size']} Q={c['vehicle_cap']} "
          f"| seeds={res['run']['seeds']} | OR-Tools {res['run']['ortools_version']}")
    print(f"{'бюджет':>10}  {'cost,€ (mean±std)':>22}  {'vs система 631.6':>16}  {'vs OR@30с':>10}")
    print("-" * 66)
    or_best = res["ortools_best_30s_eur"]
    for pt in res["curve"]:
        m, sd = pt["cost_mean"], pt["cost_std"]
        vs_sys = m - sr["cost_eur"]
        vs_or = m - or_best
        print(f"{pt['budget_s']:>9.1f}с  {m:>10.1f} ± {sd:6.1f}       "
              f"{vs_sys:>+8.1f}€        {vs_or:>+7.1f}€")
    print("-" * 66)
    xo = res["crossover_budget_s"]
    xo_s = f"{xo:.1f}с" if xo is not None else ">30с (не достиг за 30с)"
    print(f"Система: {sr['cost_eur']:.1f}€ при wall-clock ≥{sr['static_wallclock_s']:.0f}с "
          f"({sr['static_wallclock_note']}).")
    print(f"OR-Tools ≤ 631.6€ (качество системы) достигает при бюджете: {xo_s}.")
    print(f"КАРАНТИН: 689мс — динамическая re-plan латентность (residual, cost "
          f"{sr['dynamic_replan_cost_eur']:.0f}€), НЕ статик-wall-clock для 631.6€.")
    print(f"\nВЕРДИКТ: time-matched в статике OR-Tools@30с = {or_best:.1f}€ "
          f"{'БЬЁТ' if or_best < sr['cost_eur'] else 'НЕ бьёт'} систему {sr['cost_eur']:.1f}€ "
          f"({or_best - sr['cost_eur']:+.1f}€); edge системы — только динамика, не статика.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Задача #15 — time-matched OR-Tools vs система")
    ap.add_argument("--seeds", type=int, default=10, help="число сидов (0..N-1)")
    ap.add_argument("--budgets", type=float, nargs="+", default=_BUDGETS, help="бюджеты, с")
    ap.add_argument("--out", type=Path, default=_RES / "timematch.json")
    args = ap.parse_args()
    if args.seeds < 1:
        ap.error("--seeds должен быть >= 1")

    res = run(list(range(args.seeds)), sorted(args.budgets))
    print_report(res)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"\nсводка → {args.out}  (секцию в docs/final_metrics.md эмитит final_metrics.py)")


if __name__ == "__main__":
    main()
