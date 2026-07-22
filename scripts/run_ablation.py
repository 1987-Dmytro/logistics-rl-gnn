"""Phase 6b Ablation — латентная ниша RL. БЕЗ обучения.

Отвечает на открытый вопрос [[0009]]: есть ли режим, где СТАРТОВАЯ точка RL решает — т.е. где
polish НЕ успевает выровнять старты под жёстким realtime-бюджетом (0009 выровнял старты лишь у
СХОДИМОСТИ, ~16с на n=62; здесь все бюджеты ≤ 500мс — заведомо ДО сходимости).

Тот же 0004-харнесс (`run_dynamic.iter_events`: 5 сидов × 6 событий, идентичная provenance). Для
КАЖДОГО сработавшего события — системы под жёстким END-TO-END бюджетом budget_ms ∈ {50,100,200,500}:
  • rl_raw        — greedy-decode политики (без polish, без портфеля): «мгновенный» RL-старт;
  • greedy_raw    — эвристика greedy без polish: контроль СТАРТА (изолирует старт от polish);
  • greedy_polish — greedy + polish в ОСТАТОК бюджета;
  • rl_polish     — RL greedy-decode + polish в ОСТАТОК бюджета.
Бюджет end-to-end (decode+polish); превышение = берём что есть на дедлайне (polish уже anytime по
budget_ms). Единый скорер `evaluate_solution` под тем же travel (запрет №3). Парные сравнения по
событиям (median Δ, wins), как в 0007. rl_raw/greedy_raw budget-независимы → считаем 1 раз/событие.

ВНИМАНИЕ (запрет №4): cost *_polish систем budget-bound → wall-clock-dependent (сколько улучшающих
ходов влезло до дедлайна = как быстро железо; тот же seed → другой маршрут). Детерминированы только
raw-старты (decode/greedy). Числа привязаны к config+версиям+hash чекпойнта, НЕ к абсолюту.

Запуск: python scripts/run_ablation.py [--ckpt P] [--seeds N] [--events K] [--out P]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import run_dynamic as rd  # noqa: E402

from logistics_rl_gnn.baselines.greedy import greedy_routes  # noqa: E402
from logistics_rl_gnn.config import congestion as cg  # noqa: E402
from logistics_rl_gnn.config import instance as im  # noqa: E402
from logistics_rl_gnn.env.events import make_dynamic_env  # noqa: E402
from logistics_rl_gnn.env.scoring import CostConfig, evaluate_solution  # noqa: E402
from logistics_rl_gnn.replan.local_search import polish  # noqa: E402

_CKPT = Path("results/policy_pomo_congestion.pt")
_OUT = Path("results/ablation_summary.json")
_BUDGETS_MS = (50.0, 100.0, 200.0, 500.0)
_CFG = CostConfig()
_POLISH_FLOOR_MS = 3.0  # ниже остатка polish не успеет и одного full-eval → берём raw-старт
_EPS = 1e-6


def _cost(routes, res, travel) -> float:
    """ЕДИНЫЙ скорер ВСЕХ систем: € = −reward под тем же travel (запрет №3)."""
    return -evaluate_solution(routes, res, _CFG, travel=travel)["reward"]


def _rl_decode(policy, res, travel, fleet):
    env = make_dynamic_env(res, travel=travel, fleet_size=fleet)
    with torch.no_grad():  # инференс: без autograd-графа (честная латентность)
        return policy.rollout(env, mode="greedy")[0]


def _greedy(res, travel, fleet):
    return greedy_routes(env=make_dynamic_env(res, travel=travel, fleet_size=fleet))


def _run_budget(construct, res, travel, fleet, budget_ms: float) -> dict:
    """Старт construct() + polish в ОСТАТОК end-to-end бюджета. Превышение → что есть на дедлайне.

    -> {raw_cost, cost (после polish), construct_ms, polish_ms, latency_ms}. Латентность =
    построение + polish (скоринг — вне тайминга, это измерение, не деплой-стоимость).
    """
    t0 = time.perf_counter()
    raw = construct()
    construct_ms = (time.perf_counter() - t0) * 1000.0
    remaining = budget_ms - construct_ms
    routes = raw
    if remaining > _POLISH_FLOOR_MS:  # anytime: polish сам держит дедлайн по budget_ms
        routes, _ = polish(raw, res, travel, budget_ms=remaining, fleet_size=fleet)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "raw_cost": _cost(raw, res, travel),
        "cost": _cost(routes, res, travel),
        "construct_ms": construct_ms,
        "polish_ms": max(0.0, latency_ms - construct_ms),
        "latency_ms": latency_ms,
    }


def measure_event(policy, seed, ev, res, travel, fleet) -> list[dict]:
    """Все системы на ОДНОМ событии × все бюджеты. raw-старты — один раз (budget-независимы)."""
    _rl_decode(policy, res, travel, fleet)  # warmup: гасим torch lazy-init ВНЕ тайминга
    key = {
        "seed": int(seed),
        "event": ev.kind,
        "at_min": round(ev.at_min, 1),
        "n_pending": len(res.demand) - 1,
        "fleet": int(fleet),
    }
    # budget-независимые старты (считаем один раз, держим константой по бюджетам — advisor #1/#2)
    t0 = time.perf_counter()
    rl_r = _rl_decode(policy, res, travel, fleet)
    rl_ms = (time.perf_counter() - t0) * 1000.0
    rl_raw_cost = _cost(rl_r, res, travel)
    t0 = time.perf_counter()
    gr_r = _greedy(res, travel, fleet)
    gr_ms = (time.perf_counter() - t0) * 1000.0
    gr_raw_cost = _cost(gr_r, res, travel)

    rows: list[dict] = []
    for b in _BUDGETS_MS:
        gp = _run_budget(lambda: _greedy(res, travel, fleet), res, travel, fleet, b)
        rp = _run_budget(lambda: _rl_decode(policy, res, travel, fleet), res, travel, fleet, b)
        rows += [
            {**key, "budget_ms": b, "system": "greedy_raw", "cost": gr_raw_cost,
             "raw_cost": gr_raw_cost, "construct_ms": gr_ms, "polish_ms": 0.0,
             "latency_ms": gr_ms},
            {**key, "budget_ms": b, "system": "rl_raw", "cost": rl_raw_cost,
             "raw_cost": rl_raw_cost, "construct_ms": rl_ms, "polish_ms": 0.0,
             "latency_ms": rl_ms},
            {**key, "budget_ms": b, "system": "greedy_polish", **gp},
            {**key, "budget_ms": b, "system": "rl_polish", **rp},
        ]
    return rows


def _paired(rows, sys_a, sys_b, budget) -> dict:
    """Δ = cost[a] − cost[b] по событиям при данном budget. a лучше b при Δ < 0."""
    a = {(r["seed"], r["at_min"]): r["cost"]
         for r in rows if r["system"] == sys_a and r["budget_ms"] == budget}
    b = {(r["seed"], r["at_min"]): r["cost"]
         for r in rows if r["system"] == sys_b and r["budget_ms"] == budget}
    assert set(a) == set(b), f"event-key mismatch {sys_a} vs {sys_b} @ {budget}"  # advisor #5
    d = [a[k] - b[k] for k in a]
    return {
        "budget_ms": budget,
        "n": len(d),
        "median_delta_eur": float(np.median(d)) if d else 0.0,
        "mean_delta_eur": float(np.mean(d)) if d else 0.0,
        "a_wins": int(sum(x < -_EPS for x in d)),  # a строго дешевле
        "b_wins": int(sum(x > _EPS for x in d)),
        "ties": int(sum(abs(x) <= _EPS for x in d)),
    }


def _win_size_profile(rows, budget) -> dict:
    """Где rl_polish обыгрывает greedy_polish — концентрация на КРУПНЫХ residual (advisor #3)?"""
    rp = {(r["seed"], r["at_min"]): r
          for r in rows if r["system"] == "rl_polish" and r["budget_ms"] == budget}
    gp = {(r["seed"], r["at_min"]): r
          for r in rows if r["system"] == "greedy_polish" and r["budget_ms"] == budget}
    keys = set(rp) & set(gp)
    win_n = [rp[k]["n_pending"] for k in keys if rp[k]["cost"] < gp[k]["cost"] - _EPS]
    all_n = [rp[k]["n_pending"] for k in keys]
    return {
        "median_n_pending_all": float(np.median(all_n)) if all_n else 0.0,
        "max_n_pending": int(max(all_n)) if all_n else 0,
        "median_n_pending_rl_wins": float(np.median(win_n)) if win_n else None,
        "n_rl_wins": len(win_n),
    }


def analyze(rows) -> dict:
    budgets = sorted({r["budget_ms"] for r in rows})
    systems = ("rl_raw", "greedy_raw", "greedy_polish", "rl_polish")

    def _mean_cost(sys, b):
        c = [r["cost"] for r in rows if r["system"] == sys and r["budget_ms"] == b]
        return float(np.mean(c)) if c else 0.0

    def _mean_lat(sys, b):
        v = [r["latency_ms"] for r in rows if r["system"] == sys and r["budget_ms"] == b]
        return float(np.median(v)) if v else 0.0

    table = {
        b: {s: {"cost_eur_mean": _mean_cost(s, b), "latency_ms_median": _mean_lat(s, b)}
            for s in systems}
        for b in budgets
    }
    # СТАРТ: опережает ли raw-RL raw-greedy ДО polish (budget-независимо — берём min budget)
    start = _paired(rows, "rl_raw", "greedy_raw", budgets[0])
    # НИША: rl_polish vs greedy_polish по бюджетам + профиль размера побед
    rl_vs_gp = {b: _paired(rows, "rl_polish", "greedy_polish", b) for b in budgets}
    profile = {b: _win_size_profile(rows, b) for b in budgets}
    # доп.: вклад polish поверх каждого старта (rl_polish vs rl_raw; greedy_polish vs greedy_raw)
    polish_gain_rl = {b: _paired(rows, "rl_polish", "rl_raw", b) for b in budgets}
    polish_gain_gr = {b: _paired(rows, "greedy_polish", "greedy_raw", b) for b in budgets}

    niche_budgets = [
        b for b in budgets
        if rl_vs_gp[b]["median_delta_eur"] < -_EPS
        and rl_vs_gp[b]["a_wins"] > rl_vs_gp[b]["b_wins"]
    ]
    start_edge = start["median_delta_eur"] < -_EPS and start["a_wins"] > start["b_wins"]
    return {
        "budgets_ms": budgets,
        "table": table,
        "start_rl_vs_greedy": start,  # a=rl_raw, b=greedy_raw
        "rl_polish_vs_greedy_polish": rl_vs_gp,
        "win_size_profile": profile,
        "polish_gain_rl": polish_gain_rl,
        "polish_gain_greedy": polish_gain_gr,
        "niche_budgets_ms": niche_budgets,
        "start_edge_rl": bool(start_edge),
    }


def _cell(t, s: str) -> str:
    return f"{t[s]['cost_eur_mean']:.1f}€/{t[s]['latency_ms_median']:.0f}мс"


def _print_report(an: dict) -> None:
    print("\n=== Ablation: budget × система (cost €, латентность мс) — 0004-харнесс ===")
    print(f"{'budget':>7} | {'rl_raw':>14} | {'greedy_raw':>14} | "
          f"{'greedy_polish':>16} | {'rl_polish':>16} | {'rl_pol vs gr_pol':>18}")
    print("-" * 100)
    for b in an["budgets_ms"]:
        t = an["table"][b]
        p = an["rl_polish_vs_greedy_polish"][b]
        verdict = f"Δ̃{p['median_delta_eur']:+.1f}€ w{p['a_wins']}/l{p['b_wins']}/t{p['ties']}"
        print(f"{b:6.0f}м | {_cell(t, 'rl_raw'):>14} | {_cell(t, 'greedy_raw'):>14} | "
              f"{_cell(t, 'greedy_polish'):>16} | {_cell(t, 'rl_polish'):>16} | {verdict:>18}")
    print("-" * 100)
    s = an["start_rl_vs_greedy"]
    edge = "RL-старт лучше" if an["start_edge_rl"] else "RL-старт НЕ лучше greedy"
    print(f"СТАРТ (до polish): rl_raw vs greedy_raw — Δ̃ {s['median_delta_eur']:+.1f}€, "
          f"rl-wins {s['a_wins']}/{s['n']} → {edge}")
    for b in an["budgets_ms"]:
        pr = an["win_size_profile"][b]
        mn = pr["median_n_pending_rl_wins"]
        print(f"  @{b:.0f}мс: rl_polish-побед {pr['n_rl_wins']} "
              f"(медиана n_pending побед={mn if mn is None else round(mn, 1)}, "
              f"всего медиана n={pr['median_n_pending_all']:.1f}, max n={pr['max_n_pending']})")
    if an["niche_budgets_ms"]:
        print(f"ВЕРДИКТ: ниша ЕСТЬ на бюджетах {an['niche_budgets_ms']}мс "
              f"(rl_polish устойчиво < greedy_polish: median Δ<0 И wins>losses).")
    else:
        print("ВЕРДИКТ: ниши НЕТ в достижимом режиме — polish выравнивает старты на ВСЕХ "
              "бюджетах {50,100,200,500}мс (median Δ≥0 либо wins≤losses). "
              "Открытый вопрос 0009 закрыт отрицательно для этого харнесса.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 6b Ablation — латентная ниша RL")
    ap.add_argument("--ckpt", type=str, default=str(_CKPT))
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--events", type=int, default=6)
    ap.add_argument("--out", type=str, default=str(_OUT))
    args = ap.parse_args()

    ckpt = Path(args.ckpt)
    torch.manual_seed(0)
    policy = rd._load_policy(ckpt)

    rows: list[dict] = []
    for seed, ev, res, travel, fleet in rd.iter_events(range(args.seeds), n_events=args.events):
        rows += measure_event(policy, seed, ev, res, travel, fleet)
    an = analyze(rows)
    _print_report(an)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "phase": "6b-ablation-latency-niche",
        "config": {
            "ckpt": str(ckpt),
            "seeds": list(range(args.seeds)),
            "n_events": args.events,
            "budgets_ms": list(_BUDGETS_MS),
            "fleet_size": im.FLEET_SIZE,
            "vehicle_cap": im.VEHICLE_CAP,
            "t_max_min": im.T_MAX_MIN,
            "delivery_weekday": im.DELIVERY_WEEKDAY,
            "congestion_tag": cg.CALIBRATION_TAG,
        },
        "provenance": rd._provenance(ckpt),
        "analysis": an,
        "records": rows,
        "note": "cost *_polish систем budget-bound → wall-clock-dependent (сколько ходов влезло до "
        "дедлайна = как быстро железо; тот же seed → другой маршрут, как OR-Tools в 0004). "
        "Детерминированы raw-старты (decode/greedy). Единый скорер evaluate_solution под тем же "
        "travel. Бюджет end-to-end (decode+polish); превышение = дедлайн-снимок. Вне git (№1).",
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"\nсводка → {out} ({len(rows)} записей)")


if __name__ == "__main__":
    main()
