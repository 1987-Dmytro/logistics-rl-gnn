"""Phase 7 dynamic-прогон: поток событий на M сидов → RL vs OR-Tools vs greedy по
ЛАТЕНТНОСТИ и КАЧЕСТВУ re-plan. Пишет results/dynamic.json (+ см. decision 0004).

Модель: 8 машин исполняют начальный greedy-план (параллельно от 08:00 под диурнал-congestion).
На каждом событии-триггере (traffic/breakdown/urgent) снимаем частичное состояние (served — из
timeline исполняемого плана) и re-plan-им ОСТАВШЕЕСЯ каждым методом.

Воспроизводимость (запрет №4): латентность И качество OR-Tools — wall-clock-dependent (GLS крутит
соседства до дедлайна → быстрее/свободнее железо укладывает больше итераций → ДРУГОЙ маршрут →
другой cost/on-time/unserved; проверено: тот же seed+config даёт разные исходы OR-Tools). Только
качество RL/greedy детерминировано seed+config (+ фикс. чекпойнт весов). Все числа — median/mean
на фикс. железе, привязаны к config+версиям+hash чекпойнта в dynamic.json, НЕ к абсолюту.
Запуск: python scripts/run_dynamic.py [--seeds N] [--deadline S] [--events K].
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import numpy as np
import torch

from logistics_rl_gnn.baselines.greedy import greedy_routes
from logistics_rl_gnn.config import congestion as cg
from logistics_rl_gnn.config import instance as im
from logistics_rl_gnn.env.events import (
    DynamicState,
    congestion_for,
    event_stream,
    make_dynamic_env,
    residual_instance,
    served_by,
)
from logistics_rl_gnn.models.policy import VRPPolicy
from logistics_rl_gnn.replan.compare import compare_replan

_CKPT = Path("results/policy_best.pt")
_OUT = Path("results/dynamic.json")
_METHODS = ("rl", "greedy", "ortools")


def _pkg_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _provenance(ckpt: Path) -> dict:
    """Артефакты метрики: hash RL-чекпойнта (вне git) + версии решателей (запрет №4)."""
    return {
        "checkpoint": {
            "path": str(ckpt),
            "sha256_16": hashlib.sha256(ckpt.read_bytes()).hexdigest()[:16],
        },
        "torch": torch.__version__,
        "numpy": np.__version__,
        "ortools": _pkg_version("ortools"),
        "platform": platform.platform(),
    }


def _load_policy(ckpt: Path) -> VRPPolicy:
    if not ckpt.exists():
        raise FileNotFoundError(f"нет весов {ckpt} — сначала обучение (train_pomo.py)")
    pol = VRPPolicy()
    pol.load_state_dict(torch.load(ckpt, weights_only=True))
    pol.eval()
    return pol


def run(seeds, *, deadline_s: int, n_events: int, ckpt: Path, rl_planner=None) -> dict:
    pol = _load_policy(ckpt)
    dow = im.DELIVERY_WEEKDAY
    base_k = im.FLEET_SIZE
    records: list[dict] = []
    for seed in seeds:
        inst = im.generate_instance(seed=seed)
        exec_travel = congestion_for(inst, dow=dow)  # диурнал (без инцидентов) = timeline плана
        exec_routes = greedy_routes(env=make_dynamic_env(inst, travel=exec_travel))
        state = DynamicState(inst, dow)
        for ev in event_stream(seed, inst, dow, n_events=n_events):
            state.now_min = float(ev.at_min)
            state.served = served_by(exec_routes, inst, exec_travel, ev.at_min)
            triggered = ev.apply(state)  # мутирует incidents/broken/urgent; диурнал → False
            if not triggered:
                continue
            n = len(inst.demand)
            pending = [i for i in range(1, n) if i not in state.served]
            if not pending and not state.urgent:
                continue  # всё обслужено → нечего re-plan-ить
            res = residual_instance(state)
            travel = congestion_for(
                res, dow=dow, offset_min=state.now_min, incidents=state.incidents
            )
            out = compare_replan(
                res, travel, pol, fleet_size=state.fleet(base_k), deadline_s=deadline_s,
                rl_planner=rl_planner,
            )
            for m in _METHODS:
                records.append(
                    {
                        "seed": int(seed),
                        "event": ev.kind,
                        "at_min": round(ev.at_min, 1),
                        "n_pending": len(res.demand) - 1,
                        "fleet": state.fleet(base_k),
                        "method": m,
                        **out[m],
                    }
                )
    return {"records": records}


def _agg(records, method: str) -> dict:
    r = [x for x in records if x["method"] == method]
    lat = np.array([x["latency_ms"] for x in r])
    cost = np.array([-x["reward"] for x in r])  # € (положит.)
    return {
        "n": len(r),
        "latency_ms_median": float(np.median(lat)),
        "latency_ms_mean": float(lat.mean()),
        "cost_eur_mean": float(cost.mean()),
        "on_time_pct_mean": float(np.mean([x["on_time_pct"] for x in r])),
        "unserved_mean": float(np.mean([x["unserved"] for x in r])),
    }


def _print_table(agg: dict) -> None:
    rl_lat = agg["rl"]["latency_ms_median"]
    print("\n=== Phase 7 «Стало по скорости реакции» (re-plan, mean по событиям) ===")
    print(
        f"{'метод':8s} | {'латентность (медиана)':>22s} | {'cost,€':>9s} | "
        f"{'on-time%':>8s} | {'unserved':>8s} | {'× медленнее RL':>14s}"
    )
    print("-" * 92)
    for m in _METHODS:
        a = agg[m]
        slow = a["latency_ms_median"] / rl_lat
        unit = f"{a['latency_ms_median']:8.1f} мс"
        print(
            f"{m:8s} | {unit:>22s} | {a['cost_eur_mean']:9.1f} | "
            f"{a['on_time_pct_mean']:8.1f} | {a['unserved_mean']:8.2f} | {slow:13.0f}×"
        )
    print("-" * 92)
    print(
        f"ГЛАВНЫЙ ГЕЙТ: RL-латентность {rl_lat:.1f}мс << OR-Tools "
        f"{agg['ortools']['latency_ms_median']:.0f}мс "
        f"(×{agg['ortools']['latency_ms_median'] / rl_lat:.0f})."
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3, help="число сидов (0..N-1)")
    ap.add_argument("--deadline", type=int, default=2, help="дедлайн re-solve OR-Tools, сек")
    ap.add_argument("--events", type=int, default=6, help="событий на сид")
    ap.add_argument("--ckpt", type=str, default=str(_CKPT), help="RL-чекпойнт (Шаг 2: congestion)")
    ap.add_argument("--out", type=str, default=str(_OUT), help="выходной JSON")
    args = ap.parse_args()
    if args.seeds < 1:
        ap.error("--seeds ≥ 1")

    ckpt, out = Path(args.ckpt), Path(args.out)
    seeds = list(range(args.seeds))
    res = run(seeds, deadline_s=args.deadline, n_events=args.events, ckpt=ckpt)
    agg = {m: _agg(res["records"], m) for m in _METHODS}
    _print_table(agg)

    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "seeds": seeds,
            "deadline_s": args.deadline,
            "n_events": args.events,
            "fleet_size": im.FLEET_SIZE,
            "vehicle_cap": im.VEHICLE_CAP,
            "t_max_min": im.T_MAX_MIN,
            "delivery_weekday": im.DELIVERY_WEEKDAY,
            "congestion_tag": cg.CALIBRATION_TAG,
        },
        "provenance": _provenance(ckpt),
        "aggregates": agg,
        "records": res["records"],
        "note": "латентность И качество OR-Tools — wall-clock-dependent (GLS до дедлайна → "
        "быстрее железо = другой маршрут = другой cost/on-time/unserved). Детерминировано "
        "seed+config ТОЛЬКО качество RL/greedy (+ hash чекпойнта). RL free-flow-trained → OOD.",
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"\nсводка → {out} ({len(res['records'])} записей)")


if __name__ == "__main__":
    main()
