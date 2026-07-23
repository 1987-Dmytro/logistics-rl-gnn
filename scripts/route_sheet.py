"""Phase 8 — route sheet: человекочитаемый план из ТОГО ЖЕ eval-пайплайна, что system_metrics.

Читает config из results/system_metrics.json, гоняет полир. портфель (`system_routes`) на seed из
конфига (дефолт 0), ASSERT-ит cost == per_seed_cost_eur[seed] (лист описывает РОВНО тот план, что в
таблицах — запреты №3/№4). Рендерит docs/route_sheet.md (шапка + пер-машину + приложение динамики) и
машиночитаемый results/route_sheet.json (тест сверяет парити без прогона солвера).

Имена аптек — опционально из data/snapshots/<snap>/names.parquet (scripts/enrich_names.py); нет →
stop-id. Приложение «динамика»: харнесс 0004 (event_stream/residual_instance/served_by), seed 0,
первое traffic-событие → RL re-plan → diff перераспределённых стопов (старое/новое прибытие) +
латентность forward-pass. Детерминирован (torch.manual_seed(0), фикс. seed/config).

Запуск: python scripts/route_sheet.py [--seed N]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
import eval_system as es  # noqa: E402
import run_dynamic as rd  # noqa: E402

from logistics_rl_gnn.baselines.greedy import greedy_routes  # noqa: E402
from logistics_rl_gnn.config import instance as im  # noqa: E402
from logistics_rl_gnn.env.events import (  # noqa: E402
    DynamicState,
    congestion_for,
    event_stream,
    make_dynamic_env,
    residual_instance,
    served_by,
)
from logistics_rl_gnn.env.scoring import CostConfig, evaluate_solution  # noqa: E402

_SM = Path("results/system_metrics.json")
_MD = Path("docs/route_sheet.md")
_JSON = Path("results/route_sheet.json")
_CFG = CostConfig()
_DEPOT_ADDR = "Benzstraße 10, 86391 Stadtbergen"
_WEEKDAY_RU = ("понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье")
_DUR_MEDIAN_MS = 14.4  # durable медиана RL forward-pass re-plan (dynamic.json, dec-0004)
_LAT_GREEDY_MS = 5.9  # durable медиана greedy re-plan — БЫСТРЕЕ сырого RL (dynamic.json → нет ниши)
_LAT_SYSTEM_MS = 689  # деплой-система (портфель+polish) реакция; ×2.9 к OR, сопост. качество (0009)
_LAT_OR_MS = 2001  # OR-Tools re-solve (dynamic.json)


# ---------- имена аптек (опц., аддитивно) ----------


def load_names(snap_dir) -> dict[int, tuple[str, str | None]]:
    """{snapshot_stop: (name, addr)} из names.parquet; нет файла → {} (фолбэк на stop-id)."""
    p = Path(snap_dir) / "names.parquet"
    if not p.exists():
        return {}
    df = pd.read_parquet(p).set_index("stop")
    out: dict[int, tuple[str, str | None]] = {}
    for stop, row in df.iterrows():
        nm = row.get("name")
        if pd.isna(nm) or not str(nm).strip():
            continue
        addr = row.get("addr")
        out[int(stop)] = (str(nm), None if pd.isna(addr) else str(addr))
    return out


def _label(snap: int, names: dict) -> str:
    return names.get(snap, (None,))[0] or f"stop #{snap}"


# ---------- walk (единая семантика с evaluate_solution) ----------


def walk_route(route, inst, travel=None):
    """Пер-стоп timeline одного маршрута (arrival/wait/service/load) — та же прогулка, что
    `evaluate_solution`. travel=None → free-flow (статика); travel=CongestionTravel → динамика
    (dt = travel.time(a,b,t), t мин от старта тура). -> (stops, totals)."""
    time_m = inst.time_matrix / 60.0
    dist_km = inst.dist_matrix / 1000.0
    win = inst.windows / 60.0
    svc = inst.service / 60.0
    dem = inst.demand
    t = load = drive = wait_tot = svc_tot = km = 0.0
    stops: list[dict] = []
    ret = None
    for a, b in zip(route[:-1], route[1:], strict=False):
        dt = float(travel.time(a, b, t)) if travel is not None else float(time_m[a, b])
        leg_km = float(dist_km[a, b])
        km += leg_km
        drive += dt
        arrival = t + dt
        if b == 0:  # возврат в депо: без окна/сервиса
            ret = {"clock_min": arrival, "leg_km": leg_km, "leg_min": dt}
            t = arrival
            continue
        w = max(0.0, win[b, 0] - arrival)
        load += float(dem[b])
        stops.append({
            "n": int(b), "snap": int(inst.snapshot_stops[b]), "arr_min": arrival,
            "e_min": float(win[b, 0]), "l_min": float(win[b, 1]), "wait": w,
            "service": float(svc[b]), "load_after": int(load), "leg_km": leg_km, "leg_min": dt,
            "tw_source": inst.tw_source[b],
        })
        wait_tot += w
        svc_tot += float(svc[b])
        t = arrival + w + float(svc[b])
    totals = {"km": km, "drive": drive, "wait": wait_tot, "service": svc_tot,
              "boxes": int(load), "n_stops": len(stops), "ret": ret}
    return stops, totals


def build_sheet(routes, inst, travel=None) -> dict:
    """Структурный лист: непустые маршруты → машины + гранд-итоги + cost (== evaluate_solution)."""
    vehicles = []
    for route in routes:
        if len(route) < 2 or all(n == 0 for n in route):
            continue  # пустая машина ([0]/[0,0]) — как в scorer
        stops, tot = walk_route(route, inst, travel)
        vehicles.append({"stops": stops, "totals": tot})
    g_km = sum(v["totals"]["km"] for v in vehicles)
    g_time = sum(v["totals"]["drive"] + v["totals"]["wait"] + v["totals"]["service"]
                 for v in vehicles)
    cost = (_CFG.c_f * len(vehicles) + _CFG.c_d * g_km + _CFG.c_t * g_time / 60.0)
    n_visits = sum(len(v["stops"]) for v in vehicles)
    on_time = sum(1 for v in vehicles for s in v["stops"] if s["arr_min"] <= s["l_min"] + 1e-9)
    return {
        "vehicles": vehicles,
        "totals": {
            "vehicles_used": len(vehicles),
            "n_stops": n_visits,
            "boxes": sum(v["totals"]["boxes"] for v in vehicles),
            "km": g_km, "time_min": g_time,
            "drive": sum(v["totals"]["drive"] for v in vehicles),
            "wait": sum(v["totals"]["wait"] for v in vehicles),
            "service": sum(v["totals"]["service"] for v in vehicles),
            # честный on-time из walk (НЕ хардкод): доля стопов с arrival ≤ l_i
            "on_time_pct": 100.0 if n_visits == 0 else 100.0 * on_time / n_visits,
        },
        "cost_eur": cost,
    }


# ---------- рендер ----------


def _clock(base_dt, minutes: float) -> str:
    return (base_dt + timedelta(minutes=float(minutes))).strftime("%H:%M")


def _win(base_dt, e_min, l_min) -> str:
    return f"[{_clock(base_dt, e_min)}–{_clock(base_dt, l_min)}]"


def render_md(sheet, inst, names, dyn=None, *, seed: int, cost_anchor: float) -> str:
    """dyn=None → статик-лист без приложения динамики (переиспользуется demo.py, у него своя
    живая portfolio-re-plan-история в терминале + plan_after.html)."""
    base = inst.start_datetime
    T = sheet["totals"]
    L = [
        "# Route sheet — план доставки в аптеки Аугсбурга",
        "",
        f"> Сгенерировано `scripts/route_sheet.py` из того же eval-пайплайна, что `system_metrics` "
        f"(полир. портфель). Cost парити с `results/system_metrics.json` per_seed[{seed}] = "
        f"**{cost_anchor:.1f} €** — лист описывает РОВНО тот план, что в таблицах README.",
        "",
        f"- **Дата:** {base.strftime('%Y-%m-%d')} ({_WEEKDAY_RU[base.weekday()]}), "
        f"диспетч-окно {base.strftime('%H:%M')}–{_clock(base, inst.horizon_s / 60.0)}",
        f"- **Депо:** PHOENIX, {_DEPOT_ADDR}",
        f"- **Флот:** K = {im.FLEET_SIZE} машин, вместимость Q = {im.VEHICLE_CAP} боксов, "
        f"T_max = {im.T_MAX_MIN / 60:.0f} ч/тур",
        f"- **Инстанс:** seed {seed}, {len(inst.demand) - 1} аптек "
        f"(окна: REAL {inst.tw_source.count('REAL')} / ASSUMED {inst.tw_source.count('ASSUMED')})",
        "",
        "## Итого по дню",
        "",
        "| Стопов | Боксов | Пробег | Время (маршр.) | Машин задействовано | Cost |",
        "|---:|---:|---:|---:|---:|---:|",
        f"| {T['n_stops']} | {T['boxes']} | {T['km']:.1f} км | {T['time_min'] / 60:.1f} ч "
        f"| {T['vehicles_used']} / {im.FLEET_SIZE} | **{sheet['cost_eur']:.1f} €** |",
        "",
        f"Время = езда {T['drive'] / 60:.1f} ч + ожидание окон {T['wait'] / 60:.1f} ч + "
        f"сервис {T['service'] / 60:.1f} ч. On-time {T['on_time_pct']:.0f} % (arrival ≤ l_i, "
        f"честно из walk — не хардкод).",
        "",
        "## По машинам",
    ]
    for vi, v in enumerate(sheet["vehicles"], start=1):
        tot = v["totals"]
        L += [
            "",
            f"### Машина {vi} — выезд из депо {base.strftime('%H:%M')}",
            "",
            "| № | Аптека | Прибытие | Окно | Ожид. | Сервис | Загрузка | Плечо |",
            "|---:|:--|:--:|:--:|---:|---:|---:|---:|",
        ]
        for k, s in enumerate(v["stops"], start=1):
            L.append(
                f"| {k} | {_label(s['snap'], names)} | {_clock(base, s['arr_min'])} "
                f"| {_win(base, s['e_min'], s['l_min'])} | {s['wait']:.0f}′ "
                f"| {s['service']:.0f}′ | {s['load_after']} | "
                f"{s['leg_km']:.1f} км / {s['leg_min']:.0f}′ |"
            )
        ret = tot["ret"]
        duty = (tot["drive"] + tot["wait"] + tot["service"]) / 60
        L += [
            f"| — | ↩ возврат в депо | {_clock(base, ret['clock_min'])} | — | — | — | — "
            f"| {ret['leg_km']:.1f} км / {ret['leg_min']:.0f}′ |",
            "",
            f"**Итог машины {vi}:** {tot['km']:.1f} км · duty {duty:.1f} ч "
            f"(езда {tot['drive'] / 60:.1f} / ожид {tot['wait'] / 60:.1f} / "
            f"сервис {tot['service'] / 60:.1f}) · {tot['boxes']} боксов",
        ]

    if dyn is None:  # статик-only лист (demo.py)
        L.append("")
        return "\n".join(L)

    # --- приложение: динамика ---
    L += [
        "",
        "---",
        "",
        "## Приложение — динамика (re-plan на событии)",
        "",
        f"Сценарий из харнесса dec-0004 (seed {dyn['seed']}): исполняется стартовый план под "
        f"диурнал-congestion, в **{dyn['event_clock']}** срабатывает traffic-инцидент (закрытие/"
        f"пробка на зоне рёбер у аптеки #{dyn['center_stop']}). Снимаем частичное состояние "
        f"({dyn['n_served']} стопов уже обслужены) и re-plan-им оставшиеся {dyn['n_pending']} "
        f"**сырым нейро forward-pass** (periodic re-optimization остатка от депо, dec-0001 §4; это "
        f"НЕ полир. портфель из шапки, а его быстрый старт — для иллюстрации реакции).",
        "",
        f"- **Латентность forward-pass:** {dyn['latency_ms']:.0f} мс этого прогона (медиана 5 "
        f"реплик, hardware-dependent), durable медиана харнесса **{_DUR_MEDIAN_MS} мс** — это "
        f"ПОТОЛОК скорости, но старт «quality-inferior на самом себе»: greedy и быстрее "
        f"(**~{_LAT_GREEDY_MS} мс**), и не хуже по стоимости → латентной ниши у сырого RL нет "
        f"(dec-0010). Общий бейзлайн (запрет №3) — greedy + OR-Tools, не только OR-Tools.",
        f"- **Устойчивый выигрыш — у деплой-системы** (портфель+polish, та, что дала "
        f"{cost_anchor:.0f}€): реакция **~{_LAT_SYSTEM_MS} мс** vs OR-Tools re-solve "
        f"**{_LAT_OR_MS} мс** = **×2.9** при сопоставимом качестве (dec-0009). ×~140 (14.4 мс "
        f"vs только-OR) — НЕ выигрыш системы, а сравнение без общего бейзлайна.",
        f"- **Перераспределено стопов:** {dyn['n_moved']} из {dyn['n_pending']} сменили машину "
        f"(метки машин по max-overlap → чистая перенумерация отфильтрована; часть стопов прибывает "
        f"ПОЗЖЕ — цена сырого forward-pass без polish).",
        "",
    ]
    if dyn["moved"]:
        capped = dyn["n_moved"] > len(dyn["moved"])
        cap = f" (топ-{len(dyn['moved'])} по |Δ прибытия|)" if capped else ""
        L += [
            f"Стопы, сменившие машину{cap} — старое (стартовый план) → новое (re-plan) прибытие:",
            "",
            "| Аптека | Машина было→стало | Прибытие было | Прибытие стало | Δ |",
            "|:--|:--:|:--:|:--:|---:|",
        ]
        for m in dyn["moved"]:
            L.append(
                f"| {_label(m['snap'], names)} | {m['old_veh']}→{m['new_veh']} "
                f"| {m['old_clock']} | {m['new_clock']} | {m['delta_min']:.0f}′ |"
            )
    else:
        L.append("_На этом событии переоптимизация сохранила прежние назначения машин._")
    L.append("")
    return "\n".join(L)


# ---------- приложение: динамика (харнесс 0004) ----------


_DIFF_TOP = 12  # cap строк diff-таблицы (residual re-opt тасует много → показываем топ по |Δ|)


def _match_labels(new_routes_full, old_veh) -> dict[int, int]:
    """Стабильные метки машин: каждой НОВОЙ машине — метка СТАРОЙ с макс. пересечением стопов.

    Residual re-optimization нумерует машины с нуля → без матчинга «сменили машину» = чистый
    шум перенумерации. Матч по max-overlap оставляет ТОЛЬКО реальные межмашинные переносы."""
    used: set[int] = set()
    labels: dict[int, int] = {}
    next_free = max(old_veh.values(), default=0)
    for r in sorted(range(len(new_routes_full)), key=lambda k: -len(new_routes_full[k])):
        counts: dict[int, int] = {}
        for s in new_routes_full[r]:
            ov = old_veh.get(s)
            if ov is not None and ov not in used:
                counts[ov] = counts.get(ov, 0) + 1
        if counts:
            lab = max(counts, key=lambda k: counts[k])
        else:
            next_free += 1
            lab = next_free
        used.add(lab)
        labels[r] = lab
    return labels


def dynamics_appendix(pol, inst, *, seed: int) -> dict:
    """Первое traffic-событие потока 0004 → RL re-plan остатка → diff назначений/прибытий."""
    dow = im.DELIVERY_WEEKDAY
    base = inst.start_datetime
    exec_travel = congestion_for(inst, dow=dow)
    exec_routes = greedy_routes(env=make_dynamic_env(inst, travel=exec_travel))

    ev = next(e for e in event_stream(seed, inst, dow) if e.kind == "traffic")
    state = DynamicState(inst, dow, now_min=float(ev.at_min))
    state.served = served_by(exec_routes, inst, exec_travel, ev.at_min)
    ev.apply(state)
    res = residual_instance(state)
    travel = congestion_for(res, dow=dow, offset_min=state.now_min, incidents=state.incidents)

    env = make_dynamic_env(res, travel=travel, fleet_size=state.fleet(im.FLEET_SIZE))
    # латентность: прогрев + медиана реплик (репрезентативнее cold-start; mode=greedy → тот же план)
    lat: list[float] = []
    new = None
    for r in range(6):
        t0 = time.perf_counter()
        with torch.no_grad():
            new = pol.rollout(env, mode="greedy")[0]
        if r > 0:  # первый прогон — прогрев
            lat.append((time.perf_counter() - t0) * 1000.0)
    latency_ms = float(np.median(lat))

    idx = [0] + sorted(i for i in range(1, len(inst.demand)) if i not in state.served)
    old_veh, old_arr = _assign(exec_routes, inst, exec_travel)

    # новый план: маршруты в ПОЛНОЙ нумерации (для матча меток) + прибытия (residual walk)
    new_routes_full: list[list[int]] = []
    new_arr: dict[int, float] = {}
    for route in new:
        stops, _ = walk_route(route, res, travel)
        if not stops:
            continue
        new_routes_full.append([idx[s["n"]] for s in stops])
        for s in stops:
            new_arr[idx[s["n"]]] = s["arr_min"]
    labels = _match_labels(new_routes_full, old_veh)
    new_veh = {stop: labels[r] for r, full in enumerate(new_routes_full) for stop in full}

    ev_base = base + timedelta(minutes=float(ev.at_min))
    center_stop = min(range(1, len(inst.demand)),
                      key=lambda i: abs(inst.coords[i][0] - ev.incident.center[0])
                      + abs(inst.coords[i][1] - ev.incident.center[1]))
    moved = []
    for i in new_veh:  # перепланированные (pending) стопы, сменившие машину (по стабильным меткам)
        if i in old_veh and new_veh[i] != old_veh[i]:
            new_abs = float(ev.at_min) + new_arr[i]
            moved.append({
                "snap": int(inst.snapshot_stops[i]),
                "old_veh": old_veh[i], "new_veh": new_veh[i],
                "old_clock": _clock(base, old_arr[i]), "new_clock": _clock(base, new_abs),
                "delta_min": abs(new_abs - old_arr[i]),
            })
    moved.sort(key=lambda m: -m["delta_min"])  # топ по сдвигу прибытия
    return {
        "seed": seed, "event_clock": ev_base.strftime("%H:%M"),
        "center_stop": int(inst.snapshot_stops[center_stop]),
        "n_served": len(state.served), "n_pending": len(res.demand) - 1,
        "latency_ms": latency_ms, "n_moved": len(moved), "moved": moved[:_DIFF_TOP],
    }


def _assign(routes, inst, travel, remap=None):
    """(veh[i]=номер машины 1.., arr[i]=прибытие мин) по стопам непустых маршрутов. remap (список
    residual→full) ремапит ключи, если routes/inst в residual-нумерации; None → как есть."""
    veh: dict[int, int] = {}
    arr: dict[int, float] = {}
    vi = 0
    for route in routes:
        if len(route) < 2 or all(n == 0 for n in route):
            continue
        vi += 1
        stops, _ = walk_route(route, inst, travel)
        for s in stops:
            key = remap[s["n"]] if remap is not None else s["n"]
            veh[key] = vi
            arr[key] = s["arr_min"]
    return veh, arr


# ---------- main ----------


def main() -> None:
    sm = json.loads(_SM.read_text())
    cfg = sm["config"]
    ap = argparse.ArgumentParser(description="Phase 8 — route sheet (парити system_metrics)")
    ap.add_argument("--seed", type=int, default=0, help="seed инстанса (per_seed[seed] — якорь)")
    ap.add_argument("--tol", type=float, default=0.5, help="парити-допуск к per_seed[seed], €")
    ap.add_argument("--out-md", default=str(_MD))
    ap.add_argument("--out-json", default=str(_JSON))
    args = ap.parse_args()

    ckpt = Path(cfg["ckpt"])
    anchor = float(sm["per_seed_cost_eur"][args.seed])
    torch.manual_seed(0)
    pol = rd._load_policy(ckpt)
    inst = im.generate_instance(seed=args.seed)
    routes = es.system_routes(
        pol, inst, budget_ms=cfg["budget_ms"], k_samples=cfg["k_samples"],
        temp=cfg["temperature"], rl_starts=cfg["rl_starts"],
    )
    sheet = build_sheet(routes, inst)

    # ПАРИТИ-СТРАЖ: cost листа == evaluate_solution == durable per_seed[seed] (запреты №3/№4)
    q = evaluate_solution(routes, inst, _CFG)
    assert abs(sheet["cost_eur"] - (-q["reward"])) < 1e-6, "walk-cost != evaluate_solution"
    assert abs(sheet["cost_eur"] - anchor) < args.tol, (
        f"ПАРИТИ FAIL: лист {sheet['cost_eur']:.3f}€ != per_seed[{args.seed}] {anchor:.3f}€ "
        f"(|Δ|={abs(sheet['cost_eur'] - anchor):.3f} > {args.tol}) — НЕ тот план"
    )
    # лист заявляет on-time в шапке → верифицируем честно из walk (система: means.on_time_pct=100)
    assert sheet["totals"]["on_time_pct"] == 100.0, (
        f"ON-TIME FAIL: {sheet['totals']['on_time_pct']:.1f}% < 100 — есть опоздание, "
        f"шапка листа врёт про on-time"
    )

    snap = im._latest_snapshot_dir()
    names = load_names(snap)
    dyn = dynamics_appendix(pol, inst, seed=args.seed)

    md = render_md(sheet, inst, names, dyn, seed=args.seed, cost_anchor=anchor)
    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text(md, encoding="utf-8")
    # машиночитаемый артефакт (тест сверяет парити без прогона солвера); вне git (запрет №1)
    Path(args.out_json).write_text(json.dumps({
        "seed": args.seed, "cost_eur": sheet["cost_eur"], "anchor_per_seed": anchor,
        "totals": sheet["totals"], "names_present": bool(names),
        "dynamics": {k: dyn[k] for k in ("seed", "event_clock", "n_served", "n_pending",
                                         "latency_ms", "center_stop", "n_moved")},
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"→ {args.out_md}  (cost {sheet['cost_eur']:.1f}€ = per_seed[{args.seed}] "
          f"{anchor:.1f}€ ✓, {sheet['totals']['vehicles_used']} машин, "
          f"{sheet['totals']['n_stops']} стопов, имена {'да' if names else 'нет (stop-id)'})")
    print(f"→ {args.out_json}")


if __name__ == "__main__":
    main()
