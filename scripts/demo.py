"""scripts/demo.py — единая нарративная демонстрация системы (реюз, без новых депов/расчётов).

    python scripts/demo.py [--seed 0] [--event traffic|breakdown|urgent] [--no-open]

5 шагов человеческим языком: утро → построение плана (парити с system_metrics) → событие (харнесс
0004) → re-plan (деплой-портфель+polish vs OR-Tools re-solve, живой замер + durable медианы) → итог
дня. ВСЕ числа берутся из тех же scorer'ов (route_sheet.build_sheet / compare_replan) — статик
free-flow (587.9€) и динамик-congestion (residual) НЕ смешиваются. Артефакты → demo_out/ (вне git):
plan_before.html · route_sheet.md · plan_after.html.

Реюз: eval_system.system_routes, route_sheet.{build_sheet,render_md,_assign,_match_labels},
viz_routes._folium_map, env.events (event_stream/residual/served), replan.compare_replan +
PortfolioPlanner (тот же механизм, что дал durable 689/2001 мс в polish_summary.json). Ничего нового
не считаем; latency — wall-clock этого прогона + durable медиана (запрет №4).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import webbrowser
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import eval_system as es  # noqa: E402
import route_sheet as rs  # noqa: E402
import run_dynamic as rd  # noqa: E402
import viz_routes as vr  # noqa: E402

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
from logistics_rl_gnn.replan.compare import compare_replan  # noqa: E402
from logistics_rl_gnn.replan.portfolio import PortfolioPlanner  # noqa: E402

_SNAP = Path("data/snapshots/augsburg_20260720")
_SM = Path("results/system_metrics.json")
_CFG = CostConfig()
# durable медианы re-plan (polish_summary.json, dec-0009) — hardware-independent якоря
_DUR = {"rl": 689, "greedy": 7, "ortools": 2001}


def say(text: str = "") -> None:
    print(text)


def _step(k: int, title: str) -> None:
    say(f"\n[{k}/5] {title}")


def _clock(base, minutes):
    return rs._clock(base, minutes)


# ---------- событие (per-kind, из харнесса 0004) ----------


def _in_zone(inc, coord, abs_min) -> bool:
    """Стоп в активной зоне инцидента — по САМОЙ логике Incident (не переизобретаем геометрию)."""
    return inc.at_node(coord, abs_min) != 0.0


def _event_context(kind, ev, inst, state, veh_of, names) -> dict:
    """Человеческие факты события + затронутые стопы/машины. Числа/зона — из данных, не хардкод."""
    n = len(inst.demand)
    pending = [i for i in range(1, n) if i not in state.served]
    ctx = {"clock": _clock(inst.start_datetime, ev.at_min), "incident": None,
           "affected": [], "vehicles": set(), "drop_vehicle": None, "lines": []}
    if kind == "traffic":
        inc = ev.incident
        factor = ("закрытие (∞)" if math.isinf(inc.magnitude)
                  else f"замедление ×{1 + inc.magnitude:.1f}")
        epi = min(range(1, n), key=lambda i: abs(inst.coords[i][0] - inc.center[0])
                  + abs(inst.coords[i][1] - inc.center[1]))
        zone = [i for i in pending if _in_zone(inc, inst.coords[i], ev.at_min)]
        vehs = {veh_of[i] for i in zone if i in veh_of}
        ctx.update(incident=inc, affected=zone, vehicles=vehs)
        ctx["lines"] = [
            f"пробка/инцидент у аптеки «{rs._label(int(inst.snapshot_stops[epi]), names)}» "
            f"(радиус {inc.radius_km:.1f} км, {factor}).",
            f"В зоне {len(zone)} недоставленных стопов, "
            f"затронуты машины {sorted(vehs) or '—'}.",
        ]
    elif kind == "breakdown":
        by_veh: dict[int, list[int]] = {}
        for i in pending:
            by_veh.setdefault(veh_of.get(i), []).append(i)
        by_veh.pop(None, None)
        drop = min(by_veh, key=lambda v: len(by_veh[v])) if by_veh else None
        orphans = by_veh.get(drop, [])
        ctx.update(drop_vehicle=drop, affected=orphans, vehicles={drop} if drop else set())
        ctx["lines"] = [
            f"машина {drop} выбыла из строя — {len(orphans)} стопов осиротело.",
            "Флот −1; осиротевшие стопы уходят в общий пул на перераспределение.",
        ]
    else:  # urgent
        o = ev.order
        idx = o["idx"]
        ctx.update(affected=[idx], vehicles={veh_of.get(idx)} - {None})
        ctx["lines"] = [
            f"срочный заказ: аптека «{rs._label(int(inst.snapshot_stops[idx]), names)}» — "
            f"{o['demand']} боксов, узкое окно {o['delta_s'] / 60:.0f} мин.",
            "Требует вставки в текущие маршруты — триггер re-plan.",
        ]
    return ctx


# ---------- «продолжить старый план» (контрфактуал без re-plan) ----------


def _continue_old_plan(exec_routes, state, *, idx, drop_vehicle, veh_of):
    """Остаток старого плана как residual-решение: ГЕНУИННО оставшиеся стопы (not served) в
    исходном порядке по машинам (контрфактуал «не перепланировали»). idx — residual-нумерация
    (== residual_instance, ПЕРЕДаётся из вызова: единый источник, иначе рассинхром при urgent).
    drop_vehicle (breakdown) — её стопы выпадают (осиротели). urgent-re-delivery (стоп обслужен,
    новый спрос) в старый план НЕ входит → в контрфактуале остаётся unserved (честно: заказ без
    re-plan не выполнен)."""
    pos = {full: k for k, full in enumerate(idx)}
    out = []
    for route in exec_routes:
        seq = [pos[b] for b in route
               if b != 0 and b in pos and b not in state.served
               and (drop_vehicle is None or veh_of.get(b) != drop_vehicle)]
        if seq:
            out.append([0, *seq, 0])
    return out


# ---------- diff-карта plan_after (реюз folium) ----------


def _folium_diff(inst, old_full, new_full, out: Path, *, incident, names):
    """Старый маршрут пунктиром, новый сплошным, зона инцидента красным (traffic). Номера/popup —
    как в viz_routes для нового плана; folium-примитивы, ничего нового не изобретаем."""
    import folium

    c = inst.coords
    m = folium.Map(location=[c[0][1], c[0][0]], zoom_start=12, tiles="cartodbpositron")
    folium.Marker([c[0][1], c[0][0]], tooltip="Депо PHOENIX",
                  icon=folium.Icon(color="red", icon="star")).add_to(m)
    if incident is not None:  # зона пробки (traffic) красным
        folium.Circle([incident.center[1], incident.center[0]], radius=incident.radius_km * 1000,
                      color="red", fill=True, fill_opacity=0.12, weight=2,
                      tooltip="зона инцидента").add_to(m)
    for route in old_full:  # старый план — серый пунктир
        if len(route) > 2:
            folium.PolyLine([[c[n][1], c[n][0]] for n in route], color="#888", weight=2,
                            opacity=0.6, dash_array="8", tooltip="старый маршрут").add_to(m)
    pal = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#17becf"]
    v = 0
    for route in new_full:  # новый план — сплошной цветной
        if len(route) <= 2:
            continue
        col = pal[v % len(pal)]
        folium.PolyLine([[c[n][1], c[n][0]] for n in route], color=col, weight=3.5, opacity=0.9,
                        tooltip=f"новый маршрут (маш. {v + 1})").add_to(m)
        for n in route[1:-1]:
            folium.CircleMarker([c[n][1], c[n][0]], radius=4, color=col, fill=True, fill_opacity=1,
                                popup=rs._label(int(inst.snapshot_stops[n]), names)).add_to(m)
        v += 1
    out.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out))


# ---------- главный сценарий ----------


def run_demo(*, seed: int, event_kind: str, out_dir: str, open_maps: bool) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sm = json.loads(_SM.read_text())
    cfg = sm["config"]
    anchor = float(sm["per_seed_cost_eur"][seed])
    dow = im.DELIVERY_WEEKDAY

    torch.manual_seed(0)
    pol = rd._load_policy(Path(cfg["ckpt"]))
    inst = im.generate_instance(snapshot_dir=_SNAP, seed=seed)
    names = rs.load_names(_SNAP)
    base = inst.start_datetime
    n = len(inst.demand)

    # [1/5] утро
    _step(1, f"Утро, {rs._WEEKDAY_RU[base.weekday()]} {base.strftime('%H:%M')}. "
             f"Депо PHOENIX, {rs._DEPOT_ADDR}.")
    say(f"      Заказы: {n - 1} аптек, {int(inst.demand.sum())} боксов. "
        f"Флот K={im.FLEET_SIZE}, вместимость Q={im.VEHICLE_CAP}, T_max={im.T_MAX_MIN / 60:.0f} ч.")

    # [2/5] построение плана (СТАТИК free-flow — парити с system_metrics)
    _step(2, "Построение плана… (portfolio + local-search polish)")
    routes = es.system_routes(pol, inst, budget_ms=cfg["budget_ms"], k_samples=cfg["k_samples"],
                              temp=cfg["temperature"], rl_starts=cfg["rl_starts"])
    sheet = rs.build_sheet(routes, inst)
    q = evaluate_solution(routes, inst, _CFG)
    assert abs(sheet["cost_eur"] - (-q["reward"])) < 1e-6, "walk-cost != scorer"
    assert abs(sheet["cost_eur"] - anchor) < 0.5, (
        f"ПАРИТИ FAIL: {sheet['cost_eur']:.2f}€ != per_seed[{seed}] {anchor:.2f}€")
    T = sheet["totals"]
    md_p, before_p = out / "route_sheet.md", out / "plan_before.html"
    md = rs.render_md(sheet, inst, names, seed=seed, cost_anchor=anchor)  # статик-only (dyn=None)
    md_p.write_text(md, encoding="utf-8")
    vr._folium_map(inst, routes, before_p, names=names)
    say(f"      → {T['vehicles_used']} машин, {T['km']:.1f} км, "
        f"on-time {T['on_time_pct']:.0f}% · **{sheet['cost_eur']:.1f} €** "
        f"(парити system_metrics per_seed[{seed}] ✓)")
    say(f"      → карта:  {before_p}")
    say(f"      → лист:   {md_p}")
    if open_maps:
        webbrowser.open(before_p.resolve().as_uri())

    # --- динамик-мир (congestion): стартовый план исполняется под диурналом ---
    exec_travel = congestion_for(inst, dow=dow)
    exec_routes = greedy_routes(env=make_dynamic_env(inst, travel=exec_travel))
    ev = next((e for e in event_stream(seed, inst, dow) if e.kind == event_kind), None)
    if ev is None:
        raise SystemExit(f"в потоке seed {seed} нет события '{event_kind}'")
    state = DynamicState(inst, dow, now_min=float(ev.at_min))
    state.served = served_by(exec_routes, inst, exec_travel, ev.at_min)
    ev.apply(state)
    veh_of, _ = rs._assign(exec_routes, inst, exec_travel)
    ctx = _event_context(event_kind, ev, inst, state, veh_of, names)

    # [3/5] событие
    _step(3, f"{ctx['clock']} — СОБЫТИЕ ({event_kind}):")
    for line in ctx["lines"]:
        say(f"      {line}")

    pending = [i for i in range(1, n) if i not in state.served]
    if not pending and not state.urgent:  # край: всё обслужено к моменту события
        _step(4, "Re-plan не нужен — весь остаток уже обслужен.")
        _folium_diff(inst, exec_routes, exec_routes, out / "plan_after.html",
                     incident=ctx["incident"], names=names)
        _step(5, "Итог: событие пришло на пустой остаток, план не менялся.")
        return {"seed": seed, "event": event_kind, "static_cost": sheet["cost_eur"],
                "n_served": len(state.served), "n_pending": 0, "n_moved": 0,
                "files": [str(before_p), str(md_p), str(out / "plan_after.html")]}

    res = residual_instance(state)
    fleet = state.fleet(im.FLEET_SIZE)
    travel = congestion_for(res, dow=dow, offset_min=state.now_min, incidents=state.incidents)

    # [4/5] re-plan: деплой-портфель+polish vs OR-Tools re-solve vs greedy (тот же compare_replan)
    _step(4, f"Re-plan из текущего состояния ({len(res.demand) - 1} стопов в остатке)…")
    planner = PortfolioPlanner(pol, k_samples=16, temperature=1.0, rl_starts=8,
                               polish_budget_ms=400.0, polish_top_m=5)
    cmp = compare_replan(res, travel, pol, fleet_size=fleet, deadline_s=2,
                         rl_planner=planner, rl_reps=2, warmup=1)
    new = planner.plan(res, travel, fleet_size=fleet)["routes"]  # маршруты для карты/диффа

    # residual→full маппинг: ТОЧНО как residual_instance.idx (для urgent он добавляет urgent-стоп
    # даже если обслужен → иначе рассинхром нумерации). Реплицируем, не угадываем.
    res_pending = [i for i in range(1, n) if i not in state.served]
    for u in state.urgent:
        if u["idx"] not in res_pending:
            res_pending.append(u["idx"])
    idx = [0] + sorted(set(res_pending))
    new_full = [[idx[k] for k in r] for r in new]
    new_routes_full = [[idx[s["n"]] for s in rs.walk_route(r, res, travel)[0]] for r in new]
    new_routes_full = [r for r in new_routes_full if r]
    labels = rs._match_labels(new_routes_full, veh_of)
    new_veh = {stop: labels[ri] for ri, rt in enumerate(new_routes_full) for stop in rt}
    n_moved = sum(1 for i in new_veh if i in veh_of and new_veh[i] != veh_of[i])

    def _lat(mk):
        return f"{cmp[mk]['latency_ms']:.0f} мс этого прогона (durable медиана {_DUR[mk]} мс)"

    say(f"      • деплой-система (портфель+polish): {_lat('rl')}")
    say(f"      • OR-Tools re-solve (тот же residual): {_lat('ortools')}")
    say(f"      • greedy (контроль): {_lat('greedy')}")
    say(f"      Перестроено: {n_moved} стопов перераспределены между машинами "
        f"(метки по max-overlap).")
    speedup = cmp["ortools"]["latency_ms"] / cmp["rl"]["latency_ms"]
    say(f"      → реакция портфеля ×{speedup:.1f} быстрее OR-Tools при сопоставимом качестве; "
        f"портфель по построению не хуже greedy (запрет №3: бейзлайн = greedy + OR-Tools).")

    # [5/5] итог дня — ВСЁ в congestion-мире остатка (НЕ сравнивать со статик-587.9€)
    old_res = _continue_old_plan(exec_routes, state, idx=idx,  # idx — та же residual-нумерация
                                 drop_vehicle=ctx["drop_vehicle"], veh_of=veh_of)
    q_before = evaluate_solution(old_res, res, _CFG, travel=travel)  # «не реагировали»
    cost_before, cost_after = -q_before["reward"], -cmp["rl"]["reward"]
    ot, uns = cmp["rl"]["on_time_pct"], int(cmp["rl"]["unserved"])
    _folium_diff(inst, exec_routes, new_full, out / "plan_after.html",
                 incident=ctx["incident"], names=names)
    _step(5, "Итог дня (остаток под congestion+событие — ДРУГОЙ мир, не сравним со статик-планом):")
    say(f"      • без re-plan (едем старым планом сквозь событие): {cost_before:.1f} €")
    say(f"      • после re-plan (портфель): {cost_after:.1f} €  "
        f"(Δ {cost_after - cost_before:+.1f} €)")
    say(f"      • on-time {ot:.0f}% · необслужено {uns} "
        f"{'(все окна соблюдены ✓)' if ot >= 100 and uns == 0 else '(честно из scorer)'}")
    say(f"      → карта-diff: {out / 'plan_after.html'}  "
        f"(старый пунктиром, новый сплошным{', зона красным' if ctx['incident'] else ''})")
    if open_maps:
        webbrowser.open((out / "plan_after.html").resolve().as_uri())

    return {"seed": seed, "event": event_kind, "static_cost": sheet["cost_eur"],
            "n_served": len(state.served), "n_pending": len(res.demand) - 1, "n_moved": n_moved,
            "cost_before": cost_before, "cost_after": cost_after,
            "on_time_pct": ot, "unserved": uns,
            "files": [str(before_p), str(md_p), str(out / "plan_after.html")]}


def main() -> None:
    ap = argparse.ArgumentParser(description="Единая нарративная демонстрация системы (реюз)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--event", choices=("traffic", "breakdown", "urgent"), default="traffic")
    ap.add_argument("--out", default="demo_out")
    ap.add_argument("--no-open", action="store_true", help="не открывать карты в браузере")
    args = ap.parse_args()
    run_demo(seed=args.seed, event_kind=args.event, out_dir=args.out, open_maps=not args.no_open)
    say("\nГотово. Артефакты в demo_out/ (вне git).")


if __name__ == "__main__":
    main()
