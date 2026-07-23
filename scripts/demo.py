"""scripts/demo.py — единая нарративная демонстрация системы (реюз, без новых депов/расчётов).

    python scripts/demo.py [--seed 0] [--event traffic|breakdown|urgent] [--no-open]

5 шагов человеческим языком: утро → построение плана (парити с system_metrics) → событие (харнесс
0004) → re-plan (сцена A/B/C: do-nothing vs OR-Tools vs система, живой замер + durable медианы) →
итог дня. ВСЕ числа берутся из тех же scorer'ов (route_sheet.build_sheet / compare_replan) — статик
free-flow (587.9€, полный день, карта #1) и динамик-congestion residual (карты #2/#3) — РАЗНЫЕ миры,
НЕ смешиваются. Артефакты → demo_out/ (вне git), самоописательные имена:
  1_morning_plan.html · route_sheet.md · 2_incident_no_replan.html · 3_incident_replan.html ·
  compare.html (два iframe #2|#3 + таблица A/B/C — кадр для скринкаста).
Хопы карт — реальными улицами (nx.shortest_path по graph.graphml, кэш путей); старый план на #3 —
выключаемый пунктирный слой; зона инцидента подписана.

Реюз: eval_system.system_routes, route_sheet.{build_sheet,render_md,walk_route,_assign,
_match_labels}, env.events (event_stream/residual/served), replan.compare_replan + PortfolioPlanner
(тот же механизм, что дал durable 689/2001 мс в polish_summary.json). Ничего нового не считаем;
latency в шапках/таблице — durable медиана (запрет №4), живой wall-clock — только в логе шага 4.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import webbrowser
from pathlib import Path

import networkx as nx
import osmnx as ox
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
import eval_system as es  # noqa: E402
import route_sheet as rs  # noqa: E402
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
from logistics_rl_gnn.replan.compare import compare_replan  # noqa: E402
from logistics_rl_gnn.replan.portfolio import PortfolioPlanner  # noqa: E402

_SNAP = Path("data/snapshots/augsburg_20260720")
_SM = Path("results/system_metrics.json")
_CFG = CostConfig()
# durable медианы re-plan (polish_summary.json, dec-0009) — hardware-independent якоря
_DUR = {"rl": 689, "greedy": 7, "ortools": 2001}
_SPEEDUP = _DUR["ortools"] / _DUR["rl"]  # ×2.9 реакция система vs OR-Tools (durable)
# заголовок сцены A/B/C по типу события (клок берётся из данных, не хардкод)
_EVENT_TITLE = {"traffic": "Straßensperrung", "breakdown": "Ausfall — машина выбыла",
                "urgent": "Eilauftrag — срочный заказ"}
# палитра машин (folium) — общая для всех карт демо
_PAL = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#17becf"]


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


# ---------- дорожная геометрия (реальные улицы, кэш путей) ----------


def _stop_to_node(inst) -> dict[int, int]:
    """stop-индекс инстанса → OSM node_id в graph.graphml (через nodes.parquet, тот же снапшот)."""
    nd = pd.read_parquet(_SNAP / "nodes.parquet").set_index("stop")["node_id"].astype(int)
    return {n: int(nd[int(inst.snapshot_stops[n])]) for n in range(len(inst.demand))}


def _road_latlon(graph, na: int, nb: int, cache: dict) -> list:
    """Полилиния хопа na→nb реальными улицами (nx.shortest_path, weight=length) с кэшем. Fallback
    (нет пути/узла) — прямая по КООРДИНАТАМ УЗЛОВ графа (все вершины остаются узлами графа)."""
    key = (na, nb)
    if key not in cache:
        try:
            path = nx.shortest_path(graph, na, nb, weight="length")
            cache[key] = [(graph.nodes[p]["y"], graph.nodes[p]["x"]) for p in path]
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            cache[key] = [(graph.nodes[na]["y"], graph.nodes[na]["x"]),
                          (graph.nodes[nb]["y"], graph.nodes[nb]["x"])]
    return cache[key]


def _route_polyline(route, stop2node, graph, cache) -> list:
    """Весь маршрут [0,s1,…,0] → одна полилиния реальными улицами (без дубля узла на стыке)."""
    pts: list = []
    for a, b in zip(route, route[1:], strict=False):
        seg = _road_latlon(graph, stop2node[a], stop2node[b], cache)
        pts.extend(seg if not pts else seg[1:])
    return pts


def _num_icon(k: int, col: str):
    import folium

    return folium.DivIcon(html=(
        f'<div style="font-size:10px;color:#fff;background:{col};border-radius:50%;width:18px;'
        f'height:18px;text-align:center;line-height:18px;border:1px solid #333">{k}</div>'),
        icon_size=(18, 18), icon_anchor=(9, 9))


# ---------- карта демо (шапка + реальные улицы + слой старого плана + зона) ----------


def _render_map(inst, primary, out: Path, *, graph, stop2node, cache, names, price, price_val,
                title, caption, banner_color, incident=None, old_routes=None, show_eta=False):
    """Одна карта демо: floating-шапка (крупная цена + титул + what-you-see), депо, зона инцидента
    (подписана), опц. выключаемый пунктирный слой «старый план», основной план сплошными реальными
    улицами + пронумерованные стопы (popup: имя; ETA/окно только на #1 free-flow). price_val —
    машиночитаемая цена в шапке (data-demo-price) для страж-теста «числа шапок == demo-выводу»."""
    import folium

    c = inst.coords
    base = inst.start_datetime
    m = folium.Map(location=[c[0][1], c[0][0]], zoom_start=12, tiles="cartodbpositron")
    m.get_root().html.add_child(folium.Element(  # шапка; сдвигаем zoom-контролы из-под неё
        '<style>.leaflet-top{top:76px}</style>'
        f'<div style="position:fixed;top:0;left:0;right:0;z-index:9999;background:{banner_color};'
        'color:#fff;padding:8px 16px;font-family:system-ui,-apple-system,sans-serif;'
        'box-shadow:0 2px 8px rgba(0,0,0,.35)">'
        f'<span data-demo-price="{price_val:.6f}" style="font-size:22px;font-weight:800">{price}'
        f'</span><span style="font-size:15px;font-weight:600;margin-left:12px">{title}</span>'
        f'<div style="font-size:12px;opacity:.92;margin-top:2px">{caption}</div></div>'))
    folium.Marker([c[0][1], c[0][0]], tooltip="Депо PHOENIX",
                  icon=folium.Icon(color="red", icon="star")).add_to(m)
    if incident is not None:  # зона инцидента красным + подпись
        folium.Circle([incident.center[1], incident.center[0]], radius=incident.radius_km * 1000,
                      color="red", fill=True, fill_opacity=0.12, weight=2,
                      tooltip=f"зона инцидента (r={incident.radius_km:.1f} км)").add_to(m)
        folium.Marker([incident.center[1], incident.center[0]], icon=folium.DivIcon(
            html='<div style="font-size:11px;color:#c00;font-weight:700;white-space:nowrap;'
                 'transform:translate(-50%,-24px)">🚧 Sperrung</div>')).add_to(m)
    if old_routes:  # старый план — выключаемый пунктирный слой (off)
        fg = folium.FeatureGroup(name="старый план (без re-plan)", show=False)
        for route in old_routes:
            if len(route) > 2:
                folium.PolyLine(_route_polyline(route, stop2node, graph, cache), color="#777",
                                weight=2.5, opacity=0.7, dash_array="6").add_to(fg)
        fg.add_to(m)
    v = 0
    for route in primary:
        if len(route) <= 2:
            continue
        col = _PAL[v % len(_PAL)]
        folium.PolyLine(_route_polyline(route, stop2node, graph, cache), color=col, weight=4,
                        opacity=0.9, tooltip=f"маш. {v + 1}").add_to(m)
        stops, _ = rs.walk_route(route, inst)
        for k, s in enumerate(stops, start=1):
            eta = ""
            if show_eta:
                eta = (f"<br>ETA {rs._clock(base, s['arr_min'])} · окно "
                       f"{rs._clock(base, s['e_min'])}–{rs._clock(base, s['l_min'])}")
            popup = folium.Popup(f"<b>{k}. {rs._label(s['snap'], names)}</b><br>маш. {v + 1}{eta}",
                                 max_width=260)
            folium.Marker([c[s["n"]][1], c[s["n"]][0]], popup=popup,
                          icon=_num_icon(k, col)).add_to(m)
        v += 1
    if old_routes:
        folium.LayerControl(collapsed=False).add_to(m)
    out.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out))


def _write_compare(out: Path, *, left: str, right: str, scene_title: str, rows: list,
                   takeaway: str):
    """compare.html — кадр для скринкаста: общий заголовок «Дилемма диспетчера» + таблица A/B/C
    (все три стоимости в ОДНОМ residual-мире + латентность) + два iframe (#2 слева | #3 справа,
    одинаковый viewport). Чистый HTML, без новых депов; ссылки на соседние файлы относительные."""
    trs = "".join(
        f'<tr><td class="k">{k}</td><td>{lab}</td><td class="num">{cost}</td>'
        f'<td class="num">{lat}</td><td>{note}</td></tr>'
        for k, lab, cost, lat, note in rows)
    css = (
        "body{margin:0;font-family:system-ui,-apple-system,sans-serif;background:#f4f5f7}"
        "header{background:#b23a1e;color:#fff;padding:14px 20px}header h1{margin:0;font-size:22px}"
        "table{border-collapse:collapse;margin:14px 20px;background:#fff;"
        "box-shadow:0 1px 4px rgba(0,0,0,.15)}"
        "th,td{padding:7px 14px;border-bottom:1px solid #e2e4e8;font-size:14px;text-align:left}"
        "th{background:#2b3138;color:#fff}td.k{font-weight:800;text-align:center}"
        "td.num{text-align:right;font-variant-numeric:tabular-nums}"
        ".take{margin:0 20px 12px;font-size:14px;color:#333}"
        ".maps{display:flex;gap:10px;padding:0 12px 14px}.maps figure{flex:1;margin:0}"
        ".maps figcaption{font-size:13px;font-weight:600;padding:4px 6px}"
        "iframe{width:100%;height:78vh;border:1px solid #ccc;border-radius:4px}")
    thead = ("<tr><th></th><th>сценарий</th><th>стоимость</th><th>реакция</th>"
             "<th>что это</th></tr>")
    html = (
        f'<!doctype html><meta charset="utf-8"><title>{scene_title}</title>\n'
        f"<style>{css}</style>\n"
        f"<header><h1>{scene_title}</h1></header>\n"
        f"<table>{thead}{trs}</table>\n"
        f'<p class="take">{takeaway}</p>\n'
        '<div class="maps">\n'
        " <figure><figcaption>A: do-nothing — едем старым планом сквозь событие "
        "(B, OR-Tools, — без карты)</figcaption>\n"
        f'  <iframe src="{left}" title="do-nothing"></iframe></figure>\n'
        " <figure><figcaption>C: наш re-plan за 0.7&nbsp;с</figcaption>\n"
        f'  <iframe src="{right}" title="re-plan"></iframe></figure>\n'
        "</div>\n")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")


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

    # граф Аугсбурга для дорожной геометрии карт (реальные улицы, кэш путей). Тот же снапшот.
    graph = ox.load_graphml(_SNAP / "graph.graphml")
    stop2node = _stop_to_node(inst)
    cache: dict = {}
    p_morning = out / "1_morning_plan.html"
    p_sheet = out / "route_sheet.md"
    p_noreplan = out / "2_incident_no_replan.html"
    p_replan = out / "3_incident_replan.html"
    p_compare = out / "compare.html"
    files = [str(p_morning), str(p_sheet), str(p_noreplan), str(p_replan), str(p_compare)]

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
    morning_cost = sheet["cost_eur"]
    md = rs.render_md(sheet, inst, names, seed=seed, cost_anchor=anchor)  # статик-only (dyn=None)
    p_sheet.write_text(md, encoding="utf-8")
    _render_map(inst, routes, p_morning, graph=graph, stop2node=stop2node, cache=cache, names=names,
                price=f"{morning_cost:.1f} €", price_val=morning_cost,
                title=f"Утренний план · {base.strftime('%H:%M')} · {n - 1} стопов",
                caption=f"Статик free-flow, полный день (парити system_metrics per_seed[{seed}]). "
                        "ДРУГОЙ мир, чем карты после события — напрямую не сравнивать.",
                banner_color="#2b5797", show_eta=True)
    say(f"      → {T['vehicles_used']} машин, {T['km']:.1f} км, "
        f"on-time {T['on_time_pct']:.0f}% · **{morning_cost:.1f} €** "
        f"(парити system_metrics per_seed[{seed}] ✓)")
    say(f"      → карта:  {p_morning}")
    say(f"      → лист:   {p_sheet}")
    if open_maps:
        webbrowser.open(p_morning.resolve().as_uri())

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
    scene = f"{ctx['clock']} — {_EVENT_TITLE[event_kind]}. Дилемма диспетчера"

    # [3/5] событие
    _step(3, f"{ctx['clock']} — СОБЫТИЕ ({event_kind}):")
    for line in ctx["lines"]:
        say(f"      {line}")

    pending = [i for i in range(1, n) if i not in state.served]
    if not pending and not state.urgent:  # край: всё обслужено к моменту события
        _step(4, "Re-plan не нужен — весь остаток уже обслужен.")
        for p, ttl in ((p_noreplan, "без re-plan"), (p_replan, "после re-plan")):
            _render_map(inst, exec_routes, p, graph=graph, stop2node=stop2node, cache=cache,
                        names=names, price="0.0 €", price_val=0.0,
                        title=f"{ctx['clock']} · остаток пуст ({ttl})",
                        caption="Событие пришло на пустой остаток — план не менялся.",
                        banner_color="#555", incident=ctx["incident"])
        _write_compare(p_compare, left=p_noreplan.name, right=p_replan.name, scene_title=scene,
                       rows=[("—", "остаток пуст", "0.0 €", "—", "план не менялся")],
                       takeaway="Событие пришло на пустой остаток — re-plan не потребовался.")
        _step(5, "Итог: событие пришло на пустой остаток, план не менялся.")
        return {"seed": seed, "event": event_kind, "static_cost": morning_cost,
                "morning_cost": morning_cost, "n_served": len(state.served), "n_pending": 0,
                "n_moved": 0, "cost_before": 0.0, "cost_after": 0.0, "or_cost": 0.0,
                "savings": 0.0, "on_time_pct": 100.0, "unserved": 0, "files": files}

    res = residual_instance(state)
    fleet = state.fleet(im.FLEET_SIZE)
    travel = congestion_for(res, dow=dow, offset_min=state.now_min, incidents=state.incidents)

    # [4/5] re-plan: сцена A/B/C (do-nothing / OR-Tools / система) — тот же residual, compare_replan
    _step(4, f"Re-plan из текущего состояния ({len(res.demand) - 1} стопов в остатке)…")
    planner = PortfolioPlanner(pol, k_samples=16, temperature=1.0, rl_starts=8,
                               polish_budget_ms=400.0, polish_top_m=5)
    cmp = compare_replan(res, travel, pol, fleet_size=fleet, deadline_s=2,
                         rl_planner=planner, rl_reps=2, warmup=1)
    new = planner.plan(res, travel, fleet_size=fleet)["routes"]  # маршруты для карт

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

    # стоимости — ВСЕ в одном residual+congestion мире (запрет №3: честный бейзлайн, тот же инстанс)
    old_res = _continue_old_plan(exec_routes, state, idx=idx,  # idx — та же residual-нумерация
                                 drop_vehicle=ctx["drop_vehicle"], veh_of=veh_of)
    cost_before = -evaluate_solution(old_res, res, _CFG, travel=travel)["reward"]  # do-nothing
    # цена системы — ИМЕННО нарисованного плана `new`, а не отдельного rl-прогона внутри
    # compare_replan (polish time-budgeted → мог разойтись с картой); из compare_replan — latency
    q_new = evaluate_solution(new, res, _CFG, travel=travel)
    cost_after = -q_new["reward"]            # система (портфель+polish) — тот же plan, что на #3
    or_cost = -cmp["ortools"]["reward"]      # OR-Tools re-solve (тот же residual, deadline 2с)
    savings = cost_before - cost_after
    ot, uns = q_new["on_time_pct"], int(q_new["unserved"])

    def _lat(mk):
        return f"{cmp[mk]['latency_ms']:.0f} мс этого прогона (durable медиана {_DUR[mk]} мс)"

    say(f"      A do-nothing (едем старым планом): {cost_before:.1f} €")
    say(f"      B OR-Tools re-solve: {or_cost:.1f} € · {_lat('ortools')}")
    say(f"      C система (портфель+polish): {cost_after:.1f} € · {_lat('rl')}")
    say(f"        greedy-контроль (латентность): {_lat('greedy')}")
    say(f"      Перестроено: {n_moved} стопов перераспределены между машинами "
        f"(метки по max-overlap).")

    # честный вердикт (dec-0012/0013): edge системы = СКОРОСТЬ реакции, НЕ качество. OR-Tools при
    # полном бюджете (~30 с) обгоняет систему по качеству; здесь у OR лишь реакция-бюджет.
    or_note = ("OR за реакция-бюджет ещё не сошёлся (нужны ~30 с)" if or_cost > cost_after else
               "здесь OR по цене уже конкурентен; edge системы — скорость реакции")
    takeaway = (
        f"Стоимости (один residual-мир): бездействие {cost_before:.1f} € · OR-Tools@~2 с "
        f"{or_cost:.1f} € · система@0.7 с {cost_after:.1f} €. Ценность системы — СКОРОСТЬ реакции "
        f"(×{_SPEEDUP:.1f} к OR-Tools по latency), НЕ качество: при полном бюджете (~30 с) OR "
        f"обгоняет систему по качеству (durable-вердикт). {or_note}.")
    say(f"      → {takeaway}")

    # [5/5] карты #2/#3 + compare.html (всё в congestion-мире остатка — НЕ статик-587.9€)
    old_full = [[idx[k] for k in r] for r in old_res]  # остаток старого плана в full-нумерации
    _render_map(inst, old_full, p_noreplan, graph=graph, stop2node=stop2node, cache=cache,
                names=names, price=f"{cost_before:.1f} €", price_val=cost_before,
                title=f"{ctx['clock']} — ехать по-старому",
                caption=f"Остаток старого плана сквозь событие, без реакции: {cost_before:.1f} € "
                        "(residual+congestion).", banner_color="#b23a1e", incident=ctx["incident"])
    if savings >= 0:
        cap3 = (f"Наш re-plan за 0.7 с: {cost_after:.1f} € — экономия −{savings:.1f} € vs "
                "«по-старому» (тот же residual-мир).")
    else:
        cap3 = (f"Наш re-plan за 0.7 с: {cost_after:.1f} € (Δ {-savings:+.1f} € vs «по-старому»).")
    _render_map(inst, new_full, p_replan, graph=graph, stop2node=stop2node, cache=cache,
                names=names, price=f"{cost_after:.1f} €", price_val=cost_after,
                title="Наш re-plan за 0.7 с", caption=cap3, banner_color="#1a7a3c",
                incident=ctx["incident"], old_routes=old_full)
    _write_compare(p_compare, left=p_noreplan.name, right=p_replan.name, scene_title=scene, rows=[
        ("A", "do-nothing (без реакции)", f"{cost_before:.1f} €", "0 с",
         "едем старым планом, копятся задержки"),
        ("B", "OR-Tools re-solve", f"{or_cost:.1f} €", "~2 с (2001 мс)",
         "пересчёт с нуля, бюджет ~2 с (полное качество — при ~30 с)"),
        ("C", "наша система (портфель+polish)", f"{cost_after:.1f} €", "0.7 с (689 мс)",
         f"GNN-старт + polish, реакция ×{_SPEEDUP:.1f}")], takeaway=takeaway)

    _step(5, "Итог дня (остаток под congestion+событие — ДРУГОЙ мир, не сравним со статик-планом):")
    say(f"      • без re-plan (do-nothing): {cost_before:.1f} €")
    say(f"      • после re-plan (система): {cost_after:.1f} € (Δ {cost_after - cost_before:+.1f}€)")
    say(f"      • on-time {ot:.0f}% · необслужено {uns} "
        f"{'(все окна соблюдены ✓)' if ot >= 100 and uns == 0 else '(честно из scorer)'}")
    say(f"      → карты: {p_noreplan.name} | {p_replan.name}  ·  кадр: {p_compare}")
    if open_maps:
        webbrowser.open(p_compare.resolve().as_uri())

    return {"seed": seed, "event": event_kind, "static_cost": morning_cost,
            "morning_cost": morning_cost, "n_served": len(state.served),
            "n_pending": len(res.demand) - 1, "n_moved": n_moved, "cost_before": cost_before,
            "cost_after": cost_after, "or_cost": or_cost, "savings": savings,
            "on_time_pct": ot, "unserved": uns, "files": files}


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
