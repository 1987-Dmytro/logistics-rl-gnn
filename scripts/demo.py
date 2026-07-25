"""scripts/demo.py — единая нарративная демонстрация системы (реюз, без новых депов/расчётов).

    python scripts/demo.py [--seed 0] [--event traffic|breakdown|urgent] [--no-open]
                           [--scenario scenarios/friday_south.yaml] [--no-model]

Проверяемость (Phase 9): на старте — ПРОВЕНАНС модели (путь + sha256 + дата обучения + решение);
sha обязан совпасть с провенансом durable-сводки, иначе громкий останов (тихого фолбэка нет).
Шаги [2/5] и [4/5] печатают ТАБЛИЦУ КАНДИДАТОВ портфеля (источник → cost → кто выбран → polish),
`--no-model` гоняет тот же портфель БЕЗ RL-кандидатов и печатает вклад модели этого прогона.
`--scenario` подменяет день/аптеки/флот/события (config.scenario); без флага — дефолтный вторник.

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
import hashlib
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
from logistics_rl_gnn.replan.portfolio import SOURCE_RU, PortfolioPlanner  # noqa: E402

_SNAP = Path("data/snapshots/augsburg_20260720")
_SM = Path("results/system_metrics.json")
_CFG = CostConfig()
# durable медианы re-plan (polish_summary.json, dec-0009) — hardware-independent якоря
_DUR = {"rl": 689, "greedy": 7, "ortools": 2001}
_SPEEDUP = _DUR["ortools"] / _DUR["rl"]  # ×2.9 реакция система vs OR-Tools (durable)
# заголовок сцены A/B/C по типу события (клок берётся из данных, не хардкод)
# заголовок сцены: у traffic зависит от магнитуды (закрытие ≠ замедление — подпись не должна врать)
_EVENT_TITLE = {"breakdown": "Ausfall — машина выбыла", "urgent": "Eilauftrag — срочный заказ"}
# палитра машин (folium) — общая для всех карт демо
_PAL = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#17becf"]
# файл чекпойнта → решение, где эти веса обучены (заявление проверяемо по репо). Ключ — ИМЯ файла:
# путь могут дать и относительный, и абсолютный.
_DECISION = {"policy_pomo_congestion.pt":
             "knowledge/decisions/0007-phase6b-congestion-training.md"}


def say(text: str = "") -> None:
    print(text)


def _step(k: int, title: str) -> None:
    say(f"\n[{k}/5] {title}")


# ---------- провенанс модели (фальсифицируемость: те ли это веса) ----------


def _training_summary(ckpt: Path) -> dict:
    """Сводка ОБУЧЕНИЯ этих весов: провенанс указывает на ЭТОТ чекпойнт И есть дата (у сводок-
    потребителей весов — ablation/polish/search — поля `date` нет, они не обучали).

    BEST-EFFORT: сводки вне git, их отсутствие/устаревание НЕ роняет демо. Жёсткая сверка — только
    с той сводкой, из которой демо берёт durable-числа (system_metrics), см. check_model_provenance.
    """
    for p in sorted(_SM.parent.glob("*_summary.json")):  # та же папка, что и durable-сводка
        try:
            d = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        rec = ((d.get("provenance") or {}).get("checkpoint") or {}).get("path")
        # сверяем РАЗРЕШЁННЫЕ пути: в сводках путь относительный, вызвать демо могут с абсолютным
        same = rec is not None and Path(rec).resolve() == Path(ckpt).resolve()
        if same and d.get("date"):
            return {**d, "_path": str(p)}
    return {}


def check_model_provenance(ckpt: Path, sm: dict) -> dict:
    """sha256 весов ОБЯЗАН совпасть с провенансом durable-сводки (под которым посчитаны числа).

    Нет файла / мисматч → SystemExit: у демо НЕТ тихого фолбэка (честный прогон без модели —
    флаг `--no-model`). Возвращает поля баннера.
    """
    if not ckpt.exists():
        raise SystemExit(
            f"НЕТ ВЕСОВ: {ckpt} (чекпойнты вне git — запрет №1). Демо НЕ подменяет модель тихо: "
            f"верни чекпойнт (обучение train_pomo.py) или запусти `--no-model` — портфель без "
            f"RL-кандидатов, честно помеченный в выводе."
        )
    sha = hashlib.sha256(ckpt.read_bytes()).hexdigest()
    want = ((sm.get("provenance") or {}).get("checkpoint") or {}).get("sha256_16")
    if not want:
        raise SystemExit(f"{_SM}: нет provenance.checkpoint.sha256_16 — нечем проверить веса")
    if sha[: len(want)] != want:
        raise SystemExit(
            f"ПРОВЕНАНС-МИСМАТЧ: {ckpt} sha256={sha[:16]}…, а durable-числа {_SM} посчитаны на "
            f"{want}… — это ДРУГИЕ веса, числа демо несопоставимы с таблицами README. Останов."
        )
    tr = _training_summary(ckpt)
    return {"ckpt": str(ckpt), "sha256": sha, "sha256_16": sha[:16], "date": tr.get("date"),
            "phase": tr.get("phase"), "summary": tr.get("_path"),
            "decision": _DECISION.get(Path(ckpt).name)}


def _banner(prov: dict | None, ckpt: Path) -> None:
    """Шапка прогона: чем именно считаем (или что модель отключена флагом)."""
    say("─" * 96)
    if prov is None:
        say(f"МОДЕЛЬ ОТКЛЮЧЕНА (--no-model): портфель БЕЗ RL-кандидатов; {ckpt} не загружается.")
    else:
        say(f"Модель:   {prov['ckpt']} · sha256 {prov['sha256_16']}… "
            f"(== провенанс {_SM} ✓)")
        say(f"Обучена:  {prov['date'] or 'н/д'} · {prov['phase'] or '—'} · "
            f"сводка {prov['summary'] or 'н/д'}")
        say(f"Решение:  {prov['decision'] or '—'}")
    say("─" * 96)


def _print_candidates(rows: list, chosen: str, *, note: str = "") -> None:
    """Таблица кандидатов портфеля: источник → сырой € (лучший/средний) → после polish → выбран.

    polished = «—» означает «этот источник в polish-топ-M не попал» (re-plan полирует только топ-M),
    а НЕ «polish не помог» — подставлять сюда сырую цену нельзя (сравнение перестанет быть честным).
    """
    win = chosen.partition("+")[0]
    say(f"      кандидаты портфеля{note}:")
    say(f"        {'источник':<22}{'n':>4}{'лучший €':>11}{'средний €':>11}{'+polish €':>11}")
    for r in rows:
        pol = "—" if r["polished"] is None else f"{r['polished']:.1f}"
        mark = "  ← выбран" if r["source"] == win else ""
        say(f"        {SOURCE_RU.get(r['source'], r['source']):<22}{r['n']:>4}"
            f"{r['cost']:>11.1f}{r['mean']:>11.1f}{pol:>11}{mark}")


def _clock(base, minutes):
    return rs._clock(base, minutes)


# ---------- событие (per-kind, из харнесса 0004) ----------


def _in_zone(incidents, coord, abs_min) -> bool:
    """Стоп в активной зоне ЛЮБОГО инцидента — по САМОЙ логике Incident (геометрию не дублируем)."""
    return any(inc.at_node(coord, abs_min) != 0.0 for inc in incidents)


def _active(incidents, abs_min) -> list:
    """Инциденты, АКТИВНЫЕ в этот момент (окно ещё не истекло) — проверка их же логикой (центр
    зоны в зоне по определению). Иначе карта рисует давно рассосавшуюся пробку как живую."""
    return [inc for inc in incidents if inc.at_node(inc.center, abs_min) != 0.0]


def _event_context(kind, ev, inst, state, veh_of, names, *, abs_min) -> dict:
    """Человеческие факты события + затронутые стопы/машины. Числа/зона — из данных, не хардкод.

    abs_min — момент в АБСОЛЮТНОМ клоке congestion (= at_min + сдвиг диспетч-старта): по нему
    Incident решает, активен ли он. Зона считается по ВСЕМ активным инцидентам (сценарий может
    накопить несколько), заголовок — по событию-триггеру.
    """
    n = len(inst.demand)
    pending = [i for i in range(1, n) if i not in state.served]
    ctx = {"clock": _clock(inst.start_datetime, ev.at_min),
           "incidents": _active(state.incidents, abs_min),  # только живые на момент события
           "affected": [], "vehicles": set(), "drop_vehicle": None, "lines": []}
    if kind == "traffic":
        inc = ev.incident
        factor = ("закрытие (∞)" if math.isinf(inc.magnitude)
                  else f"замедление ×{1 + inc.magnitude:.1f}")
        epi = min(range(1, n), key=lambda i: abs(inst.coords[i][0] - inc.center[0])
                  + abs(inst.coords[i][1] - inc.center[1]))
        zone = [i for i in pending if _in_zone(ctx["incidents"], inst.coords[i], abs_min)]
        vehs = {veh_of[i] for i in zone if i in veh_of}
        ctx.update(affected=zone, vehicles=vehs)
        more = (f" Активных зон сейчас: {len(ctx['incidents'])} (стопы считаем по всем)."
                if len(ctx["incidents"]) > 1 else "")
        ctx["lines"] = [
            f"пробка/инцидент у аптеки «{rs._label(int(inst.snapshot_stops[epi]), names)}» "
            f"(радиус {inc.radius_km:.1f} км, {factor}).{more}",
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


def _eur(v: float) -> str:
    """€ для печати. ∞ = старый план проходит СКВОЗЬ закрытие (δ=∞) → физически непроезжаем;
    это не артефакт форматирования, а цена бездействия: без re-plan план не исполним вовсе."""
    return "∞ €" if not math.isfinite(v) else f"{v:.1f} €"


def _gap_line(title: str, without, with_model, note: str = "") -> str:
    """Строка вклада модели: «без модели X · с моделью Y → Δ (%)». None-сторона → честное «—»."""
    if without is None or with_model is None:
        have = ("—" if without is None and with_model is None else
                f"без модели {without:.1f} €" if without is not None else
                f"с моделью {with_model:.1f} €")
        return f"        • {title}: {have} · вторая сторона не измерена{note}"
    d = without - with_model  # > 0 → модель дешевле
    pct = (100.0 * d / without) if without else 0.0
    return (f"        • {title}: без модели {without:.1f} € · с моделью {with_model:.1f} € "
            f"→ {-d:+.1f} € ({-pct:+.1f} %){note}")


def _print_contribution(plan_report, plan_out, *, anchor, use_model, budget_ms, seed,
                        replan_nomodel=None) -> dict:
    """Вклад модели ЭТОГО прогона = портфель с RL-кандидатами vs он же без них.

    План дня: обе стороны измерены в ОДНОМ прогоне и сопоставимы — `system_routes` даёт КАЖДОМУ
    кандидату полный `budget_ms` polish'а, поэтому «без модели» здесь == polished greedy честного
    `--no-model`-прогона (проверено: 640.6 € в обоих).
    Re-plan: сторона «без модели» — ОТДЕЛЬНЫЙ портфель `PortfolioPlanner(None)` с тем же полным
    polish-бюджетом (`replan_nomodel`). Взять greedy+polish из портфеля с моделью нельзя: там
    бюджет делится на топ-M, а его цена входит в min → Δ была бы тождественно ≤ 0, и «модель
    ухудшила» стало бы непечатаемым исходом.
    Под `--no-model` «с моделью» для плана берём из durable per_seed-якоря того же инстанса и того
    же budget_ms (кастомный сценарий → якоря нет, честное «не измерено»).
    Знак: «+X €» = модель ДОРОЖЕ на X (вклад отрицательный), «−X €» = дешевле.
    """
    if use_model:
        plan_wo, plan_w, note = plan_report["cost_nomodel"], plan_report["cost_model"], ""
    else:
        plan_wo, plan_w = plan_report["cost_model"], anchor
        note = (f"  (с моделью — durable per_seed[{seed}], тот же инстанс и budget "
                f"{budget_ms:.0f} мс)" if anchor is not None
                else "  (durable-якоря у сценария нет)")
    rep_wo = replan_nomodel if use_model else plan_out["cost"]
    rep_w = plan_out["cost"] if use_model else None
    rep_note = "" if use_model else "  (сторона «с моделью» — прогон без флага)"
    say("      Вклад модели этого прогона (портфель БЕЗ RL-кандидатов vs с ними):")
    lines = [_gap_line("план дня", plan_wo, plan_w, note),
             _gap_line("re-plan  ", rep_wo, rep_w, rep_note)]
    for ln in lines:
        say(ln)
    return {"plan_without": plan_wo, "plan_with": plan_w,
            "replan_without": rep_wo, "replan_with": rep_w}


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
                title, caption, banner_color, incidents=(), old_routes=None, show_eta=False):
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
    for incident in incidents:  # ВСЕ активные зоны красным + подпись (сценарий может дать 2+)
        closed = math.isinf(incident.magnitude)  # закрытие vs замедление — подписи разные
        tag, what = (("🚧 Sperrung", "закрытие") if closed
                     else ("🚦 Stau", f"замедление ×{1 + incident.magnitude:.1f}"))
        folium.Circle([incident.center[1], incident.center[0]], radius=incident.radius_km * 1000,
                      color="red", fill=True, fill_opacity=0.12, weight=2,
                      tooltip=f"{what} (r={incident.radius_km:.1f} км)").add_to(m)
        folium.Marker([incident.center[1], incident.center[0]], icon=folium.DivIcon(
            html='<div style="font-size:11px;color:#c00;font-weight:700;white-space:nowrap;'
                 f'transform:translate(-50%,-24px)">{tag}</div>')).add_to(m)
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
                   takeaway: str, right_caption: str = "C: наш re-plan за 0.7 с"):
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
    # «реакция» — durable медианы (dec-0009), НЕ wall-clock этого прогона: подписано в шапке
    thead = ("<tr><th></th><th>сценарий</th><th>стоимость (этот прогон)</th>"
             "<th>реакция (durable медиана)</th><th>что это</th></tr>")
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
        f" <figure><figcaption>{right_caption}</figcaption>\n"
        f'  <iframe src="{right}" title="re-plan"></iframe></figure>\n'
        "</div>\n")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")


# ---------- главный сценарий ----------


def run_demo(*, seed: int, event_kind: str, out_dir: str, open_maps: bool,
             scenario_path=None, use_model: bool = True) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sm = json.loads(_SM.read_text())
    cfg = sm["config"]
    ckpt = Path(cfg["ckpt"])
    prov = check_model_provenance(ckpt, sm) if use_model else None  # мисматч/нет весов → SystemExit
    _banner(prov, ckpt)

    torch.manual_seed(0)
    pol = rd._load_policy(ckpt) if use_model else None
    scen = None
    if scenario_path is not None:
        from logistics_rl_gnn.config.scenario import load_scenario  # опц. депа (pyyaml)

        scen = load_scenario(scenario_path, snapshot_dir=_SNAP, seed=seed)
        inst, dow, anchor = scen.instance, scen.weekday, None  # durable-якоря у сценария нет
    else:
        inst = im.generate_instance(snapshot_dir=_SNAP, seed=seed)
        dow = im.DELIVERY_WEEKDAY
        anchor = float(sm["per_seed_cost_eur"][seed])
    # парити-якорь применим ТОЛЬКО к тому же портфелю, что дал durable-числа: под --no-model
    # (портфель без RL-кандидатов) прогон обязан отличаться — сверять с якорем нечего.
    parity = anchor if use_model else None
    marks = ([scen.name] if scen is not None else []) + (
        [] if use_model else ["--no-model (портфель без RL-кандидатов)"])
    run_label = " · ".join(marks) or None  # чем этот прогон отличается от durable-прогона
    names = rs.load_names(_SNAP)
    base = inst.start_datetime
    n = len(inst.demand)
    fleet_k, fleet_q = im.fleet_of(inst)
    off = im.dispatch_offset_min(inst)  # сдвиг congestion-часов, если смена стартует не в 08:00

    # [1/5] утро
    _step(1, f"Утро, {rs._WEEKDAY_RU[base.weekday()]} {base.strftime('%H:%M')}. "
             f"Депо PHOENIX, {rs._DEPOT_ADDR}.")
    if scen is not None:
        say(f"      Сценарий: «{scen.name}» ({scen.path}) — кастомный день, durable-якоря нет.")
        if scen.dropped_stops:  # закрыты по реальным opening_hours → в плане их НЕТ (говорим вслух)
            shown = ", ".join(scen.label(s) for s in scen.dropped_stops[:6])
            more = len(scen.dropped_stops) - 6
            say(f"      ⚠ закрыты в этот день и выпали из плана ({len(scen.dropped_stops)}): "
                f"{shown}{f' и ещё {more}' if more > 0 else ''}")
    say(f"      Заказы: {n - 1} аптек, {int(inst.demand.sum())} боксов. "
        f"Флот K={fleet_k}, вместимость Q={fleet_q:.0f}, T_max={im.T_MAX_MIN / 60:.0f} ч.")

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
    plan_report: dict = {}
    routes = es.system_routes(pol, inst, budget_ms=cfg["budget_ms"], k_samples=cfg["k_samples"],
                              temp=cfg["temperature"], rl_starts=cfg["rl_starts"],
                              report=plan_report)
    _print_candidates(plan_report["rows"], plan_report["chosen"],
                      note=" — план дня (единый scorer; polish получает КАЖДОГО кандидата)")
    sheet = rs.build_sheet(routes, inst)
    q = evaluate_solution(routes, inst, _CFG)
    assert abs(sheet["cost_eur"] - (-q["reward"])) < 1e-6, "walk-cost != scorer"
    if parity is not None:
        assert abs(sheet["cost_eur"] - parity) < 0.5, (
            f"ПАРИТИ FAIL: {sheet['cost_eur']:.2f}€ != per_seed[{seed}] {parity:.2f}€")
    # страж сценарного флота: план НЕ вправе использовать больше машин, чем дал сценарий
    assert sheet["totals"]["vehicles_used"] <= fleet_k, (
        f"ФЛОТ FAIL: план занял {sheet['totals']['vehicles_used']} машин при K={fleet_k}")
    T = sheet["totals"]
    morning_cost = sheet["cost_eur"]
    md = rs.render_md(sheet, inst, names, seed=seed, cost_anchor=parity,  # статик-only (dyn=None)
                      scenario=run_label)
    p_sheet.write_text(md, encoding="utf-8")
    anchor_note = (f"парити system_metrics per_seed[{seed}]" if parity is not None
                   else f"прогон «{run_label}» — durable-якоря нет, число этого прогона")
    _render_map(inst, routes, p_morning, graph=graph, stop2node=stop2node, cache=cache, names=names,
                price=f"{morning_cost:.1f} €", price_val=morning_cost,
                title=f"Утренний план · {base.strftime('%H:%M')} · {n - 1} стопов",
                caption=f"Статик free-flow, полный день ({anchor_note}). "
                        "ДРУГОЙ мир, чем карты после события — напрямую не сравнивать.",
                banner_color="#2b5797", show_eta=True)
    say(f"      → {T['vehicles_used']} машин, {T['km']:.1f} км, "
        f"on-time {T['on_time_pct']:.0f}% · **{morning_cost:.1f} €** "
        + (f"(парити system_metrics per_seed[{seed}] ✓)" if parity is not None
           else "(durable-якоря нет: кастомный прогон)"))
    say(f"      → карта:  {p_morning}")
    say(f"      → лист:   {p_sheet}")
    if open_maps:
        webbrowser.open(p_morning.resolve().as_uri())

    # --- динамик-мир (congestion): исполняемый план дня ---
    # ВАЖНО (иначе легко прочесть неверно): «старый план» ниже — это ДИСПЕТЧЕРСКИЙ greedy-план под
    # диурнал-congestion из харнесса 0004, а НЕ полированный план шага [2/5] (тот статик free-flow,
    # другой мир; смешивать их нельзя). Так же строит его run_dynamic — на этом стоят durable-числа.
    # Инциденты в исполняемый таймлайн не закладываются: кто успел обслужиться к событию, считается
    # по диурналу (упрощение харнесса 0004 — то же и в durable-прогоне).
    exec_travel = congestion_for(inst, dow=dow, offset_min=off)
    exec_routes = greedy_routes(env=make_dynamic_env(inst, travel=exec_travel))
    if scen is not None:  # события сценария: применяем ВСЕ по порядку, триггер — последнее
        if not scen.events:
            raise SystemExit(f"в сценарии «{scen.name}» нет событий (нечего перепланировать)")
        evs = list(scen.events)
    else:
        ev = next((e for e in event_stream(seed, inst, dow) if e.kind == event_kind), None)
        if ev is None:
            raise SystemExit(f"в потоке seed {seed} нет события '{event_kind}'")
        evs = [ev]
    ev = evs[-1]
    event_kind = ev.kind
    state = DynamicState(inst, dow, now_min=float(ev.at_min))
    state.served = served_by(exec_routes, inst, exec_travel, ev.at_min)
    for e in evs:  # состояние мира на момент триггера = все события, что уже случились
        e.apply(state)
    veh_of, _ = rs._assign(exec_routes, inst, exec_travel)
    ctx = _event_context(event_kind, ev, inst, state, veh_of, names, abs_min=ev.at_min + off)
    title = (_EVENT_TITLE.get(event_kind) or  # traffic: закрытие или всего лишь замедление
             ("Straßensperrung — перекрытие" if math.isinf(ev.incident.magnitude)
              else "Stau — пробка"))
    scene = f"{ctx['clock']} — {title}. Дилемма диспетчера"

    # [3/5] событие
    _step(3, f"{ctx['clock']} — СОБЫТИЕ ({event_kind}):")
    if len(evs) > 1:
        say("      цепочка событий сценария: "
            + " · ".join(f"{_clock(base, e.at_min)} {e.kind}" for e in evs))
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
                        banner_color="#555", incidents=ctx["incidents"])
        _write_compare(p_compare, left=p_noreplan.name, right=p_replan.name, scene_title=scene,
                       rows=[("—", "остаток пуст", "0.0 €", "—", "план не менялся")],
                       takeaway="Событие пришло на пустой остаток — re-plan не потребовался.")
        _step(5, "Итог: событие пришло на пустой остаток, план не менялся.")
        return {"seed": seed, "event": event_kind, "static_cost": morning_cost,
                "morning_cost": morning_cost, "n_served": len(state.served), "n_pending": 0,
                "n_moved": 0, "cost_before": 0.0, "cost_after": 0.0, "or_cost": 0.0,
                "savings": 0.0, "on_time_pct": 100.0, "unserved": 0, "files": files,
                "used_model": use_model, "scenario": None if scen is None else scen.name,
                "plan_report": plan_report, "replan_rows": [], "contribution": None}

    res = residual_instance(state)
    fleet = state.fleet(fleet_k)
    travel = congestion_for(res, dow=dow, offset_min=state.now_min + off,
                            incidents=state.incidents)

    # [4/5] re-plan: сцена A/B/C (do-nothing / OR-Tools / система) — тот же residual, compare_replan
    _step(4, f"Re-plan из текущего состояния ({len(res.demand) - 1} стопов в остатке)…")
    planner = PortfolioPlanner(pol, k_samples=16, temperature=1.0, rl_starts=8,
                               polish_budget_ms=400.0, polish_top_m=5)
    cmp = compare_replan(res, travel, pol, fleet_size=fleet, deadline_s=2,
                         rl_planner=planner, rl_reps=2, warmup=1)
    plan_out = planner.plan(res, travel, fleet_size=fleet)  # маршруты для карт + таблица
    new = plan_out["routes"]
    # сторона «БЕЗ модели» для re-plan — ОТДЕЛЬНЫЙ портфель без RL-кандидатов на ТОМ ЖЕ residual/
    # travel/флоте и с ТЕМ ЖЕ ПОЛНЫМ polish-бюджетом. Брать greedy+polish из портфеля С моделью
    # нельзя дважды: там бюджет делится на топ-M (greedy достаётся 1/M), а его цена ВХОДИТ в min →
    # разница вышла бы тождественно ≤ 0, то есть «вклад модели» был бы нефальсифицируем.
    # Вне timed-блока compare_replan → на печатаемую латентность не влияет.
    rep_nomodel = (PortfolioPlanner(None, polish_budget_ms=planner.polish_budget_ms,
                                    polish_top_m=planner.polish_top_m)
                   .plan(res, travel, fleet_size=fleet)["cost"] if use_model else None)
    _print_candidates(plan_out["rows"], plan_out["source"],
                      note=f" — re-plan остатка (единый scorer; polish — только топ-"
                           f"{planner.polish_top_m} по cost)")

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

    sys_label = ("система (портфель+polish)" if use_model
                 else "портфель БЕЗ модели (greedy+polish, --no-model)")
    # кто РЕАЛЬНО выиграл портфель (часто greedy+polish, а не RL-кандидат) — этим планом и
    # подписаны карта #3 и строка C: атрибуция «GNN-старт» там, где победил greedy, была бы ложью
    win_src = SOURCE_RU.get(plan_out["source"].partition("+")[0], plan_out["source"])
    win_pol = " + polish" if "+polish" in plan_out["source"] else " (без polish)"
    blocked = not math.isfinite(cost_before)  # старый план идёт сквозь закрытие → неисполним
    dead = " — старый план идёт СКВОЗЬ закрытие: без re-plan он неисполним" if blocked else ""
    say(f"      A do-nothing (едем старым планом): {_eur(cost_before)}{dead}")
    say(f"      B OR-Tools re-solve: {or_cost:.1f} € · {_lat('ortools')}")
    say(f"      C {sys_label}: {cost_after:.1f} € · {_lat('rl')}")
    say(f"        greedy-контроль (латентность): {_lat('greedy')}")
    say(f"      Перестроено: {n_moved} стопов перераспределены между машинами "
        f"(метки по max-overlap).")

    # честный вердикт (dec-0012/0013): edge системы = СКОРОСТЬ реакции, НЕ качество. OR-Tools при
    # полном бюджете (~30 с) обгоняет систему по качеству; здесь у OR лишь реакция-бюджет.
    or_note = ("OR за реакция-бюджет ещё не сошёлся (нужны ~30 с)" if or_cost > cost_after else
               "здесь OR по цене уже конкурентен; edge системы — скорость реакции")
    ablated = ("" if use_model else
               " ВНИМАНИЕ: прогон БЕЗ модели (--no-model) — цена C получена портфелем без "
               "RL-кандидатов, а durable-вердикт по латентности относится к полной системе.")
    takeaway = (
        f"Стоимости (один residual-мир): бездействие {_eur(cost_before)}"
        f"{' (маршрут перекрыт)' if blocked else ''} · OR-Tools@~2 с {or_cost:.1f} € · "
        f"{'система@0.7 с' if use_model else 'портфель без модели'} {cost_after:.1f} €. "
        f"Ценность системы — СКОРОСТЬ реакции "
        f"(×{_SPEEDUP:.1f} к OR-Tools по latency), НЕ качество: при полном бюджете (~30 с) OR "
        f"обгоняет систему по качеству (durable-вердикт). {or_note}.{ablated}")
    say(f"      → {takeaway}")

    # [5/5] карты #2/#3 + compare.html (всё в congestion-мире остатка — НЕ статик-587.9€)
    old_full = [[idx[k] for k in r] for r in old_res]  # остаток старого плана в full-нумерации
    _render_map(inst, old_full, p_noreplan, graph=graph, stop2node=stop2node, cache=cache,
                names=names, price=_eur(cost_before), price_val=cost_before,
                title=f"{ctx['clock']} — ехать по-старому",
                caption=("Остаток ИСПОЛНЯЕМОГО плана дня (диспетчерский greedy под congestion, "
                         f"харнесс 0004 — не план карты #1) сквозь событие: {_eur(cost_before)} "
                         + ("— маршрут перекрыт закрытием, план неисполним."
                            if blocked else "(residual+congestion).")),
                banner_color="#b23a1e", incidents=ctx["incidents"])
    # «за 0.7 с» — durable медиана РЕАКЦИИ СИСТЕМЫ; под --no-model системы в прогоне не было
    replan_title = "Наш re-plan за 0.7 с" if use_model else "Re-plan БЕЗ модели (--no-model)"
    if blocked:
        cap3 = (f"{replan_title}: {cost_after:.1f} € — исполнимый план там, где старый "
                "упирался в закрытие (∞).")
    elif savings >= 0:
        cap3 = (f"{replan_title}: {cost_after:.1f} € — экономия −{savings:.1f} € vs "
                "«по-старому» (тот же residual-мир).")
    else:
        cap3 = (f"{replan_title}: {cost_after:.1f} € (Δ {-savings:+.1f} € vs «по-старому»).")
    cap3 += f" План построил кандидат «{win_src}»{win_pol}."
    _render_map(inst, new_full, p_replan, graph=graph, stop2node=stop2node, cache=cache,
                names=names, price=f"{cost_after:.1f} €", price_val=cost_after,
                title=replan_title, caption=cap3, banner_color="#1a7a3c",
                incidents=ctx["incidents"], old_routes=old_full)
    c_lat = f"0.7 с ({_DUR['rl']} мс)" if use_model else f"{cmp['rl']['latency_ms']:.0f} мс прогона"
    c_what = (f"портфель{'' if use_model else ' БЕЗ модели (--no-model)'}: победил «{win_src}»"
              f"{win_pol}" + (f", реакция ×{_SPEEDUP:.1f}" if use_model else ""))
    _write_compare(p_compare, left=p_noreplan.name, right=p_replan.name, scene_title=scene, rows=[
        ("A", "do-nothing (без реакции)", _eur(cost_before), "0 с",
         "маршрут перекрыт: старый план неисполним" if blocked
         else "едем старым планом, копятся задержки"),
        ("B", "OR-Tools re-solve", f"{or_cost:.1f} €", f"~2 с ({_DUR['ortools']} мс)",
         "пересчёт с нуля, бюджет ~2 с (полное качество — при ~30 с)"),
        ("C", f"наша {sys_label}", f"{cost_after:.1f} €", c_lat, c_what)],
        takeaway=takeaway, right_caption=f"C: {replan_title}")

    _step(5, "Итог дня (остаток под congestion+событие — ДРУГОЙ мир, не сравним со статик-планом):")
    say(f"      • без re-plan (do-nothing): {_eur(cost_before)}"
        + (" — план перекрыт закрытием (неисполним)" if blocked else ""))
    say(f"      • после re-plan ({'система' if use_model else 'без модели'}): {cost_after:.1f} € "
        + ("(исполнимый план вместо неисполнимого)" if blocked
           else f"(Δ {cost_after - cost_before:+.1f}€)"))
    say(f"      • on-time {ot:.0f}% · необслужено {uns} "
        f"{'(все окна соблюдены ✓)' if ot >= 100 and uns == 0 else '(честно из scorer)'}")
    contrib = _print_contribution(plan_report, plan_out, anchor=anchor, use_model=use_model,
                                  budget_ms=cfg["budget_ms"], seed=seed,
                                  replan_nomodel=rep_nomodel)
    say(f"      → карты: {p_noreplan.name} | {p_replan.name}  ·  кадр: {p_compare}")
    if open_maps:
        webbrowser.open(p_compare.resolve().as_uri())

    return {"seed": seed, "event": event_kind, "static_cost": morning_cost,
            "morning_cost": morning_cost, "n_served": len(state.served),
            "n_pending": len(res.demand) - 1, "n_moved": n_moved, "cost_before": cost_before,
            "cost_after": cost_after, "or_cost": or_cost, "savings": savings,
            "on_time_pct": ot, "unserved": uns, "files": files,
            "used_model": use_model, "scenario": None if scen is None else scen.name,
            "plan_report": plan_report, "replan_rows": plan_out["rows"], "contribution": contrib}


def main() -> None:
    ap = argparse.ArgumentParser(description="Единая нарративная демонстрация системы (реюз)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--event", choices=("traffic", "breakdown", "urgent"), default="traffic")
    ap.add_argument("--out", default="demo_out")
    ap.add_argument("--no-open", action="store_true", help="не открывать карты в браузере")
    ap.add_argument("--scenario", default=None,
                    help="YAML кастомного сценария (scenarios/*.yaml); без флага — дефолтный "
                         "вторник, весь город (события берутся из сценария, --event игнорируется)")
    ap.add_argument("--no-model", action="store_true",
                    help="портфель БЕЗ RL-кандидатов (ablation: сколько даёт модель)")
    args = ap.parse_args()
    run_demo(seed=args.seed, event_kind=args.event, out_dir=args.out, open_maps=not args.no_open,
             scenario_path=args.scenario, use_model=not args.no_model)
    say("\nГотово. Артефакты в demo_out/ (вне git).")


if __name__ == "__main__":
    main()
