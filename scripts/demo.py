"""scripts/demo.py — one narrative demonstration of the system (reuse, no new deps/computations).

    python scripts/demo.py [--seed 0] [--event traffic|breakdown|urgent] [--no-open]
                           [--scenario scenarios/friday_south.yaml] [--no-model]

Verifiability (Phase 9): at start — the model PROVENANCE (path + sha256 + training date + decision);
the sha must match the durable summary's provenance, otherwise a loud stop (no silent fallback).
Steps [2/5] and [4/5] print the portfolio CANDIDATE TABLE (source → cost → who won → polish),
`--no-model` runs the same portfolio WITHOUT RL candidates and prints the model's contribution.
`--scenario` swaps the day/pharmacies/fleet/events (config.scenario); without it a default Tuesday.

5 steps in plain language: morning → plan construction (parity with system_metrics) → event (the
0004 harness) → re-plan (scene A/B/C: do-nothing vs OR-Tools vs the system, live timing + durable
medians) → the day's outcome. ALL numbers come from the same scorers (route_sheet.build_sheet /
compare_replan) — static free-flow (587.9€, the full day, map #1) and dynamic congestion residual
(maps #2/#3) are DIFFERENT worlds and are never mixed. Artefacts → demo_out/ (outside git):
  1_morning_plan.html · route_sheet.md · 2_incident_no_replan.html · 3_incident_replan.html ·
  compare.html (two iframes #2|#3 + the A/B/C table — the frame for a screencast).
Map hops follow real streets (nx.shortest_path over graph.graphml, path cache); the old plan on #3
is a toggleable dashed layer; the incident zone is labelled.

Reuse: eval_system.system_routes, route_sheet.{build_sheet,render_md,walk_route,_assign,
_match_labels}, env.events (event_stream/residual/served), replan.compare_replan + PortfolioPlanner
(the same mechanism that produced the durable 689/2001 ms in polish_summary.json). Nothing new is
computed; the latency in headers/table is the durable median (#4), live wall-clock only in step 4.
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
# durable re-plan medians (polish_summary.json, dec-0009) — hardware-independent anchors
_DUR = {"rl": 689, "greedy": 7, "ortools": 2001}
_SPEEDUP = _DUR["ortools"] / _DUR["rl"]  # ×2.9 reaction, system vs OR-Tools (durable)
# the A/B/C scene title by event kind (the clock comes from data, not hardcoded)
# scene title: for traffic it depends on the magnitude (a closure ≠ a slowdown — no lying labels)
_EVENT_TITLE = {"breakdown": "Ausfall — vehicle lost", "urgent": "Eilauftrag — urgent order"}
# vehicle palette (folium) — shared by every demo map
_PAL = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#17becf"]
# checkpoint file → the decision where these weights were trained (a claim checkable in the repo).
# The key is the FILE NAME: the path may be given relative or absolute.
_DECISION = {"policy_pomo_congestion.pt":
             "knowledge/decisions/0007-phase6b-congestion-training.md"}


def say(text: str = "") -> None:
    print(text)


def _step(k: int, title: str) -> None:
    say(f"\n[{k}/5] {title}")


# ---------- model provenance (falsifiability: are these the right weights) ----------


def _training_summary(ckpt: Path) -> dict:
    """TRAINING summary of these weights: the provenance points at THIS checkpoint AND has a date
    (summaries that only consume weights — ablation/polish/search — have no `date` field).

    BEST-EFFORT: the summaries live outside git, their absence/staleness does NOT break the demo.
    The hard check is only against the summary the demo takes its durable numbers from.
    """
    for p in sorted(_SM.parent.glob("*_summary.json")):  # the same folder as the durable summary
        try:
            d = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        rec = ((d.get("provenance") or {}).get("checkpoint") or {}).get("path")
        # compare RESOLVED paths: summaries store a relative path, the demo may be called absolute
        same = rec is not None and Path(rec).resolve() == Path(ckpt).resolve()
        if same and d.get("date"):
            return {**d, "_path": str(p)}
    return {}


def check_model_provenance(ckpt: Path, sm: dict) -> dict:
    """The weights' sha256 MUST match the provenance of the durable summary behind the numbers.

    No file / a mismatch → SystemExit: the demo has NO silent fallback (an honest run without the
    model is the `--no-model` flag). Returns the banner fields.
    """
    if not ckpt.exists():
        raise SystemExit(
            f"NO WEIGHTS: {ckpt} (checkpoints live outside git — prohibition #1). The demo never "
            f"swaps the model silently: restore the checkpoint (train_pomo.py) or run `--no-model` "
            f"— a portfolio without RL candidates, honestly marked in the output."
        )
    sha = hashlib.sha256(ckpt.read_bytes()).hexdigest()
    want = ((sm.get("provenance") or {}).get("checkpoint") or {}).get("sha256_16")
    if not want:
        raise SystemExit(f"{_SM}: no provenance.checkpoint.sha256_16 — nothing to verify against")
    if sha[: len(want)] != want:
        raise SystemExit(
            f"PROVENANCE MISMATCH: {ckpt} sha256={sha[:16]}…, while the durable numbers in {_SM} "
            f"were computed on {want}… — DIFFERENT weights, the demo is incomparable. Stop."
        )
    tr = _training_summary(ckpt)
    return {"ckpt": str(ckpt), "sha256": sha, "sha256_16": sha[:16], "date": tr.get("date"),
            "phase": tr.get("phase"), "summary": tr.get("_path"),
            "decision": _DECISION.get(Path(ckpt).name)}


def _banner(prov: dict | None, ckpt: Path) -> None:
    """Run header: what exactly we compute with (or that the model is disabled by a flag)."""
    say("─" * 96)
    if prov is None:
        say(f"MODEL DISABLED (--no-model): portfolio WITHOUT RL candidates; {ckpt} is not loaded.")
    else:
        say(f"Model:    {prov['ckpt']} · sha256 {prov['sha256_16']}… "
            f"(== provenance of {_SM} ✓)")
        say(f"Trained:  {prov['date'] or 'n/a'} · {prov['phase'] or '—'} · "
            f"summary {prov['summary'] or 'n/a'}")
        say(f"Decision: {prov['decision'] or '—'}")
    say("─" * 96)


def _print_candidates(rows: list, chosen: str, *, note: str = "") -> None:
    """Portfolio candidate table: source → raw € (best/mean) → after polish → chosen.

    polished = '—' means 'this source never reached the polish top-M' (re-plan polishes only top-M),
    NOT 'polish did not help' — substituting the raw price here would make the comparison dishonest.
    """
    win = chosen.partition("+")[0]
    say(f"      portfolio candidates{note}:")
    say(f"        {'source':<22}{'n':>4}{'best €':>11}{'mean €':>11}{'+polish €':>11}")
    for r in rows:
        pol = "—" if r["polished"] is None else f"{r['polished']:.1f}"
        mark = "  ← chosen" if r["source"] == win else ""
        say(f"        {SOURCE_RU.get(r['source'], r['source']):<22}{r['n']:>4}"
            f"{r['cost']:>11.1f}{r['mean']:>11.1f}{pol:>11}{mark}")


def _clock(base, minutes):
    return rs._clock(base, minutes)


# ---------- the event (per kind, from the 0004 harness) ----------


def _in_zone(incidents, coord, abs_min) -> bool:
    """A stop inside the active zone of ANY incident — by Incident's own logic (no duplication)."""
    return any(inc.at_node(coord, abs_min) != 0.0 for inc in incidents)


def _active(incidents, abs_min) -> list:
    """Incidents ACTIVE at this moment (the window has not expired) — checked by their own logic
    (a zone centre is inside the zone by definition). Else the map draws a long-gone jam as live."""
    return [inc for inc in incidents if inc.at_node(inc.center, abs_min) != 0.0]


def _event_context(kind, ev, inst, state, veh_of, names, *, abs_min) -> dict:
    """Human-readable facts of the event + affected stops/vehicles. Numbers/zone from data.

    abs_min — the moment on the ABSOLUTE congestion clock (= at_min + dispatch-start shift): the
    Incident decides by it whether it is active. The zone covers ALL active incidents (a scenario
    accumulate several); the title follows the triggering event.
    """
    n = len(inst.demand)
    pending = [i for i in range(1, n) if i not in state.served]
    ctx = {"clock": _clock(inst.start_datetime, ev.at_min),
           "incidents": _active(state.incidents, abs_min),  # only those alive at the event
           "affected": [], "vehicles": set(), "drop_vehicle": None, "lines": []}
    if kind == "traffic":
        inc = ev.incident
        factor = ("closure (∞)" if math.isinf(inc.magnitude)
                  else f"slowdown ×{1 + inc.magnitude:.1f}")
        epi = min(range(1, n), key=lambda i: abs(inst.coords[i][0] - inc.center[0])
                  + abs(inst.coords[i][1] - inc.center[1]))
        zone = [i for i in pending if _in_zone(ctx["incidents"], inst.coords[i], abs_min)]
        vehs = {veh_of[i] for i in zone if i in veh_of}
        ctx.update(affected=zone, vehicles=vehs)
        more = (f" Active zones right now: {len(ctx['incidents'])} (stops counted over all)."
                if len(ctx["incidents"]) > 1 else "")
        ctx["lines"] = [
            f"jam/incident near pharmacy '{rs._label(int(inst.snapshot_stops[epi]), names)}' "
            f"(radius {inc.radius_km:.1f} km, {factor}).{more}",
            f"{len(zone)} undelivered stops are in the zone, "
            f"vehicles affected: {sorted(vehs) or '—'}.",
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
            f"vehicle {drop} broke down — {len(orphans)} stops orphaned.",
            "Fleet −1; the orphaned stops go back into the shared pool for reassignment.",
        ]
    else:  # urgent
        o = ev.order
        idx = o["idx"]
        ctx.update(affected=[idx], vehicles={veh_of.get(idx)} - {None})
        ctx["lines"] = [
            f"urgent order: pharmacy '{rs._label(int(inst.snapshot_stops[idx]), names)}' — "
            f"{o['demand']} boxes, a narrow window of {o['delta_s'] / 60:.0f} min.",
            "It must be inserted into the current routes — a re-plan trigger.",
        ]
    return ctx


def _eur(v: float) -> str:
    """€ for printing. ∞ = the old plan runs THROUGH a closure (δ=∞) → physically impassable;
    not a formatting artefact but the price of inaction: without a re-plan it cannot be executed."""
    return "∞ €" if not math.isfinite(v) else f"{v:.1f} €"


def _gap_line(title: str, without, with_model, note: str = "") -> str:
    """Model contribution line: 'without the model X · with it Y → Δ (%)'. A None side → '—'."""
    if without is None or with_model is None:
        have = ("—" if without is None and with_model is None else
                f"without the model {without:.1f} €" if without is not None else
                f"with the model {with_model:.1f} €")
        return f"        • {title}: {have} · the other side was not measured{note}"
    d = without - with_model  # > 0 → the model is cheaper
    pct = (100.0 * d / without) if without else 0.0
    return (f"        • {title}: without the model {without:.1f} € · with it {with_model:.1f} € "
            f"→ {-d:+.1f} € ({-pct:+.1f} %){note}")


def _print_contribution(plan_report, plan_out, *, anchor, use_model, budget_ms, seed,
                        replan_nomodel=None) -> dict:
    """This run's model contribution = the portfolio with RL candidates vs the same without them.

    Day plan: both sides are measured in ONE run and are comparable — `system_routes` gives EVERY
    candidate the full `budget_ms` of polish, so 'without the model' here == the polished greedy of
    an honest `--no-model` run (verified: 640.6 € in both).
    Re-plan: the 'without the model' side is a SEPARATE `PortfolioPlanner(None)` with the same full
    polish budget (`replan_nomodel`). Taking greedy+polish out of the portfolio WITH the model is
    not allowed: there the budget is split across the top-M and its price enters the min → Δ would
    be identically ≤ 0 and 'the model made it worse' would be an unprintable outcome.
    Under `--no-model` the 'with the model' side of the plan comes from the durable per_seed anchor
    of the same instance and the same budget_ms (a custom scenario has no anchor → 'not measured').
    Sign: '+X €' = the model is X DEARER (negative contribution), '−X €' = cheaper.
    """
    if use_model:
        plan_wo, plan_w, note = plan_report["cost_nomodel"], plan_report["cost_model"], ""
    else:
        plan_wo, plan_w = plan_report["cost_model"], anchor
        note = (f"  (with the model — durable per_seed[{seed}], same instance and budget "
                f"{budget_ms:.0f} ms)" if anchor is not None
                else "  (the scenario has no durable anchor)")
    rep_wo = replan_nomodel if use_model else plan_out["cost"]
    rep_w = plan_out["cost"] if use_model else None
    rep_note = "" if use_model else "  (the 'with the model' side — a run without the flag)"
    say("      This run's model contribution (portfolio WITHOUT RL candidates vs with them):")
    lines = [_gap_line("day plan", plan_wo, plan_w, note),
             _gap_line("re-plan  ", rep_wo, rep_w, rep_note)]
    for ln in lines:
        say(ln)
    return {"plan_without": plan_wo, "plan_with": plan_w,
            "replan_without": rep_wo, "replan_with": rep_w}


# ---------- 'continue the old plan' (the no-re-plan counterfactual) ----------


def _continue_old_plan(exec_routes, state, *, idx, drop_vehicle, veh_of):
    """The remainder of the old plan as a residual solution: GENUINELY remaining stops (not served)
    in the original per-vehicle order (the 'we did not re-plan' counterfactual). idx — residual
    numbering (== residual_instance, PASSED IN from the caller: one source, else urgent desyncs).
    drop_vehicle (breakdown) — its stops fall out (orphaned). An urgent re-delivery (stop served,
    new demand) is NOT part of the old plan → it stays unserved in the counterfactual (honestly:
    without a re-plan the order is not fulfilled)."""
    pos = {full: k for k, full in enumerate(idx)}
    out = []
    for route in exec_routes:
        seq = [pos[b] for b in route
               if b != 0 and b in pos and b not in state.served
               and (drop_vehicle is None or veh_of.get(b) != drop_vehicle)]
        if seq:
            out.append([0, *seq, 0])
    return out


# ---------- road geometry (real streets, path cache) ----------


def _stop_to_node(inst) -> dict[int, int]:
    """instance stop index → OSM node_id in graph.graphml (via nodes.parquet, the same snapshot)."""
    nd = pd.read_parquet(_SNAP / "nodes.parquet").set_index("stop")["node_id"].astype(int)
    return {n: int(nd[int(inst.snapshot_stops[n])]) for n in range(len(inst.demand))}


def _road_latlon(graph, na: int, nb: int, cache: dict) -> list:
    """Polyline of the hop na→nb along real streets (nx.shortest_path, weight=length) with a cache.
    Fallback (no path/node) — a straight line over GRAPH NODE coordinates (vertices stay nodes)."""
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
    """A whole route [0,s1,…,0] → one polyline along real streets (no duplicate node at joins)."""
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


# ---------- demo map (header + real streets + old-plan layer + zone) ----------


def _render_map(inst, primary, out: Path, *, graph, stop2node, cache, names, price, price_val,
                title, caption, banner_color, incidents=(), old_routes=None, show_eta=False):
    """One demo map: a floating header (large price + title + what-you-see), the depot, the incident
    zone (labelled), an optional toggleable dashed 'old plan' layer, the main plan as solid real
    streets + numbered stops (popup: name; ETA/window only on #1 free-flow). price_val — the
    machine-readable header price (data-demo-price) for the guard test 'headers == demo output'."""
    import folium

    c = inst.coords
    base = inst.start_datetime
    m = folium.Map(location=[c[0][1], c[0][0]], zoom_start=12, tiles="cartodbpositron")
    m.get_root().html.add_child(folium.Element(  # header; push the zoom controls out from under it
        '<style>.leaflet-top{top:76px}</style>'
        f'<div style="position:fixed;top:0;left:0;right:0;z-index:9999;background:{banner_color};'
        'color:#fff;padding:8px 16px;font-family:system-ui,-apple-system,sans-serif;'
        'box-shadow:0 2px 8px rgba(0,0,0,.35)">'
        f'<span data-demo-price="{price_val:.6f}" style="font-size:22px;font-weight:800">{price}'
        f'</span><span style="font-size:15px;font-weight:600;margin-left:12px">{title}</span>'
        f'<div style="font-size:12px;opacity:.92;margin-top:2px">{caption}</div></div>'))
    folium.Marker([c[0][1], c[0][0]], tooltip="PHOENIX depot",
                  icon=folium.Icon(color="red", icon="star")).add_to(m)
    for incident in incidents:  # ALL active zones in red + a label (a scenario may give 2+)
        closed = math.isinf(incident.magnitude)  # closure vs slowdown — different labels
        tag, what = (("🚧 Sperrung", "closure") if closed
                     else ("🚦 Stau", f"slowdown ×{1 + incident.magnitude:.1f}"))
        folium.Circle([incident.center[1], incident.center[0]], radius=incident.radius_km * 1000,
                      color="red", fill=True, fill_opacity=0.12, weight=2,
                      tooltip=f"{what} (r={incident.radius_km:.1f} km)").add_to(m)
        folium.Marker([incident.center[1], incident.center[0]], icon=folium.DivIcon(
            html='<div style="font-size:11px;color:#c00;font-weight:700;white-space:nowrap;'
                 f'transform:translate(-50%,-24px)">{tag}</div>')).add_to(m)
    if old_routes:  # the old plan — a toggleable dashed layer (off)
        fg = folium.FeatureGroup(name="old plan (no re-plan)", show=False)
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
                        opacity=0.9, tooltip=f"veh. {v + 1}").add_to(m)
        stops, _ = rs.walk_route(route, inst)
        for k, s in enumerate(stops, start=1):
            eta = ""
            if show_eta:
                eta = (f"<br>ETA {rs._clock(base, s['arr_min'])} · window "
                       f"{rs._clock(base, s['e_min'])}–{rs._clock(base, s['l_min'])}")
            popup = folium.Popup(f"<b>{k}. {rs._label(s['snap'], names)}</b><br>veh. {v + 1}{eta}",
                                 max_width=260)
            folium.Marker([c[s["n"]][1], c[s["n"]][0]], popup=popup,
                          icon=_num_icon(k, col)).add_to(m)
        v += 1
    if old_routes:
        folium.LayerControl(collapsed=False).add_to(m)
    out.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out))


def _write_compare(out: Path, *, left: str, right: str, scene_title: str, rows: list,
                   takeaway: str, right_caption: str = "C: our re-plan in 0.7 s"):
    """compare.html — the screencast frame: a shared title 'The dispatcher's dilemma' + the A/B/C
    table (all three costs in ONE residual world + latency) + two iframes (#2 left | #3 right, same
    viewport). Plain HTML, no new deps; links to the neighbouring files are relative."""
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
    # 'reaction' — durable medians (dec-0009), NOT this run's wall-clock: stated in the header
    thead = ("<tr><th></th><th>scenario</th><th>cost (this run)</th>"
             "<th>reaction (durable median)</th><th>what it is</th></tr>")
    html = (
        f'<!doctype html><meta charset="utf-8"><title>{scene_title}</title>\n'
        f"<style>{css}</style>\n"
        f"<header><h1>{scene_title}</h1></header>\n"
        f"<table>{thead}{trs}</table>\n"
        f'<p class="take">{takeaway}</p>\n'
        '<div class="maps">\n'
        " <figure><figcaption>A: do-nothing — driving the old plan through the event "
        "(B, OR-Tools, has no map)</figcaption>\n"
        f'  <iframe src="{left}" title="do-nothing"></iframe></figure>\n'
        f" <figure><figcaption>{right_caption}</figcaption>\n"
        f'  <iframe src="{right}" title="re-plan"></iframe></figure>\n'
        "</div>\n")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")


# ---------- main scenario ----------


def run_demo(*, seed: int, event_kind: str, out_dir: str, open_maps: bool,
             scenario_path=None, use_model: bool = True) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sm = json.loads(_SM.read_text())
    cfg = sm["config"]
    ckpt = Path(cfg["ckpt"])
    prov = check_model_provenance(ckpt, sm) if use_model else None  # mismatch/no weights → exit
    _banner(prov, ckpt)

    torch.manual_seed(0)
    pol = rd._load_policy(ckpt) if use_model else None
    scen = None
    if scenario_path is not None:
        from logistics_rl_gnn.config.scenario import load_scenario  # optional dep (pyyaml)

        scen = load_scenario(scenario_path, snapshot_dir=_SNAP, seed=seed)
        inst, dow, anchor = scen.instance, scen.weekday, None  # a scenario has no durable anchor
    else:
        inst = im.generate_instance(snapshot_dir=_SNAP, seed=seed)
        dow = im.DELIVERY_WEEKDAY
        anchor = float(sm["per_seed_cost_eur"][seed])
    # the parity anchor applies ONLY to the portfolio that produced the durable numbers: under
    # --no-model (a portfolio without RL candidates) the run must differ — nothing to compare.
    parity = anchor if use_model else None
    marks = ([scen.name] if scen is not None else []) + (
        [] if use_model else ["--no-model (portfolio without RL candidates)"])
    run_label = " · ".join(marks) or None  # how this run differs from the durable one
    names = rs.load_names(_SNAP)
    base = inst.start_datetime
    n = len(inst.demand)
    fleet_k, fleet_q = im.fleet_of(inst)
    off = im.dispatch_offset_min(inst)  # congestion hour shift when the shift does not start 08:00

    # [1/5] morning
    _step(1, f"Morning, {rs._WEEKDAY_NAMES[base.weekday()]} {base.strftime('%H:%M')}. "
             f"PHOENIX depot, {rs._DEPOT_ADDR}.")
    if scen is not None:
        say(f"      Scenario: '{scen.name}' ({scen.path}) — a custom day, no durable anchor.")
        if scen.dropped_stops:  # closed per real opening_hours → NOT in the plan (say it aloud)
            shown = ", ".join(scen.label(s) for s in scen.dropped_stops[:6])
            more = len(scen.dropped_stops) - 6
            say(f"      ⚠ closed that day and dropped from the plan ({len(scen.dropped_stops)}): "
                f"{shown}{f' and {more} more' if more > 0 else ''}")
    say(f"      Orders: {n - 1} pharmacies, {int(inst.demand.sum())} boxes. "
        f"Fleet K={fleet_k}, capacity Q={fleet_q:.0f}, T_max={im.T_MAX_MIN / 60:.0f} h.")

    # the Augsburg graph for map road geometry (real streets, path cache). The same snapshot.
    graph = ox.load_graphml(_SNAP / "graph.graphml")
    stop2node = _stop_to_node(inst)
    cache: dict = {}
    p_morning = out / "1_morning_plan.html"
    p_sheet = out / "route_sheet.md"
    p_noreplan = out / "2_incident_no_replan.html"
    p_replan = out / "3_incident_replan.html"
    p_compare = out / "compare.html"
    files = [str(p_morning), str(p_sheet), str(p_noreplan), str(p_replan), str(p_compare)]

    # [2/5] plan construction (STATIC free-flow — parity with system_metrics)
    _step(2, "Building the plan… (portfolio + local-search polish)")
    plan_report: dict = {}
    routes = es.system_routes(pol, inst, budget_ms=cfg["budget_ms"], k_samples=cfg["k_samples"],
                              temp=cfg["temperature"], rl_starts=cfg["rl_starts"],
                              report=plan_report)
    _print_candidates(plan_report["rows"], plan_report["chosen"],
                      note=" — the day plan (one scorer; EVERY candidate gets polish)")
    sheet = rs.build_sheet(routes, inst)
    q = evaluate_solution(routes, inst, _CFG)
    assert abs(sheet["cost_eur"] - (-q["reward"])) < 1e-6, "walk-cost != scorer"
    if parity is not None:
        assert abs(sheet["cost_eur"] - parity) < 0.5, (
            f"PARITY FAIL: {sheet['cost_eur']:.2f}€ != per_seed[{seed}] {parity:.2f}€")
    # scenario fleet guard: the plan may not use more vehicles than the scenario granted
    assert sheet["totals"]["vehicles_used"] <= fleet_k, (
        f"FLEET FAIL: the plan used {sheet['totals']['vehicles_used']} vehicles at K={fleet_k}")
    T = sheet["totals"]
    morning_cost = sheet["cost_eur"]
    md = rs.render_md(sheet, inst, names, seed=seed, cost_anchor=parity,  # static-only (dyn=None)
                      scenario=run_label)
    p_sheet.write_text(md, encoding="utf-8")
    anchor_note = (f"parity with system_metrics per_seed[{seed}]" if parity is not None
                   else f"run '{run_label}' — no durable anchor, the number of this run")
    _render_map(inst, routes, p_morning, graph=graph, stop2node=stop2node, cache=cache, names=names,
                price=f"{morning_cost:.1f} €", price_val=morning_cost,
                title=f"Morning plan · {base.strftime('%H:%M')} · {n - 1} stops",
                caption=f"Static free-flow, the full day ({anchor_note}). "
                        "A DIFFERENT world from the post-event maps — not directly comparable.",
                banner_color="#2b5797", show_eta=True)
    say(f"      → {T['vehicles_used']} vehicles, {T['km']:.1f} km, "
        f"on-time {T['on_time_pct']:.0f}% · **{morning_cost:.1f} €** "
        + (f"(parity with system_metrics per_seed[{seed}] ✓)" if parity is not None
           else "(no durable anchor: a custom run)"))
    say(f"      → map:   {p_morning}")
    say(f"      → sheet: {p_sheet}")
    if open_maps:
        webbrowser.open(p_morning.resolve().as_uri())

    # --- the dynamic world (congestion): the executed plan of the day ---
    # IMPORTANT (easy to misread): the 'old plan' below is the DISPATCHER's greedy plan under
    # diurnal congestion from the 0004 harness, NOT the polished plan of step [2/5] (that one is
    # static free-flow, another world; they must not be mixed). run_dynamic builds it the same way.
    # Incidents are not baked into the executed timeline: who managed to be served by the event is
    # computed on the diurnal (a simplification of harness 0004 — the same in the durable run).
    exec_travel = congestion_for(inst, dow=dow, offset_min=off)
    exec_routes = greedy_routes(env=make_dynamic_env(inst, travel=exec_travel))
    if scen is not None:  # scenario events: apply ALL in order, the trigger is the last one
        if not scen.events:
            raise SystemExit(f"scenario '{scen.name}' has no events (nothing to re-plan)")
        evs = list(scen.events)
    else:
        ev = next((e for e in event_stream(seed, inst, dow) if e.kind == event_kind), None)
        if ev is None:
            raise SystemExit(f"the stream of seed {seed} has no '{event_kind}' event")
        evs = [ev]
    ev = evs[-1]
    event_kind = ev.kind
    state = DynamicState(inst, dow, now_min=float(ev.at_min))
    state.served = served_by(exec_routes, inst, exec_travel, ev.at_min)
    for e in evs:  # the world state at the trigger = every event that already happened
        e.apply(state)
    veh_of, _ = rs._assign(exec_routes, inst, exec_travel)
    ctx = _event_context(event_kind, ev, inst, state, veh_of, names, abs_min=ev.at_min + off)
    title = (_EVENT_TITLE.get(event_kind) or  # traffic: a closure or merely a slowdown
             ("Straßensperrung — road closed" if math.isinf(ev.incident.magnitude)
              else "Stau — traffic jam"))
    scene = f"{ctx['clock']} — {title}. The dispatcher's dilemma"

    # [3/5] the event
    _step(3, f"{ctx['clock']} — EVENT ({event_kind}):")
    if len(evs) > 1:
        say("      the scenario's event chain: "
            + " · ".join(f"{_clock(base, e.at_min)} {e.kind}" for e in evs))
    for line in ctx["lines"]:
        say(f"      {line}")

    pending = [i for i in range(1, n) if i not in state.served]
    if not pending and not state.urgent:  # edge case: everything served by the event
        _step(4, "No re-plan needed — the whole remainder is already served.")
        for p, ttl in ((p_noreplan, "no re-plan"), (p_replan, "after re-plan")):
            _render_map(inst, exec_routes, p, graph=graph, stop2node=stop2node, cache=cache,
                        names=names, price="0.0 €", price_val=0.0,
                        title=f"{ctx['clock']} · the remainder is empty ({ttl})",
                        caption="The event hit an empty remainder — the plan did not change.",
                        banner_color="#555", incidents=ctx["incidents"])
        _write_compare(p_compare, left=p_noreplan.name, right=p_replan.name, scene_title=scene,
                       rows=[("—", "remainder empty", "0.0 €", "—", "the plan did not change")],
                       takeaway="The event hit an empty remainder — no re-plan was needed.")
        _step(5, "Outcome: the event hit an empty remainder, the plan did not change.")
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

    # [4/5] re-plan: scene A/B/C (do-nothing / OR-Tools / system) — same residual, compare_replan
    _step(4, f"Re-plan from the current state ({len(res.demand) - 1} stops remaining)…")
    planner = PortfolioPlanner(pol, k_samples=16, temperature=1.0, rl_starts=8,
                               polish_budget_ms=400.0, polish_top_m=5)
    cmp = compare_replan(res, travel, pol, fleet_size=fleet, deadline_s=2,
                         rl_planner=planner, rl_reps=2, warmup=1)
    plan_out = planner.plan(res, travel, fleet_size=fleet)  # routes for the maps + the table
    new = plan_out["routes"]
    # the 'WITHOUT the model' side of the re-plan — a SEPARATE portfolio without RL candidates on
    # the SAME residual/travel/fleet and with the SAME FULL polish budget. Taking greedy+polish out
    # of the portfolio WITH the model is not allowed: there the budget is split over the top-M
    # (greedy gets 1/M) and its price ENTERS the min → the difference would be identically ≤ 0.
    # Outside the timed block of compare_replan → it does not affect the printed latency.
    rep_nomodel = (PortfolioPlanner(None, polish_budget_ms=planner.polish_budget_ms,
                                    polish_top_m=planner.polish_top_m)
                   .plan(res, travel, fleet_size=fleet)["cost"] if use_model else None)
    _print_candidates(plan_out["rows"], plan_out["source"],
                      note=f" — re-plan of the remainder (one scorer; polish — only the top "
                           f"{planner.polish_top_m} by cost)")

    # residual→full mapping: EXACTLY as residual_instance.idx (for urgent it adds the urgent stop
    # even when served → otherwise the numbering desyncs). Replicated, not guessed.
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

    # costs — ALL in one residual+congestion world (#3: an honest baseline, the same instance)
    old_res = _continue_old_plan(exec_routes, state, idx=idx,  # idx — the same residual numbering
                                 drop_vehicle=ctx["drop_vehicle"], veh_of=veh_of)
    cost_before = -evaluate_solution(old_res, res, _CFG, travel=travel)["reward"]  # do-nothing
    # the system's price is that of the DRAWN plan `new`, not of a separate rl run inside
    # compare_replan (polish is time-budgeted → it could drift from the map); latency comes from it
    q_new = evaluate_solution(new, res, _CFG, travel=travel)
    cost_after = -q_new["reward"]            # the system (portfolio+polish) — the plan on #3
    or_cost = -cmp["ortools"]["reward"]      # OR-Tools re-solve (same residual, deadline 2s)
    savings = cost_before - cost_after
    ot, uns = q_new["on_time_pct"], int(q_new["unserved"])

    def _lat(mk):
        return f"{cmp[mk]['latency_ms']:.0f} ms in this run (durable median {_DUR[mk]} ms)"

    sys_label = ("system (portfolio+polish)" if use_model
                 else "portfolio WITHOUT the model (greedy+polish, --no-model)")
    # who REALLY won the portfolio (often greedy+polish, not an RL candidate) — map #3 and row C
    # are labelled by that plan: attributing a 'GNN start' where greedy won would be a lie
    win_src = SOURCE_RU.get(plan_out["source"].partition("+")[0], plan_out["source"])
    win_pol = " + polish" if "+polish" in plan_out["source"] else " (no polish)"
    blocked = not math.isfinite(cost_before)  # the old plan runs through a closure → impassable
    dead = " — the old plan runs THROUGH the closure: it cannot run" if blocked else ""
    say(f"      A do-nothing (drive the old plan): {_eur(cost_before)}{dead}")
    say(f"      B OR-Tools re-solve: {or_cost:.1f} € · {_lat('ortools')}")
    say(f"      C {sys_label}: {cost_after:.1f} € · {_lat('rl')}")
    say(f"        greedy control (latency): {_lat('greedy')}")
    say(f"      Rebuilt: {n_moved} stops reassigned between vehicles "
        f"(labels matched by max overlap).")

    # the honest verdict (dec-0012/0013): the system's edge is REACTION SPEED, NOT quality. At a
    # full budget (~30 s) OR-Tools beats the system on quality; here OR only has a reaction budget.
    or_note = ("OR has not converged in the reaction budget (~30 s needed)" if or_cost > cost_after
               else
               "here OR is already competitive on price; the edge is reaction speed")
    ablated = ("" if use_model else
               " NOTE: this run is WITHOUT the model (--no-model) — price C came from a portfolio "
               "without RL candidates, while the durable latency verdict is about the full system.")
    takeaway = (
        f"Costs (one residual world): inaction {_eur(cost_before)}"
        f"{' (route blocked)' if blocked else ''} · OR-Tools@~2 s {or_cost:.1f} € · "
        f"{'system@0.7 s' if use_model else 'portfolio without the model'} {cost_after:.1f} €. "
        f"The system's value is REACTION SPEED "
        f"(×{_SPEEDUP:.1f} vs OR-Tools in latency), NOT quality: at a full budget (~30 s) OR "
        f"beats the system on quality (the durable verdict). {or_note}.{ablated}")
    say(f"      → {takeaway}")

    # [5/5] maps #2/#3 + compare.html (all in the remainder's congestion world — not static 587.9€)
    old_full = [[idx[k] for k in r] for r in old_res]  # old-plan remainder in full numbering
    _render_map(inst, old_full, p_noreplan, graph=graph, stop2node=stop2node, cache=cache,
                names=names, price=_eur(cost_before), price_val=cost_before,
                title=f"{ctx['clock']} — carry on as before",
                caption=("The remainder of the EXECUTED day plan (the dispatcher's greedy under "
                         "congestion, harness 0004 — not the plan of map #1) through the event: "
                         + ("— the route is blocked by the closure, the plan cannot run."
                            if blocked else "(residual+congestion).")),
                banner_color="#b23a1e", incidents=ctx["incidents"])
    # 'in 0.7 s' — the durable median of the SYSTEM's reaction; under --no-model there was no system
    replan_title = "Our re-plan in 0.7 s" if use_model else "Re-plan WITHOUT the model (--no-model)"
    if blocked:
        cap3 = (f"{replan_title}: {cost_after:.1f} € — an executable plan where the old one "
                "ran into the closure (∞).")
    elif savings >= 0:
        cap3 = (f"{replan_title}: {cost_after:.1f} € — saving −{savings:.1f} € vs "
                "'carry on as before' (the same residual world).")
    else:
        cap3 = (f"{replan_title}: {cost_after:.1f} € (Δ {-savings:+.1f} € vs 'as before').")
    cap3 += f" The plan was built by candidate '{win_src}'{win_pol}."
    _render_map(inst, new_full, p_replan, graph=graph, stop2node=stop2node, cache=cache,
                names=names, price=f"{cost_after:.1f} €", price_val=cost_after,
                title=replan_title, caption=cap3, banner_color="#1a7a3c",
                incidents=ctx["incidents"], old_routes=old_full)
    c_lat = f"0.7 s ({_DUR['rl']} ms)" if use_model else f"{cmp['rl']['latency_ms']:.0f} ms"
    c_what = (f"portfolio{'' if use_model else ' WITHOUT the model (--no-model)'}: '{win_src}' won"
              f"{win_pol}" + (f", reaction ×{_SPEEDUP:.1f}" if use_model else ""))
    _write_compare(p_compare, left=p_noreplan.name, right=p_replan.name, scene_title=scene, rows=[
        ("A", "do-nothing (no reaction)", _eur(cost_before), "0 s",
         "route blocked: the old plan cannot run" if blocked
         else "drive the old plan, delays accumulate"),
        ("B", "OR-Tools re-solve", f"{or_cost:.1f} €", f"~2 s ({_DUR['ortools']} ms)",
         "recompute from scratch, budget ~2 s (full quality at ~30 s)"),
        ("C", f"our {sys_label}", f"{cost_after:.1f} €", c_lat, c_what)],
        takeaway=takeaway, right_caption=f"C: {replan_title}")

    _step(5, "Day outcome (the remainder under congestion+event — ANOTHER world):")
    say(f"      • without a re-plan (do-nothing): {_eur(cost_before)}"
        + (" — the plan is blocked by the closure (cannot run)" if blocked else ""))
    say(f"      • after re-plan ({'system' if use_model else 'no model'}): {cost_after:.1f} € "
        + ("(an executable plan instead of an impossible one)" if blocked
           else f"(Δ {cost_after - cost_before:+.1f}€)"))
    say(f"      • on-time {ot:.0f}% · unserved {uns} "
        f"{'(every window met ✓)' if ot >= 100 and uns == 0 else '(honestly from the scorer)'}")
    contrib = _print_contribution(plan_report, plan_out, anchor=anchor, use_model=use_model,
                                  budget_ms=cfg["budget_ms"], seed=seed,
                                  replan_nomodel=rep_nomodel)
    say(f"      → maps: {p_noreplan.name} | {p_replan.name}  ·  frame: {p_compare}")
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
    ap = argparse.ArgumentParser(description="One narrative demonstration of the system (reuse)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--event", choices=("traffic", "breakdown", "urgent"), default="traffic")
    ap.add_argument("--out", default="demo_out")
    ap.add_argument("--no-open", action="store_true", help="do not open the maps in a browser")
    ap.add_argument("--scenario", default=None,
                    help="YAML of a custom scenario (scenarios/*.yaml); without it — the default "
                         "Tuesday, whole city (events come from the scenario, --event is ignored)")
    ap.add_argument("--no-model", action="store_true",
                    help="portfolio WITHOUT RL candidates (ablation: what the model adds)")
    args = ap.parse_args()
    run_demo(seed=args.seed, event_kind=args.event, out_dir=args.out, open_maps=not args.no_open,
             scenario_path=args.scenario, use_model=not args.no_model)
    say("\nDone. Artefacts in demo_out/ (outside git).")


if __name__ == "__main__":
    main()
