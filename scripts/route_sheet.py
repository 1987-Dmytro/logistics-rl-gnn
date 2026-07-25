"""Phase 8 — route sheet: a human-readable plan from THE SAME eval pipeline as system_metrics.

Reads the config from results/system_metrics.json, runs the polished portfolio (`system_routes`) on
the seed from that config (default 0) and ASSERTs cost == per_seed_cost_eur[seed] (the sheet
describes EXACTLY the plan in the tables — prohibitions #3/#4). Renders docs/route_sheet.md (header
+ per vehicle + a dynamics appendix) and machine-readable results/route_sheet.json (parity tested).

Pharmacy names are optional, from data/snapshots/<snap>/names.parquet (scripts/enrich_names.py);
without them — stop-ids. The "dynamics" appendix: the 0004 harness (event_stream/residual_instance/
served_by), seed 0, the first traffic event → RL re-plan → a diff of reassigned stops (old/new
arrival) + forward-pass latency. Deterministic (torch.manual_seed(0), fixed seed/config).

Run: python scripts/route_sheet.py [--seed N]
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
_WEEKDAY_NAMES = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
_DUR_MEDIAN_MS = 14.4  # durable median of the RL forward-pass re-plan (dynamic.json, dec-0004)
_LAT_GREEDY_MS = 5.9  # durable median greedy re-plan — FASTER than raw RL (dynamic.json → no niche)
_LAT_SYSTEM_MS = 689  # deployed system (portfolio+polish) reaction; ×2.9 vs OR, same quality (0009)
_LAT_OR_MS = 2001  # OR-Tools re-solve (dynamic.json)


# ---------- pharmacy names (optional, additive) ----------


def load_names(snap_dir) -> dict[int, tuple[str, str | None]]:
    """{snapshot_stop: (name, addr)} from names.parquet; no file → {} (fallback to stop-ids)."""
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


# ---------- walk (same semantics as evaluate_solution) ----------


def walk_route(route, inst, travel=None):
    """Per-stop timeline of one route (arrival/wait/service/load) — the same walk as
    `evaluate_solution`. travel=None → free-flow (statics); travel=CongestionTravel → dynamics
    (dt = travel.time(a,b,t), t in minutes from the tour start). -> (stops, totals)."""
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
        if b == 0:  # return to the depot: no window/service
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
    """Structural sheet: non-empty routes → vehicles + totals + cost (== evaluate_solution)."""
    vehicles = []
    for route in routes:
        if len(route) < 2 or all(n == 0 for n in route):
            continue  # empty vehicle ([0]/[0,0]) — as in the scorer
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
            # honest on-time from the walk (NOT hardcoded): share of stops with arrival ≤ l_i
            "on_time_pct": 100.0 if n_visits == 0 else 100.0 * on_time / n_visits,
        },
        "cost_eur": cost,
    }


# ---------- rendering ----------


def _clock(base_dt, minutes: float) -> str:
    return (base_dt + timedelta(minutes=float(minutes))).strftime("%H:%M")


def _win(base_dt, e_min, l_min) -> str:
    return f"[{_clock(base_dt, e_min)}–{_clock(base_dt, l_min)}]"


def render_md(sheet, inst, names, dyn=None, *, seed: int, cost_anchor, scenario=None) -> str:
    """dyn=None → a static sheet without the dynamics appendix (reused by demo.py, which has its
    own live portfolio re-plan story in the terminal + the 2_incident_no_replan/3_incident_replan

    cost_anchor=None → a custom scenario: there is no durable anchor and the header says so (#4).
    """
    base = inst.start_datetime
    T = sheet["totals"]
    fleet_k, fleet_q = im.fleet_of(inst)
    head = (
        f"> Generated by `scripts/route_sheet.py` from the same eval pipeline as `system_metrics` "
        f"(polished portfolio). Cost parity with `results/system_metrics.json` per_seed[{seed}] = "
        f"**{cost_anchor:.1f} €** — the sheet describes EXACTLY the plan in the README tables."
        if cost_anchor is not None else
        f"> Generated by `scripts/route_sheet.py` with the same polished portfolio, run "
        f"'{scenario or 'custom'}'. There is no durable anchor: **{sheet['cost_eur']:.1f} €** is "
        f"the number of THIS run, not from the README tables."
    )
    L = [
        "# Route sheet — pharmacy delivery plan for Augsburg",
        "",
        head,
        "",
        f"- **Date:** {base.strftime('%Y-%m-%d')} ({_WEEKDAY_NAMES[base.weekday()]}), "
        f"dispatch window {base.strftime('%H:%M')}–{_clock(base, inst.horizon_s / 60.0)}",
        f"- **Depot:** PHOENIX, {_DEPOT_ADDR}",
        f"- **Fleet:** K = {fleet_k} vehicles, capacity Q = {fleet_q:.0f} boxes, "
        f"T_max = {im.T_MAX_MIN / 60:.0f} h/tour",
        f"- **Instance:** seed {seed}, {len(inst.demand) - 1} pharmacies "
        f"(windows: REAL {inst.tw_source.count('REAL')} / ASSUMED "
        f"{inst.tw_source.count('ASSUMED')})",
        "",
        "## Day total",
        "",
        "| Stops | Boxes | Distance | Time (routes) | Vehicles used | Cost |",
        "|---:|---:|---:|---:|---:|---:|",
        f"| {T['n_stops']} | {T['boxes']} | {T['km']:.1f} km | {T['time_min'] / 60:.1f} h "
        f"| {T['vehicles_used']} / {fleet_k} | **{sheet['cost_eur']:.1f} €** |",
        "",
        f"Time = driving {T['drive'] / 60:.1f} h + waiting for windows {T['wait'] / 60:.1f} h + "
        f"service {T['service'] / 60:.1f} h. On-time {T['on_time_pct']:.0f} % (arrival ≤ l_i, "
        f"honestly from the walk — not hardcoded).",
        "",
        "## Per vehicle",
    ]
    for vi, v in enumerate(sheet["vehicles"], start=1):
        tot = v["totals"]
        L += [
            "",
            f"### Vehicle {vi} — leaves the depot at {base.strftime('%H:%M')}",
            "",
            "| # | Pharmacy | Arrival | Window | Wait | Service | Load | Leg |",
            "|---:|:--|:--:|:--:|---:|---:|---:|---:|",
        ]
        for k, s in enumerate(v["stops"], start=1):
            L.append(
                f"| {k} | {_label(s['snap'], names)} | {_clock(base, s['arr_min'])} "
                f"| {_win(base, s['e_min'], s['l_min'])} | {s['wait']:.0f}′ "
                f"| {s['service']:.0f}′ | {s['load_after']} | "
                f"{s['leg_km']:.1f} km / {s['leg_min']:.0f}′ |"
            )
        ret = tot["ret"]
        duty = (tot["drive"] + tot["wait"] + tot["service"]) / 60
        L += [
            f"| — | ↩ return to depot | {_clock(base, ret['clock_min'])} | — | — | — | — "
            f"| {ret['leg_km']:.1f} km / {ret['leg_min']:.0f}′ |",
            "",
            f"**Vehicle {vi} total:** {tot['km']:.1f} km · duty {duty:.1f} h "
            f"(driving {tot['drive'] / 60:.1f} / waiting {tot['wait'] / 60:.1f} / "
            f"service {tot['service'] / 60:.1f}) · {tot['boxes']} boxes",
        ]

    if dyn is None:  # static-only sheet (demo.py)
        L.append("")
        return "\n".join(L)

    # --- appendix: dynamics ---
    L += [
        "",
        "---",
        "",
        "## Appendix — dynamics (re-plan on an event)",
        "",
        f"Scenario from the dec-0004 harness (seed {dyn['seed']}): the initial plan runs under "
        f"diurnal congestion, at **{dyn['event_clock']}** a traffic incident fires (closure/jam on "
        f"a zone of edges near pharmacy #{dyn['center_stop']}). We take the partial state "
        f"({dyn['n_served']} stops already served) and re-plan the remaining {dyn['n_pending']} "
        f"with a **raw neural forward pass** (periodic re-optimisation of the remainder from the "
        f"depot, dec-0001 §4; NOT the polished portfolio of the header, only its fast start).",
        "",
        f"- **Forward-pass latency:** {dyn['latency_ms']:.0f} ms in this run (median of 5 "
        f"replicas, hardware-dependent), the durable harness median is **{_DUR_MEDIAN_MS} ms** — "
        f"that is the speed CEILING, but the start is 'quality-inferior on its own terms': greedy "
        f"is both faster (**~{_LAT_GREEDY_MS} ms**) and no worse in cost → raw RL has no latency "
        f"niche (dec-0010). The shared baseline (#3) is greedy + OR-Tools, not OR-Tools alone.",
        f"- **The durable win belongs to the deployed system** (portfolio+polish, the one giving "
        f"{cost_anchor:.0f}€): reaction **~{_LAT_SYSTEM_MS} ms** vs an OR-Tools re-solve "
        f"**{_LAT_OR_MS} ms** = **×2.9** at comparable quality (dec-0009). ×~140 (14.4 ms "
        f"vs OR-only) is NOT a win of the system but a comparison without a shared baseline.",
        f"- **Stops reassigned:** {dyn['n_moved']} of {dyn['n_pending']} changed vehicle "
        f"(vehicle labels matched by max overlap → pure renumbering is filtered out; some stops "
        f"arrive LATER — the price of a raw forward pass without polish).",
        "",
    ]
    if dyn["moved"]:
        capped = dyn["n_moved"] > len(dyn["moved"])
        cap = f" (top {len(dyn['moved'])} by |Δ arrival|)" if capped else ""
        L += [
            f"Stops that changed vehicle{cap} — old (initial plan) → new (re-plan) arrival:",
            "",
            "| Pharmacy | Vehicle was→now | Arrival was | Arrival now | Δ |",
            "|:--|:--:|:--:|:--:|---:|",
        ]
        for m in dyn["moved"]:
            L.append(
                f"| {_label(m['snap'], names)} | {m['old_veh']}→{m['new_veh']} "
                f"| {m['old_clock']} | {m['new_clock']} | {m['delta_min']:.0f}′ |"
            )
    else:
        L.append("_At this event the re-optimisation kept the previous vehicle assignments._")
    L.append("")
    return "\n".join(L)


# ---------- appendix: dynamics (the 0004 harness) ----------


_DIFF_TOP = 12  # cap on diff rows (residual re-opt shuffles a lot → show the top by |Δ|)


def _match_labels(new_routes_full, old_veh) -> dict[int, int]:
    """Stable vehicle labels: every NEW vehicle gets the label of the OLD one it overlaps most.

    Residual re-optimisation numbers vehicles from zero → without matching, 'changed vehicle' is
    pure renumbering noise. Max-overlap matching leaves ONLY real inter-vehicle transfers."""
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
    """First traffic event of the 0004 stream → RL re-plan of the remainder → assignment diff."""
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

    env = make_dynamic_env(res, travel=travel, fleet_size=state.fleet(im.fleet_of(inst)[0]))
    # latency: warmup + median of replicas (more representative than cold start; same plan)
    lat: list[float] = []
    new = None
    for r in range(6):
        t0 = time.perf_counter()
        with torch.no_grad():
            new = pol.rollout(env, mode="greedy")[0]
        if r > 0:  # the first run is the warmup
            lat.append((time.perf_counter() - t0) * 1000.0)
    latency_ms = float(np.median(lat))

    idx = [0] + sorted(i for i in range(1, len(inst.demand)) if i not in state.served)
    old_veh, old_arr = _assign(exec_routes, inst, exec_travel)

    # new plan: routes in FULL numbering (to match labels) + arrivals (residual walk)
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
    for i in new_veh:  # re-planned (pending) stops that changed vehicle (by stable labels)
        if i in old_veh and new_veh[i] != old_veh[i]:
            new_abs = float(ev.at_min) + new_arr[i]
            moved.append({
                "snap": int(inst.snapshot_stops[i]),
                "old_veh": old_veh[i], "new_veh": new_veh[i],
                "old_clock": _clock(base, old_arr[i]), "new_clock": _clock(base, new_abs),
                "delta_min": abs(new_abs - old_arr[i]),
            })
    moved.sort(key=lambda m: -m["delta_min"])  # top by arrival shift
    return {
        "seed": seed, "event_clock": ev_base.strftime("%H:%M"),
        "center_stop": int(inst.snapshot_stops[center_stop]),
        "n_served": len(state.served), "n_pending": len(res.demand) - 1,
        "latency_ms": latency_ms, "n_moved": len(moved), "moved": moved[:_DIFF_TOP],
    }


def _assign(routes, inst, travel, remap=None):
    """(veh[i]=vehicle number 1.., arr[i]=arrival min) over stops of non-empty routes. remap (a
    residual→full list) remaps the keys when routes/inst are in residual numbering; None → as is."""
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
    ap = argparse.ArgumentParser(description="Phase 8 — route sheet (system_metrics parity)")
    ap.add_argument("--seed", type=int, default=0, help="instance seed (per_seed[seed] — anchor)")
    ap.add_argument("--tol", type=float, default=0.5, help="parity tolerance to per_seed[seed], €")
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

    # PARITY GUARD: sheet cost == evaluate_solution == durable per_seed[seed] (#3/#4)
    q = evaluate_solution(routes, inst, _CFG)
    assert abs(sheet["cost_eur"] - (-q["reward"])) < 1e-6, "walk-cost != evaluate_solution"
    assert abs(sheet["cost_eur"] - anchor) < args.tol, (
        f"PARITY FAIL: sheet {sheet['cost_eur']:.3f}€ != per_seed[{args.seed}] {anchor:.3f}€ "
        f"(|Δ|={abs(sheet['cost_eur'] - anchor):.3f} > {args.tol}) — NOT that plan"
    )
    # the sheet claims on-time in the header → verify it honestly from the walk (means: 100)
    assert sheet["totals"]["on_time_pct"] == 100.0, (
        f"ON-TIME FAIL: {sheet['totals']['on_time_pct']:.1f}% < 100 — there is lateness, "
        f"the sheet header lies about on-time"
    )

    snap = im._latest_snapshot_dir()
    names = load_names(snap)
    dyn = dynamics_appendix(pol, inst, seed=args.seed)

    md = render_md(sheet, inst, names, dyn, seed=args.seed, cost_anchor=anchor)
    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text(md, encoding="utf-8")
    # machine-readable artefact (the test checks parity without running a solver); outside git (#1)
    Path(args.out_json).write_text(json.dumps({
        "seed": args.seed, "cost_eur": sheet["cost_eur"], "anchor_per_seed": anchor,
        "totals": sheet["totals"], "names_present": bool(names),
        "dynamics": {k: dyn[k] for k in ("seed", "event_clock", "n_served", "n_pending",
                                         "latency_ms", "center_stop", "n_moved")},
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"→ {args.out_md}  (cost {sheet['cost_eur']:.1f}€ = per_seed[{args.seed}] "
          f"{anchor:.1f}€ ✓, {sheet['totals']['vehicles_used']} vehicles, "
          f"{sheet['totals']['n_stops']} stops, names {'yes' if names else 'no (stop-id)'})")
    print(f"→ {args.out_json}")


if __name__ == "__main__":
    main()
