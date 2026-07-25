"""Phase 8 — demo.py guard: end-to-end on seed 0. The static cost is DETERMINISTIC → exact parity
with system_metrics per_seed[0]; the dynamics (residual/latency) are wall-clock → pinned ONLY
structurally (files exist, served+pending == all customers per 0004, fields in range).

Heavy (full system_routes + re-plan + OR-Tools ~30s) → skipif without ckpt/snapshot/durable (#1).
demo pulls torch/opening_hours → importorskip gates the module on a bare runner.
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import pytest

_FILES = {"1_morning_plan.html", "route_sheet.md", "2_incident_no_replan.html",
          "3_incident_replan.html", "compare.html"}


def _banner_price(path) -> float:
    """The price from the map header (data-demo-price) — machine-readable 'headers == output'."""
    m = re.search(r'data-demo-price="([0-9.]+)"', Path(path).read_text(encoding="utf-8"))
    assert m, f"no data-demo-price in {path}"
    return float(m.group(1))

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "scripts"))

demo = pytest.importorskip("demo")  # skips on a runner without torch/opening_hours

_CKPT = _ROOT / "results" / "policy_pomo_congestion.pt"
_SM = _ROOT / "results" / "system_metrics.json"
_POLISH = _ROOT / "results" / "polish_summary.json"  # the dashboard cites its durable aggregates
_SNAP = _ROOT / "data" / "snapshots" / "augsburg_20260720"
_NEED = _CKPT.exists() and _SM.exists() and _POLISH.exists() and (_SNAP / "meta.json").exists()
_NUM = r"\d+(?:[.,]\d+)?"  # any number printed in the dashboard


@pytest.mark.skipif(not _NEED, reason="ckpt/snapshot/system_metrics outside git (#1)")
def test_demo_end_to_end_seed0(tmp_path):
    """demo runs end-to-end (seed 0, traffic): files exist, static cost parity per_seed[0],
    served+pending == all customers (0004 mechanics). The dynamics are NOT pinned exactly."""
    from logistics_rl_gnn.config import instance as im

    s = demo.run_demo(seed=0, event_kind="traffic", out_dir=str(tmp_path), open_maps=False)

    # statics are deterministic → EXACT parity with the durable per_seed[0]
    sm = json.loads(_SM.read_text())
    assert abs(s["static_cost"] - sm["per_seed_cost_eur"][0]) < 0.5

    # 5 output artefacts with self-describing names — they exist and are non-empty
    by_name = {Path(f).name: f for f in s["files"]}
    assert set(by_name) == _FILES
    for f in s["files"]:
        assert Path(f).exists() and Path(f).stat().st_size > 0

    # headers == the demo output (parse data-demo-price out of every map)
    assert abs(_banner_price(by_name["1_morning_plan.html"]) - s["morning_cost"]) < 1e-3
    assert abs(_banner_price(by_name["2_incident_no_replan.html"]) - s["cost_before"]) < 1e-3
    assert abs(_banner_price(by_name["3_incident_replan.html"]) - s["cost_after"]) < 1e-3

    # compare.html — the A/B/G/C frame: links both maps AND shows the OR-Tools cost
    # (prohibition #3: the honest baseline is not hidden in a presentation artefact)
    comp = Path(by_name["compare.html"]).read_text(encoding="utf-8")
    assert "2_incident_no_replan.html" in comp and "3_incident_replan.html" in comp
    assert f"{s['or_cost']:.1f} €" in comp

    # --- dashboard honesty (the point of the frame, not decoration) ---
    dash = s["dashboard"]
    rows = {r["k"]: r for r in dash["rows"]}
    assert list(rows) == ["A", "B", "G", "C"]  # the greedy re-plan is a ROW, not a footnote
    assert "greedy re-plan" in rows["G"]["label"]
    tbl = comp[comp.index("<table>"):comp.index("</table>")]
    assert tbl.count("<tr>") == len(dash["rows"]) + 1  # + the header row (region is non-empty)
    assert "<th>unserved</th>" in tbl and "<th>on-time %</th>" in tbl
    # every cell traced to THIS run's scorer output (one residual world, one scorer)
    q = s["replan_quality"]
    for k, r in rows.items():
        assert r["unserved"] == str(int(q[k]["unserved"]))
        assert r["on_time"] == f"{q[k]['on_time_pct']:.0f} %"
        cost = -q[k]["reward"]
        pen = int(q[k]["unserved"]) * 200.0  # p_unserved (im.PENALTY_UNSERVED)
        if math.isfinite(cost):  # a blocked old plan (∞) is priceless, not splittable
            assert r["cost"].startswith(f"{cost:.1f} €")
            assert (f"incl. {pen:.0f} € unserved penalty" in r["cost"]) == (pen > 0)
        else:
            assert r["cost"] == "∞ €" and "incl." not in r["cost"]
    # when the portfolio winner is the greedy candidate, C differs from G by POLISH only — said
    # out loud, otherwise the C−G delta reads as the model's contribution (it is not)
    if "'greedy heuristic' won" in rows["C"]["note"]:
        assert "C = G + local-search polish" in rows["C"]["note"]
    assert abs(-q["G"]["reward"] - s["greedy_cost"]) < 1e-9
    assert abs(-q["C"]["reward"] - s["cost_after"]) < 1e-9
    # the ≤ greedy guarantee (dec-0008) — also proves G is scored in the SAME world as C
    assert s["cost_after"] <= s["greedy_cost"] + 1e-6
    # no hand-typed numbers in the HTML: the table prints only what the row data holds
    cells = " ".join(v for r in dash["rows"] for v in r.values())
    assert set(re.findall(_NUM, tbl)) <= set(re.findall(_NUM, cells))
    # B is marked as budget-constrained, and the ~30 s caveat is not hidden
    assert "†" in rows["B"]["note"] and any("†" in n and "30 s" in n for n in dash["notes"])
    # the headline names BOTH deltas (vs inaction and vs the greedy re-plan)
    assert f"{abs(s['savings']):.1f} € vs inaction (this event)" in dash["headline"]
    assert f"vs greedy re-plan: {'−' if s['cost_after'] < s['greedy_cost'] else '+'}" \
           f"{abs(s['cost_after'] - s['greedy_cost']):.1f} €" in dash["headline"]
    # durable citations == results/polish_summary.json (a single event is never the general case)
    dyn = json.loads(_POLISH.read_text())["dynamic"]
    d = dash["durable"]
    assert d["system_eur"] == dyn["aggregates"]["rl"]["cost_eur_mean"]
    assert d["greedy_eur"] == dyn["aggregates"]["greedy"]["cost_eur_mean"]
    assert d["median_delta_eur"] == dyn["guarantee"]["median_delta_eur"]
    assert d["n_events"] == dyn["guarantee"]["n_events"] == 25
    assert f"{d['system_eur']:.1f} €" in dash["footer"] and f"{d['greedy_eur']:.1f} €" in comp
    # EN only — the German duplication is gone from the frame
    for de in ("Straßensperrung", "Stau", "Ausfall", "Eilauftrag", "Sperrung"):
        assert de not in comp

    # 0004 mechanics: served + remainder == all customers (traffic: no urgent re-delivery)
    inst = im.generate_instance(snapshot_dir=_SNAP, seed=0)
    assert s["n_served"] + s["n_pending"] == len(inst.demand) - 1

    # cost_before is DETERMINISTIC (greedy+served_by+_continue_old_plan+scorer, no polish)
    # → pin the durable value (catches numbering desync/swap/Δ sign, which range asserts miss)
    assert abs(s["cost_before"] - 578.7) < 2.0
    # cost_after is wall-clock-sensitive (polish 400ms) → sanity only (finite, positive)
    assert 0.0 < s["cost_after"] < 5000.0
    assert s["n_moved"] <= s["n_pending"]

    # the model contribution is measured on BOTH sides and by different mechanisms: the day plan
    # polishes every candidate in one run, the re-plan uses a SEPARATE portfolio with a full budget.
    c = s["contribution"]
    assert all(c[k] is not None for k in ("plan_without", "plan_with", "replan_without",
                                          "replan_with"))
    assert c["plan_with"] <= c["plan_without"] + 1e-9  # the model only ADDS candidates to the pool
    # the re-plan sign is NOT pinned: 'the model made it worse' is an admissible outcome (dec-0012),
    # which is exactly why the 'without the model' side is measured separately
    assert c["replan_without"] != s["cost_before"]


def _q(cost, unserved=0, on_time=100.0):
    """A minimal evaluate_solution dict (cost = −reward) — the dashboard reads only these keys."""
    return {"reward": -cost, "unserved": unserved, "on_time_pct": on_time}


_FAKE_DUR = {"lat": {"rl": 689.0, "greedy": 7.0, "ortools": 2001.0},
             "cost": {"rl": 827.3, "greedy": 851.5, "ortools": 866.9},
             "n_events": 25, "median_delta": -16.8, "violations": 0}


def _dash(*, before=578.7, greedy=508.0, system=495.9, uns_a=2):
    return demo._dashboard(
        [("A", "do-nothing", _q(before, uns_a, 60.0), None, "delays accumulate"),
         ("B", "OR-Tools re-solve", _q(826.5), 2001.0, "from scratch †"),
         ("G", "greedy re-plan (heuristic)", _q(greedy), 7.0, "the no-ML reaction"),
         ("C", "our system", _q(system), 689.0, "portfolio+polish")],
        dur=_FAKE_DUR, seed=0, p_unserved=200.0)


def test_dashboard_shows_the_whole_picture():
    """No artefacts needed: the dashboard model must carry the greedy row, the unserved/on-time
    columns, the named penalty share, both deltas and the durable 25-event context."""
    d = _dash()
    assert [r["k"] for r in d["rows"]] == ["A", "B", "G", "C"]  # greedy sits next to the system
    a, c = d["rows"][0], d["rows"][3]
    assert a["cost"] == "578.7 € (incl. 400 € unserved penalty)"  # the split is named, not hidden
    assert a["unserved"] == "2" and a["on_time"] == "60 %"
    assert c["cost"] == "495.9 €" and "incl." not in c["cost"]  # nothing unserved → no fake split
    assert d["rows"][1]["lat"] == "2.0 s (2001 ms)" and d["rows"][2]["lat"] == "7 ms"
    assert a["lat"] == "0 s (no reaction)"
    assert "−82.8 € vs inaction (this event)" in d["headline"]
    assert "vs greedy re-plan: −12.1 € (this event)" in d["headline"]
    assert "827.3 € vs greedy re-plan 851.5 €" in d["footer"]
    assert "median saving −16.8 €/event" in d["footer"]
    assert "never worse than greedy by construction (0/25 violations)" in d["footer"]
    assert any("30 s" in n and "†" in n for n in d["notes"])  # B's budget caveat
    assert any("VISITED" in n for n in d["notes"])  # on-time% is over visited stops


def test_dashboard_sign_and_infinity_are_not_assumed():
    """The system may be DEARER than greedy (dec-0012 is exactly that outcome) — the headline must
    print '+'. And inaction may be impossible (∞): no penalty split and no '−∞' saving."""
    worse = _dash(greedy=490.0, system=495.9)
    assert "vs greedy re-plan: +5.9 € (this event)" in worse["headline"]
    blocked = _dash(before=float("inf"), uns_a=3)
    assert blocked["rows"][0]["cost"] == "∞ €"  # splitting a penalty out of ∞ invents a number
    assert "Inaction is impossible" in blocked["headline"] and "−∞" not in blocked["headline"]
    assert "vs greedy re-plan:" in blocked["headline"]


def test_continue_old_plan_numbering():
    """A policy-free guard against numbering desync in ~1s: _continue_old_plan numbers by the GIVEN
    residual idx (== residual_instance) and excludes served (urgent re-delivery). idx is NOT the
    identity and has a hole → pos[b]≠b: a raw-index bug (append b, not pos[b]) would differ."""
    class _St:  # a minimal DynamicState (only .served is needed)
        served = {2, 3}  # 2 — urgent re-delivery (in idx but served); 3 — plain served (not in idx)

    idx = [0, 1, 2, 4]  # residual numbering: pending {1,4} ∪ urgent {2}; stop 3 served → not in idx
    exec_routes = [[0, 1, 2, 3, 0], [0, 4, 0]]  # veh.1 → 1,2,3; veh.2 → 4
    veh_of = {1: 1, 2: 1, 3: 1, 4: 2}
    old = demo._continue_old_plan(exec_routes, _St(), idx=idx, drop_vehicle=None, veh_of=veh_of)
    visited = {p for r in old for p in r if p != 0}
    # positions in idx: stop 1→pos1, stop 4→pos3. A raw-index bug would give {1,4} — caught here.
    assert visited == {1, 3}
    assert 2 not in visited  # a served urgent re-delivery (in idx) is NOT in the old plan
    # breakdown: vehicle 2 is lost → its stop (4, idx pos 3) is orphaned; only stop 1 (pos 1) stays
    bd = demo._continue_old_plan(exec_routes, _St(), idx=idx, drop_vehicle=2, veh_of=veh_of)
    assert {p for r in bd for p in r if p != 0} == {1}


@pytest.mark.skipif(not (_SNAP / "graph.graphml").exists(),
                    reason="graph.graphml outside git (#1)")
def test_road_polyline_on_graph_nodes():
    """Road geometry: the hop polyline follows real streets (nx.shortest_path) — a real path
    (>2 vertices, not the straight fallback) AND every vertex LIES on a graph node (node x/y)."""
    import osmnx as ox
    import pandas as pd

    g = ox.load_graphml(_SNAP / "graph.graphml")
    coords = {(round(g.nodes[p]["y"], 7), round(g.nodes[p]["x"], 7)) for p in g.nodes}
    ids = pd.read_parquet(_SNAP / "nodes.parquet")["node_id"].astype(int).tolist()
    line = demo._road_latlon(g, ids[0], ids[1], {})  # depot→pharmacy connected → a real path
    assert len(line) > 2, "expected a real street path, not the straight fallback"
    for lat, lon in line:
        assert (round(lat, 7), round(lon, 7)) in coords  # the vertex == a graph node
