"""Phase 8 — demo.py guard: end-to-end on seed 0. The static cost is DETERMINISTIC → exact parity
with system_metrics per_seed[0]; the dynamics (residual/latency) are wall-clock → pinned ONLY
structurally (files exist, served+pending == all customers per 0004, fields in range).

Heavy (full system_routes + re-plan + OR-Tools ~30s) → skipif without ckpt/snapshot/durable (#1).
demo pulls torch/opening_hours → importorskip gates the module on a bare runner.
"""

from __future__ import annotations

import json
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
_SNAP = _ROOT / "data" / "snapshots" / "augsburg_20260720"
_NEED = _CKPT.exists() and _SM.exists() and (_SNAP / "meta.json").exists()


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

    # compare.html — the A/B/C frame: links both maps AND shows the OR-Tools cost
    # (prohibition #3: the honest baseline is not hidden in a presentation artefact)
    comp = Path(by_name["compare.html"]).read_text(encoding="utf-8")
    assert "2_incident_no_replan.html" in comp and "3_incident_replan.html" in comp
    assert f"{s['or_cost']:.1f} €" in comp

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
