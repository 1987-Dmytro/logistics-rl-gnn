"""Phase 2 pipeline: OSM Augsburg → data/snapshots/augsburg_<YYYYMMDD>/. NOT a test.

Requires network access (Overpass/Nominatim). Prints counters and the maxspeed coverage %.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd

from logistics_rl_gnn.config import data as cfg
from logistics_rl_gnn.data import osm, snapshot


def main() -> None:
    print(f"[1/5] drive graph {cfg.PLACE} …")
    g = osm.load_drive_graph(cfg.PLACE)
    print(f"      nodes={g.number_of_nodes()} edges={g.number_of_edges()}")

    print("[2/5] speed imputation (German fallback) …")
    g, coverage = osm.impute_speeds(g)
    print(f"      maxspeed coverage (real tag) = {coverage:.1f}%")

    print("[3/5] pharmacies + depot …")
    ph = osm.load_pharmacies(cfg.PLACE)
    n_ph = len(ph)
    oh_col = ph["opening_hours"].astype("string").str.strip()
    oh_present = oh_col.notna() & (oh_col != "")
    oh_present_pct = round(100.0 * int(oh_present.sum()) / n_ph, 2) if n_ph else 0.0
    ph_oh = ph["opening_hours"].to_numpy()
    depot_lat, depot_lon = osm.geocode_depot(cfg.DEPOT_ADDR)
    print(f"      pharmacies={n_ph} (opening_hours={oh_present_pct}%)")
    print(f"      depot=({depot_lat:.5f}, {depot_lon:.5f})")

    print("[4/5] snapping to nodes …")
    ph_points = list(zip(ph["x"].to_numpy(), ph["y"].to_numpy(), strict=True))  # (lon, lat)
    ph_nodes = [int(n) for n in osm.snap_to_nodes(g, ph_points)]
    depot_arr, depot_dist_arr = osm.snap_to_nodes(g, [(depot_lon, depot_lat)], return_dist=True)
    depot_node = int(depot_arr[0])
    depot_dist = float(depot_dist_arr[0])
    if depot_dist > 500:
        print(f"      ⚠ depot snapped {depot_dist:.0f} m away (Stadtbergen is outside Augsburg)")

    # collision diagnostics: how many pharmacies share one nearest node
    node_counts = Counter(ph_nodes)
    shared = [n for n, c in node_counts.items() if c > 1]
    collapsed = sum(c - 1 for c in node_counts.values() if c > 1)
    print(
        f"      collision: {collapsed} co-located pharmacies on {len(shared)} shared nodes; "
        f"unique nodes={len(node_counts)}/{len(ph_nodes)} pharmacies"
    )
    if depot_node in node_counts:
        print("      the depot shares a node with a pharmacy but stays a SEPARATE stop (stop 0)")

    # INDEXED BY STOP (not by set(nearest_nodes)): the depot and every pharmacy get their own row.
    # Co-located stops are kept apart (Δ≈0 between them), the matrix is NOT collapsed.
    stop_nodes = [depot_node, *ph_nodes]  # length = 1 + n_pharmacies

    print(f"[5/5] all-pairs matrices per stop ({len(stop_nodes)}×{len(stop_nodes)}) …")
    time_m, dist_m = osm.build_matrices(g, stop_nodes)
    assert time_m.shape == (1 + n_ph, 1 + n_ph), "matrix collapsed — not indexed by stop!"

    depot_row = {
        "stop": 0,
        "kind": "depot",
        "node_id": depot_node,
        "x": depot_lon,
        "y": depot_lat,
        "opening_hours": None,
    }
    rows = [depot_row]
    for i, ((x, y), n) in enumerate(zip(ph_points, ph_nodes, strict=True), start=1):
        oh = ph_oh[i - 1]
        rows.append(
            {
                "stop": i,
                "kind": "pharmacy",
                "node_id": n,
                "x": x,
                "y": y,
                "opening_hours": None if pd.isna(oh) else str(oh),
            }
        )
    nodes_df = pd.DataFrame(rows)

    date = datetime.now().strftime("%Y%m%d")
    meta = {
        "date": date,
        "place": cfg.PLACE,
        "depot_addr": cfg.DEPOT_ADDR,
        "n_pharmacies": n_ph,
        "n_nodes": g.number_of_nodes(),
        "maxspeed_coverage_pct": round(coverage, 2),
        "opening_hours_present_pct": oh_present_pct,
    }
    out = Path("data/snapshots") / f"augsburg_{date}"
    snapshot.save_snapshot(out, g, stop_nodes, time_m, dist_m, nodes_df, meta)
    print(f"\n✅ snapshot → {out}")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
