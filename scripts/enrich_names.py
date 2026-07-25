"""Phase 8 — additive enrichment of the snapshot with pharmacy names/addresses (for the sheet).

nodes.parquet carries coordinates but NOT names. We pull pharmacy POIs (`features_from_place`, THE
SAME tags as build_snapshot) → match to snapshot pharmacies BY COORDINATE (representative_point,
threshold `--max-dist-m`) → names.parquet (stop, name, addr, osm_id, dist_m). Matrices/stops/
windows are untouched (the 631.6€ parity is protected) — only meta.json += names_present. Requires
network (Overpass). No file → the route sheet falls back to stop-ids (additive, nothing breaks).
Nothing is tagged synthetic — the data is REAL (prohibition #5).

Run: python scripts/enrich_names.py [--snap DIR] [--max-dist-m 60]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

from logistics_rl_gnn.config import data as cfg
from logistics_rl_gnn.config import instance as im
from logistics_rl_gnn.data import osm


def _haversine_m(lon1, lat1, lon2, lat2) -> float:
    r = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _addr(row) -> str | None:
    """Glue 'Street House, Postcode City' out of the addr:* tags (whichever exist)."""
    street = row.get("addr:street")
    house = row.get("addr:housenumber")
    city = row.get("addr:city")
    postcode = row.get("addr:postcode")
    line1 = " ".join(str(x) for x in (street, house) if pd.notna(x) and str(x).strip())
    line2 = " ".join(str(x) for x in (postcode, city) if pd.notna(x) and str(x).strip())
    full = ", ".join(x for x in (line1, line2) if x)
    return full or None


def _osm_id(row) -> str | None:
    for et, idc in (("element_type", "osmid"), ("element", "id")):
        if et in row and idc in row and pd.notna(row[idc]):
            return f"{row[et]}/{int(row[idc])}"
    return None


def enrich(snap_dir: Path, *, max_dist_m: float) -> pd.DataFrame:
    nodes = pd.read_parquet(snap_dir / "nodes.parquet")
    ph = nodes[nodes.kind == "pharmacy"][["stop", "x", "y"]].reset_index(drop=True)

    poi = osm.load_pharmacies(cfg.PLACE).reset_index()  # x=lon, y=lat + name/addr tags
    px, py = poi["x"].to_numpy(), poi["y"].to_numpy()

    rows = []
    for r in ph.itertuples(index=False):
        # nearest POI to the snapshot coordinate (repr. point matches if geometry did not change)
        j, best = -1, float("inf")
        for k in range(len(poi)):
            d = _haversine_m(r.x, r.y, float(px[k]), float(py[k]))
            if d < best:
                j, best = k, d
        matched = best <= max_dist_m
        prow = poi.iloc[j] if matched else None
        name = prow.get("name") if matched else None
        rows.append({
            "stop": int(r.stop),
            "name": None if (name is None or pd.isna(name)) else str(name),
            "addr": _addr(prow) if matched else None,
            "osm_id": _osm_id(prow) if matched else None,
            "dist_m": round(best, 1) if matched else None,
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 8 — pharmacy names (additive)")
    ap.add_argument("--snap", default=None, help="snapshot directory (default — the latest)")
    ap.add_argument("--max-dist-m", type=float, default=60.0, help="coordinate match threshold, m")
    args = ap.parse_args()

    snap = Path(args.snap) if args.snap else im._latest_snapshot_dir()
    if snap is None:
        raise FileNotFoundError("no snapshot — run `python scripts/build_snapshot.py` first")

    df = enrich(snap, max_dist_m=args.max_dist_m)
    n_named = int(df["name"].notna().sum())
    df.to_parquet(snap / "names.parquet")

    meta_p = snap / "meta.json"
    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    meta["names_present"] = True
    meta["names_matched"] = n_named  # additive; matrices/stops untouched
    meta_p.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"→ {snap / 'names.parquet'}  ({n_named}/{len(df)} pharmacies named, "
          f"match ≤ {args.max_dist_m:.0f} m)")
    print(f"→ {meta_p}  (+names_present)")


if __name__ == "__main__":
    main()
