"""Phase 8 — аддитивное обогащение снапшота именами/адресами аптек (для route sheet).

nodes.parquet несёт координаты, но НЕ имена. Тянем POI аптек (`features_from_place`, ТЕ ЖЕ теги, что
build_snapshot) → матч к снапшот-аптекам по КООРДИНАТЕ (representative_point, порог `--max-dist-m`)
→ names.parquet (stop, name, addr, osm_id, dist_m). Матрицы/стопы/окна НЕ трогаем (парити 631.6€
охраняется) — только meta.json += names_present. Требует сеть (Overpass). Route sheet без файла →
фолбэк на stop-id (аддитивно, не ломает). Синтетику НЕ метим — данные РЕАЛЬНЫЕ (запрет №5).

Запуск: python scripts/enrich_names.py [--snap DIR] [--max-dist-m 60]
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
    """Склейка «Улица Дом, Индекс Город» из addr:*-тегов (что есть)."""
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

    poi = osm.load_pharmacies(cfg.PLACE).reset_index()  # x=lon, y=lat + name/addr-теги
    px, py = poi["x"].to_numpy(), poi["y"].to_numpy()

    rows = []
    for r in ph.itertuples(index=False):
        # ближайший POI к снапшот-координате (repr. point совпадает, если геометрия не менялась)
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
    ap = argparse.ArgumentParser(description="Phase 8 — имена аптек (аддитивно)")
    ap.add_argument("--snap", default=None, help="каталог снапшота (по умолч. — последний)")
    ap.add_argument("--max-dist-m", type=float, default=60.0, help="порог координатного матча, м")
    args = ap.parse_args()

    snap = Path(args.snap) if args.snap else im._latest_snapshot_dir()
    if snap is None:
        raise FileNotFoundError("нет снапшота — сначала `python scripts/build_snapshot.py`")

    df = enrich(snap, max_dist_m=args.max_dist_m)
    n_named = int(df["name"].notna().sum())
    df.to_parquet(snap / "names.parquet")

    meta_p = snap / "meta.json"
    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    meta["names_present"] = True
    meta["names_matched"] = n_named  # аддитивно; матрицы/стопы не тронуты
    meta_p.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"→ {snap / 'names.parquet'}  ({n_named}/{len(df)} аптек с именем, "
          f"матч ≤ {args.max_dist_m:.0f} м)")
    print(f"→ {meta_p}  (+names_present)")


if __name__ == "__main__":
    main()
