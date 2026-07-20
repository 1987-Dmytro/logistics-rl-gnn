"""OSMnx-пайплайн Аугсбурга: граф, скорости, аптеки, депо, снаппинг, матрицы.

Всё сетевое (download/geocode) — только здесь. snapshot.load_snapshot работает offline.
API проверен на osmnx 2.1.0.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd

from logistics_rl_gnn.config import data as cfg


def load_drive_graph(place: str = cfg.PLACE) -> nx.MultiDiGraph:
    """Drive-граф места, усечённый до наибольшей СИЛЬНО-связной компоненты (routable)."""
    g = ox.graph_from_place(place, network_type="drive")
    return ox.truncate.largest_component(g, strongly=True)


def impute_speeds(g: nx.MultiDiGraph) -> tuple[nx.MultiDiGraph, float]:
    """Проставить speed_kph/travel_time с немецким фолбэком.

    Возвращает (граф, % рёбер с РЕАЛЬНЫМ тегом maxspeed до импутации).
    """
    total = tagged = 0
    for _, _, d in g.edges(data=True):
        total += 1
        if d.get("maxspeed") not in (None, "", []):
            tagged += 1
    coverage_pct = 100.0 * tagged / total if total else 0.0
    g = ox.add_edge_speeds(g, hwy_speeds=cfg.GERMAN_HWY_SPEEDS, fallback=cfg.FALLBACK_SPEED_KMH)
    g = ox.add_edge_travel_times(g)
    return g, coverage_pct


def load_pharmacies(place: str = cfg.PLACE):
    """GeoDataFrame аптек (amenity=pharmacy) в границах места.

    Добавляет колонки x=lon, y=lat (репрезентативная точка на геометрию).
    """
    gdf = ox.features_from_place(place, tags=cfg.PHARMACY_TAGS)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    if "opening_hours" not in gdf.columns:  # колонка есть только если хоть одна аптека с тегом
        gdf = gdf.assign(opening_hours=pd.NA)
    pts = gdf.geometry.representative_point()
    return gdf.assign(x=pts.x.to_numpy(), y=pts.y.to_numpy())


def geocode_depot(addr: str = cfg.DEPOT_ADDR) -> tuple[float, float]:
    """(lat, lon) депо по адресу (Nominatim)."""
    return ox.geocode(addr)


def snap_to_nodes(g, points, *, return_dist: bool = False):
    """Ближайшие OSM-узлы к точкам points=[(lon, lat), ...] (X=lon, Y=lat)."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return ox.nearest_nodes(g, X=xs, Y=ys, return_dist=return_dist)


def build_matrices(g, stop_nodes):
    """All-pairs матрицы времени (travel_time, c) и дистанции (length, м) ПО СТОПАМ.

    Индексация по СТОПАМ, не по уникальным узлам: stop_nodes может содержать
    повторяющиеся OSM-узлы (аптеки, снапнутые в один узел) — такие со-узловые стопы
    остаются ОТДЕЛЬНЫМИ строками с Δ≈0 между собой (src==tgt-узел → dijkstra=0).
    Квадратные (k×k, k=len(stop_nodes)), диагональ 0. Требует routable-граф → значения конечны.
    """
    k = len(stop_nodes)
    time_m = np.zeros((k, k), dtype=float)
    dist_m = np.zeros((k, k), dtype=float)
    # single-source Dijkstra кэшируем по УНИКАЛЬНЫМ узлам (co-node стопы не пересчитываем)
    t_cache: dict = {}
    d_cache: dict = {}
    for u in set(stop_nodes):
        t_cache[u] = nx.single_source_dijkstra_path_length(g, u, weight="travel_time")
        d_cache[u] = nx.single_source_dijkstra_path_length(g, u, weight="length")
    for i, src in enumerate(stop_nodes):
        t, dlen = t_cache[src], d_cache[src]
        for j, tgt in enumerate(stop_nodes):
            time_m[i, j] = t.get(tgt, np.inf)
            dist_m[i, j] = dlen.get(tgt, np.inf)
    return time_m, dist_m
