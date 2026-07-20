"""Сохранение/загрузка снапшота Аугсбурга. load_snapshot — offline (без сети)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd


@dataclass
class Snapshot:
    """Загруженный снапшот. graph=None, если with_graph=False."""

    meta: dict
    node_ids: list[int]  # OSM node id ПО СТОПАМ (по порядку stop; может повторяться у со-узловых)
    time_matrix: np.ndarray
    dist_matrix: np.ndarray
    nodes: pd.DataFrame  # колонки: stop, kind (depot|pharmacy), node_id, x, y
    graph: nx.MultiDiGraph | None


def save_snapshot(out_dir, g, stop_nodes, time_matrix, dist_matrix, nodes_df, meta) -> Path:
    """graphml + матрицы (parquet) + узлы (parquet) + meta.json.

    Матрицы метятся стоп-ПОЗИЦИЯМИ (0..k-1), а не node-id: node-id со-узловых стопов
    повторяются и не годятся в уникальные имена колонок. stop→node_id хранится в nodes_df.
    """
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    ox.save_graphml(g, d / "graph.graphml")
    cols = [str(i) for i in range(len(stop_nodes))]  # стоп-позиции (уникальны)
    pd.DataFrame(time_matrix, index=cols, columns=cols).to_parquet(d / "time_matrix.parquet")
    pd.DataFrame(dist_matrix, index=cols, columns=cols).to_parquet(d / "dist_matrix.parquet")
    nodes_df.to_parquet(d / "nodes.parquet")
    (d / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return d


def load_snapshot(in_dir, *, with_graph: bool = True) -> Snapshot:
    """Читает снапшот обратно БЕЗ сети."""
    d = Path(in_dir)
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    tdf = pd.read_parquet(d / "time_matrix.parquet")
    ddf = pd.read_parquet(d / "dist_matrix.parquet")
    nodes = pd.read_parquet(d / "nodes.parquet").sort_values("stop").reset_index(drop=True)
    node_ids = [int(n) for n in nodes["node_id"]]  # per-stop node id (в порядке stop)
    graph = ox.load_graphml(d / "graph.graphml") if with_graph else None
    return Snapshot(meta, node_ids, tdf.to_numpy(), ddf.to_numpy(), nodes, graph)
