"""Saving/loading the Augsburg snapshot. load_snapshot is offline (no network)."""

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
    """A loaded snapshot. graph=None when with_graph=False."""

    meta: dict
    node_ids: list[int]  # OSM node id PER STOP (in stop order; may repeat for co-located stops)
    time_matrix: np.ndarray
    dist_matrix: np.ndarray
    nodes: pd.DataFrame  # columns: stop, kind (depot|pharmacy), node_id, x, y
    graph: nx.MultiDiGraph | None


def save_snapshot(out_dir, g, stop_nodes, time_matrix, dist_matrix, nodes_df, meta) -> Path:
    """graphml + matrices (parquet) + nodes (parquet) + meta.json.

    Matrices are labelled by stop POSITION (0..k-1) rather than node id: node ids of co-located
    stops repeat and cannot serve as unique column names. stop→node_id lives in nodes_df.
    """
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    ox.save_graphml(g, d / "graph.graphml")
    cols = [str(i) for i in range(len(stop_nodes))]  # stop positions (unique)
    pd.DataFrame(time_matrix, index=cols, columns=cols).to_parquet(d / "time_matrix.parquet")
    pd.DataFrame(dist_matrix, index=cols, columns=cols).to_parquet(d / "dist_matrix.parquet")
    nodes_df.to_parquet(d / "nodes.parquet")
    (d / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return d


def load_snapshot(in_dir, *, with_graph: bool = True) -> Snapshot:
    """Reads the snapshot back WITHOUT network access."""
    d = Path(in_dir)
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    tdf = pd.read_parquet(d / "time_matrix.parquet")
    ddf = pd.read_parquet(d / "dist_matrix.parquet")
    nodes = pd.read_parquet(d / "nodes.parquet").sort_values("stop").reset_index(drop=True)
    node_ids = [int(n) for n in nodes["node_id"]]  # per-stop node id (in stop order)
    graph = ox.load_graphml(d / "graph.graphml") if with_graph else None
    return Snapshot(meta, node_ids, tdf.to_numpy(), ddf.to_numpy(), nodes, graph)
