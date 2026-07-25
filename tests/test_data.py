"""Offline test of the Phase 2 snapshot (no network access)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from logistics_rl_gnn.config import data as cfg
from logistics_rl_gnn.data import snapshot

SNAP_ROOT = Path("data/snapshots")


def _latest_snapshot():
    if not SNAP_ROOT.is_dir():
        return None
    dirs = sorted(p for p in SNAP_ROOT.glob("augsburg_*") if (p / "meta.json").is_file())
    return dirs[-1] if dirs else None


@pytest.fixture(scope="module")
def snap():
    d = _latest_snapshot()
    if d is None:
        pytest.skip("no snapshot: run `python scripts/build_snapshot.py` first (needs network)")
    return snapshot.load_snapshot(d, with_graph=False)


def test_matrices_valid(snap):
    assert len(snap.node_ids) > 0
    for name, m in (("time", snap.time_matrix), ("dist", snap.dist_matrix)):
        assert m.ndim == 2 and m.shape[0] == m.shape[1], f"{name}: not square"
        assert m.shape[0] == len(snap.node_ids), f"{name}: size ≠ node_ids"
        assert np.isfinite(m).all(), f"{name}: inf/nan (graph not strongly connected?)"
        assert (m >= 0).all(), f"{name}: negative weights"
        assert np.allclose(np.diag(m), 0.0), f"{name}: diagonal is not 0"


def test_matrix_dim_is_stops(snap):
    # the matrix is square AND size = n_depot + n_pharmacies (BY STOP, not by unique node)
    n_depot = int((snap.nodes["kind"] == "depot").sum())
    n_pharm = int((snap.nodes["kind"] == "pharmacy").sum())
    assert n_depot == 1, "the depot is not unique"
    dim = n_depot + n_pharm
    for name, m in (("time", snap.time_matrix), ("dist", snap.dist_matrix)):
        assert m.shape[0] == m.shape[1] == dim, f"{name}: dim ≠ 1+n_pharm (collapsed?)"
    assert n_depot + n_pharm == 1 + snap.meta["n_pharmacies"]


def test_conode_stops_zero_delta(snap):
    # co-located stops (same node_id) stay separate rows with Δ=0 between them
    first_seen: dict[int, int] = {}
    conode_pairs = 0
    for i, n in enumerate(snap.node_ids):
        if n in first_seen:
            j = first_seen[n]
            conode_pairs += 1
            assert snap.time_matrix[i, j] == 0 and snap.time_matrix[j, i] == 0
            assert snap.dist_matrix[i, j] == 0 and snap.dist_matrix[j, i] == 0
        else:
            first_seen[n] = i
    # unique nodes < stops ⇔ there were collisions (the Augsburg data has them)
    assert len(first_seen) == len(snap.node_ids) - conode_pairs


def test_pharmacy_count(snap):
    lo, hi = cfg.PHARMACY_COUNT_RANGE
    assert lo <= snap.meta["n_pharmacies"] <= hi


def test_depot_resolved(snap):
    depot = snap.nodes[snap.nodes["kind"] == "depot"]
    assert len(depot) == 1, "the depot row is not unique"
    assert int(depot.iloc[0]["node_id"]) in set(snap.node_ids), "depot node not in the matrix"


def test_meta_coverage(snap):
    cov = snap.meta["maxspeed_coverage_pct"]
    assert cov is not None and 0 < cov <= 100, "maxspeed coverage outside (0, 100]"


def test_opening_hours(snap):
    assert "opening_hours" in snap.nodes.columns, "no opening_hours column"
    pct = snap.meta["opening_hours_present_pct"]
    assert 0 <= pct <= 100, "opening_hours_present_pct outside [0, 100]"
