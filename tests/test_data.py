"""Offline-тест снапшота Phase 2 (сеть не трогаем)."""

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
        pytest.skip("нет снапшота: сначала `python scripts/build_snapshot.py` (нужна сеть)")
    return snapshot.load_snapshot(d, with_graph=False)


def test_matrices_valid(snap):
    assert len(snap.node_ids) > 0
    for name, m in (("time", snap.time_matrix), ("dist", snap.dist_matrix)):
        assert m.ndim == 2 and m.shape[0] == m.shape[1], f"{name}: не квадратная"
        assert m.shape[0] == len(snap.node_ids), f"{name}: размер ≠ node_ids"
        assert np.isfinite(m).all(), f"{name}: inf/nan (граф не сильно-связный?)"
        assert (m >= 0).all(), f"{name}: отрицательные веса"
        assert np.allclose(np.diag(m), 0.0), f"{name}: диагональ не 0"


def test_matrix_dim_is_stops(snap):
    # матрица квадратная И размер = n_depot + n_pharmacies (по СТОПАМ, не по уникальным узлам)
    n_depot = int((snap.nodes["kind"] == "depot").sum())
    n_pharm = int((snap.nodes["kind"] == "pharmacy").sum())
    assert n_depot == 1, "депо не единственный"
    dim = n_depot + n_pharm
    for name, m in (("time", snap.time_matrix), ("dist", snap.dist_matrix)):
        assert m.shape[0] == m.shape[1] == dim, f"{name}: dim ≠ 1+n_pharm (схлопнута?)"
    assert n_depot + n_pharm == 1 + snap.meta["n_pharmacies"]


def test_conode_stops_zero_delta(snap):
    # со-узловые стопы (одинаковый node_id) остаются отдельными строками с Δ=0 между собой
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
    # число уникальных узлов < числа стопов ⇔ были коллизии (данные Аугсбурга их дают)
    assert len(first_seen) == len(snap.node_ids) - conode_pairs


def test_pharmacy_count(snap):
    lo, hi = cfg.PHARMACY_COUNT_RANGE
    assert lo <= snap.meta["n_pharmacies"] <= hi


def test_depot_resolved(snap):
    depot = snap.nodes[snap.nodes["kind"] == "depot"]
    assert len(depot) == 1, "депо-строка не единственная"
    assert int(depot.iloc[0]["node_id"]) in set(snap.node_ids), "депо-узел не в матрице"


def test_meta_coverage(snap):
    cov = snap.meta["maxspeed_coverage_pct"]
    assert cov is not None and 0 < cov <= 100, "покрытие maxspeed вне (0, 100]"


def test_opening_hours(snap):
    assert "opening_hours" in snap.nodes.columns, "нет колонки opening_hours"
    pct = snap.meta["opening_hours_present_pct"]
    assert 0 <= pct <= 100, "opening_hours_present_pct вне [0, 100]"
