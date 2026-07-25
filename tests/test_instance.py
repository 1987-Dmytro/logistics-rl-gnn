"""Instance generation test: real windows from OSM opening_hours (Part D).

The helper unit tests always run; instance tests are skipped without a snapshot.
"""

from __future__ import annotations

import numpy as np
import pytest

from logistics_rl_gnn.config import instance as im
from logistics_rl_gnn.data import snapshot as snap_mod

WEEKDAY = im.DELIVERY_WEEKDAY  # Tuesday


# ---------- helper unit tests (no snapshot) ----------


def test_24_7_full_horizon():
    ds, de, es, hz = im._day_bounds(WEEKDAY)
    status, fo, lc = im.real_day_window("24/7", ds, de)
    assert status == "REAL"
    e_s = max(0.0, (fo - es).total_seconds())
    l_s = min(float(hz), (lc - es).total_seconds())
    assert e_s == 0.0 and l_s == float(hz)  # 24/7 → the whole horizon


def test_empty_tag_is_assumed():
    rng = np.random.default_rng(0)
    hz = im._day_bounds(WEEKDAY)[3]
    for bad in (None, "", "   ", "nan"):
        src, e_s, l_s = im.stop_window(bad, WEEKDAY, rng)
        assert src == "ASSUMED"
        assert 0 <= e_s < l_s <= hz


def test_garbage_is_assumed():
    rng = np.random.default_rng(0)
    src, _, _ = im.stop_window("garbage!!", WEEKDAY, rng)
    assert src == "ASSUMED"


def test_closed_day_excluded():
    # open on Monday only → closed on Tuesday → EXCLUDED
    rng = np.random.default_rng(0)
    src, e_s, l_s = im.stop_window("Mo 08:00-18:00", WEEKDAY, rng)
    assert src == "EXCLUDED" and e_s is None and l_s is None


def test_real_window_clipped():
    # open 06:00-22:00, dispatch window 08:00-18:00 → clipped to [0, horizon]
    rng = np.random.default_rng(0)
    src, e_s, l_s = im.stop_window("Mo-Su 06:00-22:00", WEEKDAY, rng)
    hz = im._day_bounds(WEEKDAY)[3]
    assert src == "REAL" and e_s == 0.0 and l_s == float(hz)


# ---------- instance on the real snapshot (skipped without one) ----------


@pytest.fixture(scope="module")
def inst():
    if im._latest_snapshot_dir() is None:
        pytest.skip("no snapshot: run `python scripts/build_snapshot.py` first")
    return im.generate_instance(seed=0, delivery_weekday=WEEKDAY)


def test_deterministic(inst):
    other = im.generate_instance(seed=0, delivery_weekday=WEEKDAY)
    assert inst.snapshot_stops == other.snapshot_stops
    assert inst.tw_source == other.tw_source
    assert np.array_equal(inst.windows, other.windows)
    assert np.array_equal(inst.demand, other.demand)


def test_depot_first(inst):
    assert inst.kinds[0] == "depot"
    assert inst.tw_source[0] == "DEPOT"
    assert np.array_equal(inst.windows[0], [0.0, float(inst.horizon_s)])


def test_real_windows_valid(inst):
    for i, src in enumerate(inst.tw_source):
        if src == "REAL":
            e_s, l_s = inst.windows[i]
            assert 0 <= e_s < l_s <= inst.horizon_s, f"stop {i}: REAL window invalid"


def test_counts_consistent(inst):
    n_pharm = sum(1 for k in inst.kinds if k == "pharmacy")
    assert inst.meta["n_real"] + inst.meta["n_fallback"] == n_pharm
    assert inst.meta["n_excluded"] == len(inst.excluded_stops)
    assert inst.meta["n_included_stops"] == len(inst.snapshot_stops)


def test_excluded_consistent(inst):
    snap = snap_mod.load_snapshot(im._latest_snapshot_dir(), with_graph=False)
    ds, de, _, _ = im._day_bounds(WEEKDAY)
    nodes = snap.nodes.set_index("stop")
    excluded = set(inst.excluded_stops)
    for stop_i in excluded:
        status, *_ = im.real_day_window(nodes.loc[stop_i, "opening_hours"], ds, de)
        assert status == "CLOSED", f"stop {stop_i} excluded but not CLOSED ({status})"
    for stop_i in inst.snapshot_stops:  # included pharmacies must not be CLOSED
        if stop_i == 0:
            continue
        status, *_ = im.real_day_window(nodes.loc[stop_i, "opening_hours"], ds, de)
        assert status != "CLOSED", f"stop {stop_i} included but CLOSED"


# ---------- feasibility guard (calibration) ----------

_HAS_SNAP = im._latest_snapshot_dir() is not None


@pytest.mark.skipif(not _HAS_SNAP, reason="no snapshot")
def test_feasibility_invariant():
    cap = im.FLEET_SIZE * im.VEHICLE_CAP
    for seed in range(5):  # no raise + demand ≤ K*Q
        inst = im.generate_instance(seed=seed, delivery_weekday=WEEKDAY)
        assert float(inst.demand.sum()) <= cap


@pytest.mark.skipif(not _HAS_SNAP, reason="no snapshot")
def test_reachability_invariant():
    t_max_s = im.T_MAX_MIN * 60.0
    for seed in range(5):
        inst = im.generate_instance(seed=seed, delivery_weekday=WEEKDAY)
        tm, svc = inst.time_matrix, inst.service
        for i in range(1, len(inst.node_ids)):  # 0 = depot
            assert tm[0, i] + svc[i] + tm[i, 0] <= t_max_s, f"pharmacy {i} unreachable in T_max"


@pytest.mark.skipif(not _HAS_SNAP, reason="no snapshot")
def test_feasibility_guard_raises(monkeypatch):
    # artificially shrink Q → demand > K*Q → the guard must raise
    monkeypatch.setattr(im, "VEHICLE_CAP", 1)
    with pytest.raises(ValueError, match="infeasible"):
        im.generate_instance(seed=0, delivery_weekday=WEEKDAY)
