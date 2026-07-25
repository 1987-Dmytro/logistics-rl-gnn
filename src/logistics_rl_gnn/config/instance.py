"""CVRPTW instance generator from a snapshot: real pharmacy windows from OSM opening_hours.

Windows: REAL from the tag (clipped to the depot dispatch window), ASSUMED synthetics when the
tag is missing or unparsable, EXCLUDED — the pharmacy is closed on the delivery day (stop dropped).
Everything is deterministic: seed + delivery_weekday + snapshot. The date is anchored to a fixed
value (NOT now()) → reproducibility (dec-0001 prohibition #4).

opening_hours_py is installed as the package `opening_hours_py` but imported as the module
`opening_hours` (verified against the installed version).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
from opening_hours import OpeningHours

# --- instance config (all [ASSUMED, config], dec-0001 §2) ---
DELIVERY_WEEKDAY = 1  # Tuesday (Mon=0)
DISPATCH_OPEN_H = 8  # depot: dispatch window (hours of the day)
DISPATCH_CLOSE_H = 18
REFERENCE_DATE = date(2024, 1, 1)  # anchor → a reproducible weekday date without public holidays
DEMAND_RANGE = (3, 12)  # boxes per stop
SERVICE_MIN = 4.0  # service minutes per stop
DEFAULT_SEED = 0
REACH_MARGIN_S = 60.0  # reachability margin when clipping a synthetic window (OR-Tools rounding)

# --- fleet / cost / penalties (dec-0001 §2-3, [ASSUMED, config]; calibrated in Phase 4) ---
FLEET_SIZE = 8  # K vehicles (calibration: 6→8 so that K*Q covers demand)
VEHICLE_CAP = 80  # Q boxes (calibration: 60→80)
T_MAX_MIN = 240.0  # max tour, minutes (4h) — rationale in dec-0001 §6
COST_PER_VEHICLE = 50.0  # c_f, € per vehicle used
COST_PER_KM = 0.5  # c_d, €/km
COST_PER_HOUR = 20.0  # c_t, €/h
PENALTY_LATE_PER_MIN = 2.0  # lateness penalty, €/min
PENALTY_UNSERVED = 200.0  # € per unserved pharmacy
PENALTY_INVALID = 1000.0  # € per invalid action (terminal)
DENSE_REWARD = False  # per-step dense shaping (off by default)

SNAP_ROOT = Path("data/snapshots")


@dataclass
class Instance:
    """CVRPTW instance. Every vector has length k (included stops, depot = index 0)."""

    node_ids: list[int]  # OSM node id per stop
    snapshot_stops: list[int]  # snapshot stop indices that entered the instance
    kinds: list[str]  # depot | pharmacy
    time_matrix: np.ndarray  # (k,k) sub-matrix of the included stops, seconds
    dist_matrix: np.ndarray  # (k,k), metres
    coords: np.ndarray  # (k,2) [x=lon, y=lat] per stop
    windows: np.ndarray  # (k,2) [e_i,l_i] s from start_datetime; depot=[0,horizon]
    demand: np.ndarray  # (k,) boxes; depot=0
    service: np.ndarray  # (k,) seconds; depot=0
    tw_source: list[str]  # DEPOT | REAL | ASSUMED per stop
    excluded_stops: list[int]  # snapshot stop indices closed on the delivery day
    start_datetime: datetime
    horizon_s: int
    meta: dict = field(default_factory=dict)


def _day_bounds(delivery_weekday: int, dispatch_open_h=None) -> tuple:
    """(day_start, day_end, episode_start, horizon_s) for the delivery day.

    dispatch_open_h — opening hour of the dispatch window (scenario); None → DISPATCH_OPEN_H.
    """
    open_h = DISPATCH_OPEN_H if dispatch_open_h is None else float(dispatch_open_h)
    monday = REFERENCE_DATE - timedelta(days=REFERENCE_DATE.weekday())
    day = monday + timedelta(days=delivery_weekday)
    day_start = datetime(day.year, day.month, day.day)
    day_end = day_start + timedelta(days=1)
    episode_start = day_start + timedelta(hours=open_h)
    horizon_s = int((DISPATCH_CLOSE_H - open_h) * 3600)
    return day_start, day_end, episode_start, horizon_s


def fleet_of(instance) -> tuple[int, float]:
    """(K, Q) of the instance: scenario override from meta or the global defaults. One point —
    otherwise a scenario fleet quietly drifts from the 8×80 default in env/polish/OR-Tools."""
    m = getattr(instance, "meta", None) or {}
    return int(m.get("fleet_size", FLEET_SIZE)), float(m.get("vehicle_cap", VEHICLE_CAP))


def dispatch_offset_min(instance) -> float:
    """Congestion hour shift, minutes: the scenario dispatch window may not open at DISPATCH_OPEN_H.

    CongestionTravel maps abs_min → hour as DISPATCH_OPEN_H + abs_min/60 → a 09:00 start needs
    offset_min=60, otherwise the c(dow,h) profile is shifted by an hour (a silent error)."""
    m = getattr(instance, "meta", None) or {}
    return (float(m.get("dispatch_open_h", DISPATCH_OPEN_H)) - DISPATCH_OPEN_H) * 60.0


def real_day_window(oh_string, day_start: datetime, day_end: datetime):
    """(status, first_open, last_close) for the day [day_start, day_end).

    status: REAL — there are open intervals; CLOSED — valid but closed (exclude);
            INVALID — empty/unparsable/unknown only → synthetic fallback.
    """
    if oh_string is None:
        return "INVALID", None, None
    s = str(oh_string).strip()
    if not s or s.lower() == "nan":
        return "INVALID", None, None
    try:
        ivs = list(OpeningHours(s).intervals(day_start, day_end))
    except Exception:
        return "INVALID", None, None
    states = [str(iv[2]).lower() for iv in ivs]
    if any(st == "open" for st in states):
        opens = [(iv[0], iv[1]) for iv in ivs if str(iv[2]).lower() == "open"]
        return "REAL", min(a for a, _ in opens), max(b for _, b in opens)
    if any(st == "unknown" for st in states):
        return "INVALID", None, None  # unknown → fallback, do NOT exclude
    return "CLOSED", None, None


def _synthetic_window(
    rng: np.random.Generator, horizon_s: int, e_max: float | None = None
) -> tuple[float, float]:
    """Seeded synthetic window [e,l] in seconds, e<l, within the horizon (ASSUMED).

    e_max — upper bound of the early window by reachability within T_max (clip e down so that an
    ASSUMED placeholder never makes a real pharmacy structurally unreachable); None → no clip.
    Clipping after the draw → the rng stream is unchanged (feasible windows stay as they were).
    """
    e = float(rng.uniform(0.0, 0.4 * horizon_s))
    width = float(rng.uniform(0.3 * horizon_s, 0.6 * horizon_s))
    if e_max is not None:
        e = min(e, max(0.0, e_max))  # not later than "wait/serve/return" still fits
    return e, min(float(horizon_s), e + width)


def stop_window(oh_string, delivery_weekday: int, rng: np.random.Generator, e_max=None,
                dispatch_open_h=None):
    """(source, e_s, l_s) for one stop. source: REAL | ASSUMED | EXCLUDED.

    e_s,l_s — seconds from episode_start (depot open); EXCLUDED → (None, None). rng is spent
    only on the ASSUMED branch. e_max clips the SYNTHETIC only (real windows are untouched —
    prohibition #5); the snapshot's real windows are reachable (verified by the guard).
    """
    day_start, day_end, episode_start, horizon_s = _day_bounds(delivery_weekday, dispatch_open_h)
    status, fo, lc = real_day_window(oh_string, day_start, day_end)
    if status == "CLOSED":
        return "EXCLUDED", None, None
    if status == "REAL":
        e_s = max(0.0, (fo - episode_start).total_seconds())
        l_s = min(float(horizon_s), (lc - episode_start).total_seconds())
        if l_s > e_s:  # open within the dispatch window
            return "REAL", e_s, l_s
        # open, but outside the dispatch window → synthetic
    e_s, l_s = _synthetic_window(rng, horizon_s, e_max)
    return "ASSUMED", e_s, l_s


def _latest_snapshot_dir():
    if not SNAP_ROOT.is_dir():
        return None
    dirs = sorted(p for p in SNAP_ROOT.glob("augsburg_*") if (p / "meta.json").is_file())
    return dirs[-1] if dirs else None


def _check_feasibility(demand, service, time_m, win_e, fleet_size=None, vehicle_cap=None) -> None:
    """Instance guard (a Phase 3 bug caught once → a permanent check).

    raise when total demand > K*Q (infeasible); warn when utilisation > 0.9;
    assert reachability of every pharmacy INCLUDING THE WAIT until e_i: a vehicle starts at t=0,
    waits until e_i, serves, returns — max(t[0,i], e_i)+service+t[i,0] <= T_max.
    demand/service/win_e — lists (depot = index 0); time_m — seconds. fleet_size/vehicle_cap
    (scenario) — None → the GLOBAL constants, read in the body (the monkeypatch test still works).
    """
    k = FLEET_SIZE if fleet_size is None else int(fleet_size)
    q = VEHICLE_CAP if vehicle_cap is None else float(vehicle_cap)
    cap = k * q
    total = float(sum(demand))
    if total > cap:
        raise ValueError(
            f"infeasible: demand {total:.0f} > K*Q={k}×{q:.0f}={cap:.0f} (too little capacity)"
        )
    heavy = [i for i in range(1, len(demand)) if float(demand[i]) > q]
    if heavy:  # a demand override above vehicle capacity → the stop can NEVER be delivered
        raise ValueError(
            f"infeasible: demand of stop #{heavy[0]} = {float(demand[heavy[0]]):.0f} > Q={q:.0f} "
            f"(a single vehicle cannot carry that stop)"
        )
    util = total / cap
    if util > 0.9:
        warnings.warn(
            f"high utilisation {util:.0%} (demand {total:.0f}/{cap}) — unserved risk",
            stacklevel=2,
        )
    t_max_s = T_MAX_MIN * 60.0
    for i in range(1, len(demand)):  # 0 = depot
        reach = max(float(time_m[0, i]), float(win_e[i])) + float(service[i]) + float(time_m[i, 0])
        assert reach <= t_max_s + 1e-6, (
            f"pharmacy #{i} unreachable within T_max including the wait until e: "
            f"{reach / 60:.1f} > {T_MAX_MIN} min (e={win_e[i] / 60:.1f})"
        )


def generate_instance(
    snapshot_dir=None,
    *,
    seed: int = DEFAULT_SEED,
    delivery_weekday: int = DELIVERY_WEEKDAY,
    include_stops=None,
    demand_overrides=None,
    fleet_size=None,
    vehicle_cap=None,
    dispatch_open_h=None,
) -> Instance:
    """CVRPTW instance from a snapshot with real pharmacy windows for the delivery day.

    Scenario overrides (config.scenario; ALL None → the default path byte for byte):
    include_stops — a subset of snapshot pharmacy stops (the depot is always in); demand_overrides
    — {snapshot_stop: boxes} on top of the draw (the rng stream is NOT changed); fleet_size/
    vehicle_cap — scenario K/Q (into meta → `fleet_of`, and from there env/polish/OR-Tools);
    dispatch_open_h — the dispatch window opening hour.
    """
    from logistics_rl_gnn.data import snapshot as snap_mod

    d = snapshot_dir or _latest_snapshot_dir()
    if d is None:
        raise FileNotFoundError("no snapshot — run `python scripts/build_snapshot.py` first")
    snap = snap_mod.load_snapshot(d, with_graph=False)
    _, _, episode_start, horizon_s = _day_bounds(delivery_weekday, dispatch_open_h)
    want = None if include_stops is None else {int(s) for s in include_stops}
    over = {int(k): int(v) for k, v in (demand_overrides or {}).items()}
    rng = np.random.default_rng(seed)
    depot_stop = int(snap.nodes.loc[snap.nodes.kind == "depot", "stop"].iloc[0])
    t_max_s = T_MAX_MIN * 60.0

    included: list[int] = []
    excluded: list[int] = []
    node_ids: list[int] = []
    coords: list[list[float]] = []
    kinds: list[str] = []
    tw_source: list[str] = []
    win_e: list[float] = []
    win_l: list[float] = []
    demand: list[int] = []
    service: list[float] = []

    for row in snap.nodes.itertuples(index=False):
        stop_i = int(row.stop)
        if row.kind == "depot":
            included.append(stop_i)
            node_ids.append(int(row.node_id))
            coords.append([float(row.x), float(row.y)])
            kinds.append("depot")
            tw_source.append("DEPOT")
            win_e.append(0.0)
            win_l.append(float(horizon_s))
            demand.append(0)
            service.append(0.0)
            continue
        if want is not None and stop_i not in want:
            continue  # scenario subset: the stop was not requested (no rng spent)
        # early-window budget: wait until e, serve and return to the depot within T_max
        ret_s = float(snap.time_matrix[stop_i, depot_stop])
        e_max = t_max_s - SERVICE_MIN * 60.0 - ret_s - REACH_MARGIN_S
        src, e_s, l_s = stop_window(row.opening_hours, delivery_weekday, rng, e_max=e_max,
                                    dispatch_open_h=dispatch_open_h)
        if src == "EXCLUDED":
            excluded.append(stop_i)
            continue
        included.append(stop_i)
        node_ids.append(int(row.node_id))
        coords.append([float(row.x), float(row.y)])
        kinds.append("pharmacy")
        tw_source.append(src)
        win_e.append(e_s)
        win_l.append(l_s)
        drawn = int(rng.integers(DEMAND_RANGE[0], DEMAND_RANGE[1] + 1))  # ALWAYS drawn (rng)
        demand.append(over.get(stop_i, drawn))
        service.append(SERVICE_MIN * 60.0)

    idx = np.array(included)
    time_m = snap.time_matrix[np.ix_(idx, idx)]
    # guard (the same as for the default): serve ≤ K*Q and reachable within T_max
    _check_feasibility(demand, service, time_m, win_e, fleet_size, vehicle_cap)
    return Instance(
        node_ids=node_ids,
        snapshot_stops=included,
        kinds=kinds,
        time_matrix=time_m,
        dist_matrix=snap.dist_matrix[np.ix_(idx, idx)],
        coords=np.array(coords, dtype=float),
        windows=np.column_stack([win_e, win_l]),
        demand=np.array(demand),
        service=np.array(service),
        tw_source=tw_source,
        excluded_stops=excluded,
        start_datetime=episode_start,
        horizon_s=horizon_s,
        meta={
            "seed": seed,
            "delivery_weekday": delivery_weekday,
            "n_real": tw_source.count("REAL"),
            "n_fallback": tw_source.count("ASSUMED"),
            "n_excluded": len(excluded),
            "n_included_stops": len(included),
            # effective K/Q/opening hour (== defaults when the scenario is silent) → fleet_of
            "fleet_size": FLEET_SIZE if fleet_size is None else int(fleet_size),
            "vehicle_cap": VEHICLE_CAP if vehicle_cap is None else float(vehicle_cap),
            "dispatch_open_h": DISPATCH_OPEN_H if dispatch_open_h is None else float(
                dispatch_open_h
            ),
        },
    )


if __name__ == "__main__":
    inst = generate_instance()
    m = inst.meta
    print(
        f"instance: stops included={m['n_included_stops']} (depot+pharmacies); "
        f"windows REAL={m['n_real']} ASSUMED={m['n_fallback']} excluded={m['n_excluded']}; "
        f"weekday={m['delivery_weekday']} seed={m['seed']} horizon={inst.horizon_s}s"
    )
