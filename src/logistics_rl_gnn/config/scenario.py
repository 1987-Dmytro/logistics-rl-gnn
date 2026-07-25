"""A custom day scenario from YAML → Instance + event stream (Phase 9, acceptance).

A scenario describes a DISPATCHER'S DAY on the same real snapshot data: which weekday (pharmacy
windows come from the real `opening_hours` of that day + the congestion profile c(dow,h)), which
pharmacies (by name / stop-id / `all`), which fleet K/Q, and which events happen when.

Nothing is "re-modelled": the instance is built by `generate_instance` (the same real windows and
the same guard Σdemand ≤ K·Q + reachability within T_max), the events are the same `env.events`
classes the 0004 harness runs. A scenario only PARAMETRISES them instead of the seeded stream.

Schema (every field except `name` is optional):

    name: friday_south           # scenario name (headers/logs)
    weekday: friday              # 0..6 or monday..sunday; default — DELIVERY_WEEKDAY (Tuesday)
    dispatch_start: "08:00"      # dispatch window opening hour; default DISPATCH_OPEN_H
    pharmacies: all              # or a list of names/stop-ids: ["Vita-Apotheke", 12, ...]
    demand: {"Vita-Apotheke": 20}  # demand overrides (boxes) on top of the seeded draw
    fleet: {K: 2, Q: 80}         # the scenario fleet
    events:
      - {at: "09:15", type: traffic,   where: "Linden-Apotheke",
         params: {magnitude: 1.5, radius_km: 1.5, duration_min: 45}}   # closure: true → δ=∞
      - {at: "09:40", type: breakdown}
      - {at: "10:05", type: urgent, where: 12, params: {demand: 10, window_min: 45}}

Names resolve against the snapshot's `names.parquet`: exact match → unique substring → fuzzy
(difflib); otherwise a ValueError listing the near matches. Event times are "minutes from the
shift start" for the episode state and ABSOLUTE minutes from DISPATCH_OPEN_H for an incident (that
is how CongestionTravel reads them) — the shift is applied when dispatch_start ≠ DISPATCH_OPEN_H.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml

from logistics_rl_gnn.config import instance as im
from logistics_rl_gnn.config.congestion import (
    INCIDENT_DUR_RANGE_MIN,
    INCIDENT_MAG_RANGE,
    INCIDENT_RADIUS_KM,
)
from logistics_rl_gnn.env.events import BreakdownEvent, TrafficEvent, UrgentEvent
from logistics_rl_gnn.env.travel import Incident

_WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
_KINDS = ("traffic", "breakdown", "urgent")
_FUZZY_CUTOFF = 0.75  # below this we do not guess but ask (near matches listed in the error)


@dataclass
class Scenario:
    """A loaded scenario: the ready instance + events + what was asked for and what dropped out."""

    name: str
    weekday: int
    instance: object
    events: list
    path: Path
    requested_stops: list[int] = field(default_factory=list)  # snapshot stops from the YAML
    dropped_stops: list[int] = field(default_factory=list)  # closed that day → dropped
    names: dict = field(default_factory=dict)  # {snapshot_stop: name}

    @property
    def fleet(self) -> tuple[int, float]:
        return im.fleet_of(self.instance)

    def label(self, stop: int) -> str:
        return self.names.get(int(stop), f"stop #{int(stop)}")


# ---------- pharmacy names ----------


def load_names(snapshot_dir) -> dict[int, str]:
    """{snapshot_stop: name} from the snapshot names.parquet; no file → {} (id resolution only)."""
    p = Path(snapshot_dir) / "names.parquet"
    if not p.exists():
        return {}
    df = pd.read_parquet(p)
    return {
        int(s): str(n).strip()
        for s, n in zip(df["stop"], df["name"], strict=False)
        if pd.notna(n) and str(n).strip()
    }


def resolve_stop(token, names: dict[int, str], *, known: set[int] | None = None) -> int:
    """Name/stop-id from YAML → snapshot stop. Exact → substring → fuzzy; else a clear error."""
    if isinstance(token, bool):
        raise ValueError(f"pharmacy given as a boolean {token!r} — expected a name or a stop-id")
    if isinstance(token, int) or (isinstance(token, str) and token.strip().lstrip("#").isdigit()):
        stop = int(str(token).strip().lstrip("#"))
        if known is not None and stop not in known:
            raise ValueError(f"stop-id {stop} not in the snapshot (available 0..{max(known)})")
        return stop
    q = str(token).strip()
    if not names:
        raise ValueError(f"pharmacy '{q}': no names.parquet in snapshot — use a numeric stop-id")
    ql = q.casefold()
    exact = [s for s, n in names.items() if n.casefold() == ql]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        raise ValueError(f"name '{q}' is ambiguous: stops {sorted(exact)} — give a stop-id")
    sub = [s for s, n in names.items() if ql in n.casefold()]
    if len(sub) == 1:
        return sub[0]
    if len(sub) > 1:
        raise ValueError(
            f"name '{q}' is ambiguous — candidates: "
            + ", ".join(f"«{names[s]}» (stop {s})" for s in sorted(sub)[:5])
        )
    by_name: dict[str, list[int]] = {}  # same-named pharmacies → a LIST of stops, else fuzzy would
    for s, n in names.items():          # silently pick one (an exact match fails honestly)
        by_name.setdefault(n.casefold(), []).append(s)
    close = difflib.get_close_matches(ql, list(by_name), n=3, cutoff=_FUZZY_CUTOFF)
    if close:  # confident fuzzy match
        hit = by_name[close[0]]
        if len(hit) > 1:
            raise ValueError(f"name '{q}' ≈ '{names[hit[0]]}' is ambiguous: stops {sorted(hit)} "
                             f"— give a stop-id")
        return hit[0]
    hint = difflib.get_close_matches(ql, list(by_name), n=3, cutoff=0.3) or list(by_name)[:3]
    raise ValueError(
        f"pharmacy '{q}' not found in names.parquet. Similar: "
        + ", ".join(f"«{names[by_name[c][0]]}» (stop {by_name[c][0]})" for c in hint)
    )


# ---------- field parsing ----------


def _weekday(v) -> int:
    if v is None:
        return im.DELIVERY_WEEKDAY
    if isinstance(v, int) and not isinstance(v, bool):
        if not 0 <= v <= 6:
            raise ValueError(f"weekday {v} outside 0..6 (0 = Monday)")
        return v
    s = str(v).strip().casefold()
    for i, name in enumerate(_WEEKDAYS):
        if s in (name, name[:3]):
            return i
    raise ValueError(f"weekday '{v}' not recognised (0..6 or {', '.join(_WEEKDAYS)})")


def _hour(v, field_name: str) -> float:
    """'08:30' / 8 / 8.5 → hour of the day (float)."""
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        h = float(v)
    else:
        parts = str(v).strip().split(":")
        try:
            h = float(parts[0]) + (float(parts[1]) / 60.0 if len(parts) > 1 else 0.0)
        except ValueError as e:
            raise ValueError(f"{field_name} '{v}': expected the HH:MM format") from e
    if not 0.0 <= h < 24.0:
        raise ValueError(f"{field_name} '{v}' outside the day")
    return h


def _at_min(v, *, open_h: float, horizon_min: float) -> float:
    """Event time '09:15' → minutes from the shift start. Outside the shift → a clear error.

    round(…, 6) — otherwise 08:55 yields 54.999999… min and the demo clock prints '08:54' (the
    event looking a minute earlier than in the YAML).
    """
    at = round((_hour(v, "at") - open_h) * 60.0, 6)
    if not 0.0 <= at < horizon_min:
        end_h = open_h + horizon_min / 60.0
        raise ValueError(
            f"event at '{v}' outside the dispatch shift {open_h:02.0f}:00–{end_h:02.0f}:00 "
            f"({at:.0f} min from the start)"
        )
    return at


_EVENT_KEYS = {"at", "type", "where", "params"}
_PARAM_KEYS = {  # a typo in params would stay silent: `closure` misplaced → incident not closure
    "traffic": {"magnitude", "radius_km", "duration_min", "closure"},
    "urgent": {"demand", "window_min"},
    "breakdown": set(),
}


def _build_event(spec: dict, inst, *, open_h: float, names: dict, snap_stops: list):
    """One event description from YAML → an env.events object (the class the 0004 harness runs)."""
    kind = str(spec.get("type", "")).strip().casefold()
    if kind not in _KINDS:
        raise ValueError(f"event type '{spec.get('type')}' is unknown (expected {_KINDS})")
    bad = set(spec) - _EVENT_KEYS
    if bad:
        raise ValueError(f"event '{kind}' at '{spec.get('at')}': unknown fields {sorted(bad)} "
                         f"(parameters go into params: {sorted(_PARAM_KEYS[kind]) or '—'})")
    horizon_min = inst.horizon_s / 60.0
    at = _at_min(spec.get("at"), open_h=open_h, horizon_min=horizon_min)
    p = dict(spec.get("params") or {})
    bad_p = set(p) - _PARAM_KEYS[kind]
    if bad_p:
        raise ValueError(f"event '{kind}' at '{spec.get('at')}': unknown params "
                         f"{sorted(bad_p)} (expected {sorted(_PARAM_KEYS[kind]) or '—'})")
    if kind == "breakdown":
        return BreakdownEvent(at)

    if spec.get("where") is None:
        raise ValueError(f"event '{kind}' at '{spec.get('at')}' requires the field where")
    stop = resolve_stop(spec["where"], names, known=set(snap_stops))
    if stop not in snap_stops:
        raise ValueError(
            f"pharmacy '{spec['where']}' (stop {stop}) is not part of the scenario instance "
            "(not in the pharmacies list, or closed that day)"
        )
    idx = snap_stops.index(stop)  # instance-local index (as idx in the seeded stream)
    if kind == "urgent":
        return UrgentEvent(at, {
            "idx": idx,
            "demand": int(p.get("demand", im.DEMAND_RANGE[1])),
            "delta_s": float(p.get("window_min", 45.0)) * 60.0,
        })
    mag = float("inf") if bool(p.get("closure")) else float(
        p.get("magnitude", sum(INCIDENT_MAG_RANGE) / 2)
    )
    inc = Incident(
        center=tuple(inst.coords[idx]),
        radius_km=float(p.get("radius_km", INCIDENT_RADIUS_KM)),
        magnitude=mag,
        # the incident lives on the ABSOLUTE congestion clock (from DISPATCH_OPEN_H), the event
        # itself is counted from the shift start
        t_start_min=at + im.dispatch_offset_min(inst),
        duration_min=float(p.get("duration_min", sum(INCIDENT_DUR_RANGE_MIN) / 2)),
    )
    return TrafficEvent(at, inc)


# ---------- loading ----------


def load_scenario(path, *, snapshot_dir=None, seed: int = im.DEFAULT_SEED) -> Scenario:
    """YAML → Scenario. Validation by the existing guards (Σdemand ≤ K·Q, reachability)."""
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"no scenario file: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a YAML mapping, got {type(raw).__name__}")
    unknown = set(raw) - {"name", "weekday", "dispatch_start", "pharmacies", "demand", "fleet",
                          "events"}
    if unknown:
        raise ValueError(f"{path}: unknown fields {sorted(unknown)}")

    snap = Path(snapshot_dir) if snapshot_dir else im._latest_snapshot_dir()
    if snap is None:
        raise FileNotFoundError("no snapshot — run `python scripts/build_snapshot.py` first")
    names = load_names(snap)
    all_stops = set(pd.read_parquet(Path(snap) / "nodes.parquet")["stop"].astype(int))

    weekday = _weekday(raw.get("weekday"))
    start = raw.get("dispatch_start")
    open_h = _hour(start, "dispatch_start") if start is not None else None
    fleet = raw.get("fleet") or {}
    if not isinstance(fleet, dict):
        raise ValueError(f"{path}: fleet — a mapping {{K: …, Q: …}}, got {fleet!r}")
    if set(fleet) - {"K", "k", "Q", "q"}:  # a typo would silently restore the default 8×80 fleet
        raise ValueError(f"{path}: fleet — only K and Q, got {sorted(fleet)}")
    k = fleet.get("K", fleet.get("k"))
    q = fleet.get("Q", fleet.get("q"))

    ph = raw.get("pharmacies", "all")
    if isinstance(ph, str) and ph.strip().casefold() == "all":
        requested, include = [], None
    elif isinstance(ph, list):
        requested = [resolve_stop(t, names, known=all_stops) for t in ph]
        include = set(requested)
        if not include:
            raise ValueError(f"{path}: pharmacies — empty list")
    else:
        raise ValueError(f"{path}: pharmacies — 'all' or a list of names/stop-ids, got {ph!r}")

    overrides = {
        resolve_stop(t, names, known=all_stops): int(v)
        for t, v in (raw.get("demand") or {}).items()
    }
    bad = sorted(set(overrides) - include) if include is not None else []
    if bad:
        raise ValueError(f"{path}: demand override outside the pharmacies list: {bad}")

    try:
        inst = im.generate_instance(
            snap, seed=seed, delivery_weekday=weekday, include_stops=include,
            demand_overrides=overrides, fleet_size=k, vehicle_cap=q, dispatch_open_h=open_h,
        )
    except (ValueError, AssertionError) as e:  # the instance guard → point at the scenario file
        raise ValueError(f"{path}: scenario is invalid — {e}") from e

    open_eff = float(inst.meta["dispatch_open_h"])
    spec = raw.get("events") or []
    if not isinstance(spec, list) or any(not isinstance(s, dict) for s in spec):
        raise ValueError(f"{path}: events — a list of mappings [{{at, type, …}}], got {spec!r}")
    events = [_build_event(dict(s), inst, open_h=open_eff, names=names,
                           snap_stops=list(inst.snapshot_stops)) for s in spec]
    events.sort(key=lambda e: e.at_min)
    # dropped stops: from those requested (a subset) or all closed that day (pharmacies: all)
    known = set(inst.snapshot_stops)
    dropped = ([s for s in requested if s not in known] if requested
               else list(inst.excluded_stops))
    return Scenario(
        name=str(raw.get("name") or path.stem), weekday=weekday, instance=inst, events=events,
        path=path, requested_stops=requested, dropped_stops=dropped, names=names,
    )
