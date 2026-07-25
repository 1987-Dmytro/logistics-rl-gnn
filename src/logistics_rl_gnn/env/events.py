"""Events of a dynamic episode + residual re-plan (Phase 7, dec-0001 §4).

Three events (re-plan triggers): traffic (an incident over a zone of edges), breakdown (remove a
vehicle → its stops return to the unassigned pool), urgent_order (an existing pharmacy with a
narrow window). The smooth diurnal c(h) is NOT a trigger (it already lives in CongestionTravel).
Events are seeded → reproducible.

The env is NOT rewritten: DynamicVRPEnv is a thin subclass overriding _load so CongestionTravel
survives the internal reset() (rollout/greedy reset the environment). residual = re-optimisation
of the REMAINING stops from the depot at the moment of the event (fresh T_max/vehicle; NOT a
mid-route continuation — standard periodic re-optimisation of dynamic VRP). Windows shift into
the event's time base.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from logistics_rl_gnn.config import congestion as cg
from logistics_rl_gnn.config import instance as im
from logistics_rl_gnn.env.travel import CongestionTravel, Incident
from logistics_rl_gnn.env.vrp_env import VRPEnv

# ---------- drop-in environment with congestion travel ----------


class DynamicVRPEnv(VRPEnv):
    """VRPEnv injecting a travel model via a factory (survives reset). Base env untouched."""

    def __init__(self, *, travel_factory=None, **kw):
        self._travel_factory = travel_factory  # (env) -> TravelModel; None → base free-flow
        super().__init__(**kw)

    def _load(self, inst) -> None:
        super()._load(inst)
        if getattr(self, "_travel_factory", None) is not None:
            self.travel = self._travel_factory(self)


def make_dynamic_env(instance, *, travel=None, **kw) -> DynamicVRPEnv:
    """Env on a fixed residual instance with a ready travel model (or free-flow when travel=None).

    K/Q/weekday come FROM THE INSTANCE (meta; the default instance carries the global values) —
    otherwise a scenario fleet would be silently lost in VRPEnv def-time defaults. An explicit
    kwarg still wins.
    """
    k, q = im.fleet_of(instance)
    kw.setdefault("fleet_size", k)
    kw.setdefault("vehicle_cap", q)
    kw.setdefault("delivery_weekday", int((getattr(instance, "meta", None) or {}).get(
        "delivery_weekday", im.DELIVERY_WEEKDAY)))
    fac = (lambda env: travel) if travel is not None else None
    return DynamicVRPEnv(instance_fn=lambda s: instance, travel_factory=fac, **kw)


def congestion_for(instance, *, dow: int, offset_min: float = 0.0, incidents=None, flat_c=False):
    """CongestionTravel over the instance's free-flow t0 (minutes) and its coordinates."""
    t0_min = np.asarray(instance.time_matrix, dtype=float) / 60.0
    return CongestionTravel(
        t0_min,
        instance.coords,
        dow=dow,
        offset_min=offset_min,
        incidents=incidents,
        flat_c=flat_c,
    )


# ---------- state of a dynamic episode ----------


@dataclass
class DynamicState:
    instance: object  # Instance (full, before any slicing)
    dow: int
    served: set = field(default_factory=set)  # instance-local indices of stops already served
    broken: int = 0  # number of vehicles lost
    urgent: list = field(default_factory=list)  # [{idx, demand, delta_s}]
    incidents: list = field(default_factory=list)  # accumulated Incident objects
    now_min: float = 0.0

    def fleet(self, base_k: int) -> int:
        return max(1, base_k - self.broken)


def served_by(routes, instance, travel, now_min: float) -> set[int]:
    """Stops COMPLETED (end of service) before now_min while walking routes with travel. The
    executed plan timeline → the "partial state" follows the real dispatch plan, not chance."""
    win = np.asarray(instance.windows, dtype=float) / 60.0
    svc = np.asarray(instance.service, dtype=float) / 60.0
    done: set[int] = set()
    for route in routes:
        t = 0.0
        for a, b in zip(route[:-1], route[1:], strict=False):
            arrival = t + float(travel.time(a, b, t))
            if b == 0:
                t = arrival
                continue
            finish = max(arrival, win[b, 0]) + svc[b]
            if finish <= now_min:
                done.add(int(b))
            t = finish
    return done


def residual_instance(state: DynamicState):
    """Instance of the remaining task: depot + unserved + urgent (windows shifted to event base).

    Tagged meta.dynamic=True, tag='simulated-on-real' — a perturbation of a real instance (#5).
    """
    inst = state.instance
    n = len(inst.demand)
    now_s = state.now_min * 60.0
    pending = [i for i in range(1, n) if i not in state.served]
    for u in state.urgent:  # urgent stops are included even if served (re-delivery)
        if u["idx"] not in pending:
            pending.append(u["idx"])
    idx = [0] + sorted(set(pending))
    ridx = np.array(idx, dtype=int)

    win = inst.windows[ridx].astype(float).copy()  # seconds from episode_start
    dem = inst.demand[ridx].copy()
    svc = inst.service[ridx].astype(float).copy()
    # window shift: residual t=0 = the event moment → [e-now, l-now] s, e≥0; depot = [0, horizon]
    win[1:, 0] = np.maximum(0.0, win[1:, 0] - now_s)
    win[1:, 1] = win[1:, 1] - now_s
    win[0] = [0.0, float(inst.horizon_s)]
    pos = {orig: k for k, orig in enumerate(idx)}
    for u in state.urgent:  # urgent: narrow window [0, delta] (event-relative) + fresh demand
        k = pos[u["idx"]]
        win[k] = [0.0, float(u["delta_s"])]
        dem[k] = int(u["demand"])

    return replace(
        inst,
        node_ids=[inst.node_ids[i] for i in idx],
        snapshot_stops=[inst.snapshot_stops[i] for i in idx],
        kinds=[inst.kinds[i] for i in idx],
        tw_source=[inst.tw_source[i] for i in idx],
        time_matrix=inst.time_matrix[np.ix_(ridx, ridx)],
        dist_matrix=inst.dist_matrix[np.ix_(ridx, ridx)],
        coords=inst.coords[ridx],
        windows=win,
        demand=dem,
        service=svc,
        meta={
            **inst.meta,
            "dynamic": True,
            "tag": cg.CALIBRATION_TAG,
            "now_min": state.now_min,
            "n_pending": len(pending),
            "broken": state.broken,
            # fleet of the REMAINDER: lost vehicles are unavailable. Otherwise an env built from
            # a residual without an explicit fleet_size would silently plan on the original K
            # (today every call passes it explicitly).
            "fleet_size": state.fleet(int(inst.meta.get("fleet_size", im.FLEET_SIZE))),
        },
    )


# ---------- events ----------


@dataclass
class TrafficEvent:
    at_min: float
    incident: Incident
    kind: str = "traffic"

    def apply(self, state: DynamicState) -> bool:
        state.incidents.append(self.incident)
        return True  # an incident is a re-plan trigger


@dataclass
class BreakdownEvent:
    at_min: float
    kind: str = "breakdown"

    def apply(self, state: DynamicState) -> bool:
        state.broken += 1  # vehicle lost; its stops are already unserved (the residual takes them)
        return True


@dataclass
class UrgentEvent:
    at_min: float
    order: dict
    kind: str = "urgent"

    def apply(self, state: DynamicState) -> bool:
        state.urgent.append(self.order)
        return True


@dataclass
class DiurnalTick:
    at_min: float
    kind: str = "diurnal"

    def apply(self, state: DynamicState) -> bool:
        return False  # the smooth c(h) already lives in travel — NOT a trigger (dec-0001 §4)


def event_stream(seed: int, instance, dow: int, n_events: int = 6) -> list:
    """Seeded stream of events for an episode. Ordered by time; triggers + diurnal ticks mixed."""
    rng = np.random.default_rng(seed + 7_000_003)  # decorrelate from the instance seed
    n = len(instance.demand)
    horizon_min = instance.horizon_s / 60.0
    # events in the ACTIVE execution phase: 8 vehicles run in parallel from t=0, plan ~2h → the
    # window [~20, ~130] min yields meaningful partial states (later almost everything is served).
    times = np.sort(rng.uniform(0.03 * horizon_min, 0.22 * horizon_min, size=n_events))
    kinds = ["traffic", "urgent", "breakdown", "diurnal", "traffic", "urgent"]
    events: list = []
    for i, at in enumerate(times):
        kind = kinds[i % len(kinds)]
        if kind == "traffic":
            center = tuple(instance.coords[int(rng.integers(1, n))])
            closure = bool(rng.random() < cg.INCIDENT_CLOSURE_PROB)
            mag = np.inf if closure else float(rng.uniform(*cg.INCIDENT_MAG_RANGE))
            dur = float(rng.uniform(*cg.INCIDENT_DUR_RANGE_MIN))
            inc = Incident(center, cg.INCIDENT_RADIUS_KM, mag, float(at), dur)
            events.append(TrafficEvent(float(at), inc))
        elif kind == "urgent":
            order = {
                "idx": int(rng.integers(1, n)),
                "demand": int(rng.integers(im.DEMAND_RANGE[0], im.DEMAND_RANGE[1] + 1)),
                "delta_s": float(rng.uniform(30.0, 75.0) * 60.0),  # narrow window 30–75 min
            }
            events.append(UrgentEvent(float(at), order))
        elif kind == "breakdown":
            events.append(BreakdownEvent(float(at)))
        else:
            events.append(DiurnalTick(float(at)))
    return events
