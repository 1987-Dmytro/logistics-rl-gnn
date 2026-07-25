"""Travel-time model. The TravelModel interface + FreeFlowTravel (Phase 3) +
CongestionTravel (Phase 7).

CongestionTravel — the same interface, but at_minute affects the time: a diurnal multiplier
c(dow,h) × the incident contribution. Drop-in: swapped through env.travel (survives reset in
the dynamic env). The unit is minutes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from logistics_rl_gnn.config import congestion as cg
from logistics_rl_gnn.config.instance import DISPATCH_OPEN_H

_CLOSED_LEVEL = 5.0  # node level for "a closure nearby" (finite sentinel instead of edge inf)


class TravelModel:
    """Interface: travel time i→j (min) when departing at at_minute (min from the start)."""

    offset_min: float = 0.0  # tour start shift into absolute time of day (0 for static/free-flow)

    def time(self, i: int, j: int, at_minute: float) -> float:
        raise NotImplementedError

    def node_congestion(self, coords, at_minute: float = 0.0) -> np.ndarray:
        """Congestion at every node (0 = free-flow / no active incident). Base → zeros."""
        return np.zeros(len(coords), dtype=np.float32)


class FreeFlowTravel(TravelModel):
    """Free-flow: time does not depend on at_minute (Phase 3) → time_m[i, j]."""

    def __init__(self, time_m_min: np.ndarray):
        self.time_m = np.asarray(time_m_min, dtype=float)

    def time(self, i: int, j: int, at_minute: float = 0.0) -> float:
        return float(self.time_m[i, j])

    def matrix(self, at_minute: float = 0.0) -> np.ndarray:
        """The whole time matrix (min) — free-flow is time-independent → t0."""
        return self.time_m


def _km(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Approximate lon/lat distance → km (local flat approximation, Augsburg ~48.4°).

    ponytail: flat-earth is fine for a zone of radius 1–2 km; a city needs no more precision.
    """
    lat = math.radians((a[1] + b[1]) / 2.0)
    dlon = (a[0] - b[0]) * 111.320 * math.cos(lat)
    dlat = (a[1] - b[1]) * 110.574
    return math.hypot(dlon, dlat)


def _km_to_center(coords: np.ndarray, center: tuple[float, float]) -> np.ndarray:
    """Vectorised distance (km) from every node to center — EXACTLY as _km (per-node midpoint lat).

    Matching _km is mandatory: else a node on the zone boundary flips in/out (silent bug). -> [k].
    """
    coords = np.asarray(coords, dtype=float)
    clon, clat = float(center[0]), float(center[1])
    midlat = np.radians((clat + coords[:, 1]) / 2.0)  # per-node midpoint, as in _km
    dlon = (clon - coords[:, 0]) * 111.320 * np.cos(midlat)
    dlat = (clat - coords[:, 1]) * 110.574
    return np.hypot(dlon, dlat)


@dataclass
class Incident:
    """A local incident: zone (centre coordinate + radius) × time window × magnitude.

    Geometry is coordinate-based (not node-index) → it survives the instance slice on re-plan.
    magnitude=inf → closure (the edge is removed). Decay is linear towards the end of duration.
    Time is in ABSOLUTE minutes from dispatch-open (as at_minute+offset in CongestionTravel).
    """

    center: tuple[float, float]  # (lon, lat)
    radius_km: float
    magnitude: float  # δ; inf = edge closure
    t_start_min: float
    duration_min: float

    def contrib(self, coord_i, coord_j, abs_min: float) -> float:
        """Contribution to (1+Σ) on edge i→j at abs_min: δ·decay, inf on closure, else 0."""
        if not (self.t_start_min <= abs_min <= self.t_start_min + self.duration_min):
            return 0.0
        in_zone = (
            _km(self.center, tuple(coord_i)) <= self.radius_km
            or _km(self.center, tuple(coord_j)) <= self.radius_km
        )
        if not in_zone:
            return 0.0
        if math.isinf(self.magnitude):
            return math.inf
        decay = 1.0 - (abs_min - self.t_start_min) / self.duration_min  # 1→0 linearly
        return self.magnitude * decay

    def at_node(self, coord, abs_min: float) -> float:
        """Contribution to a NODE (centre in the zone): δ·decay if active and the node is inside
        the radius; _CLOSED_LEVEL on closure; else 0. Finite (unlike contrib's inf) — a node
        feature is never infinite."""
        if not (self.t_start_min <= abs_min <= self.t_start_min + self.duration_min):
            return 0.0
        if _km(self.center, tuple(coord)) > self.radius_km:
            return 0.0
        if math.isinf(self.magnitude):
            return _CLOSED_LEVEL
        return self.magnitude * (1.0 - (abs_min - self.t_start_min) / self.duration_min)


def time_context(abs_min: float, dow: int) -> np.ndarray:
    """Cyclic time context: [sin_h, cos_h, sin_dow, cos_dow], h = DISPATCH_OPEN_H+abs/60.

    Phase of the day (for diurnal congestion) + weekday. Non-zero under free-flow too — but that
    is a CONSTANT input with no congestion signal at c≡1, harmless to convergence (Phase 6b Step 0).
    """
    hour = DISPATCH_OPEN_H + abs_min / 60.0
    ah = 2.0 * math.pi * hour / 24.0
    ad = 2.0 * math.pi * (int(dow) % 7) / 7.0
    return np.array([math.sin(ah), math.cos(ah), math.sin(ad), math.cos(ad)], dtype=np.float32)


class CongestionTravel(TravelModel):
    """t(i,j,at) = t0[i,j]·c(dow, h) · (1 + Σ_k I_k), h = DISPATCH_OPEN_H + (offset+at)/60.

    offset_min — shift of the tour start time (for a residual re-plan: a midday event → tours
    begin at the real hour). incidents — a list of Incident (geometry from coords).
    Parity: c≡1 (flat_c=True passed) AND no incidents ⇒ exactly FreeFlow (t0[i,j]).
    """

    def __init__(
        self,
        t0_min: np.ndarray,
        coords: np.ndarray,
        *,
        dow: int,
        offset_min: float = 0.0,
        incidents: list[Incident] | None = None,
        flat_c: bool = False,
    ):
        self.time_m = np.asarray(t0_min, dtype=float)
        self.coords = np.asarray(coords, dtype=float)
        self.dow = int(dow)
        self.offset_min = float(offset_min)
        self.incidents = list(incidents or [])
        self.flat_c = bool(flat_c)  # test parity: disable the diurnal (c≡1)

    def time(self, i: int, j: int, at_minute: float = 0.0) -> float:
        t0 = float(self.time_m[i, j])
        abs_min = self.offset_min + at_minute
        c = 1.0 if self.flat_c else cg.congestion(self.dow, DISPATCH_OPEN_H + abs_min / 60.0)
        inc = 0.0
        for k in self.incidents:
            d = k.contrib(self.coords[i], self.coords[j], abs_min)
            if math.isinf(d):
                return math.inf  # edge closure → unreachable (masked in the env)
            inc += d
        return t0 * c * (1.0 + inc)

    def matrix(self, at_minute: float = 0.0) -> np.ndarray:
        """The whole matrix t(i,j,at) in one shot (vectorised) — EXACT parity with time().

        Incident zones via an outer-OR of boolean node masks; closure → inf on the edge (union over
        all closure incidents, on top of any finite contributions — as time()'s early-return inf).
        build_graph calls this (k² Python time() calls choke the POMO retrain at k=62).
        """
        abs_min = self.offset_min + at_minute
        c = 1.0 if self.flat_c else cg.congestion(self.dow, DISPATCH_OPEN_H + abs_min / 60.0)
        tm = self.time_m * c
        if not self.incidents:
            return tm
        k = tm.shape[0]
        inc = np.zeros((k, k), dtype=float)
        closed = np.zeros((k, k), dtype=bool)
        for ic in self.incidents:
            if not (ic.t_start_min <= abs_min <= ic.t_start_min + ic.duration_min):
                continue  # outside the incident window — contribution 0 (as in contrib())
            zone = _km_to_center(self.coords, ic.center) <= ic.radius_km  # [k] nodes in zone
            edge_zone = zone[:, None] | zone[None, :]  # the edge is hit if i OR j is in the zone
            if math.isinf(ic.magnitude):
                closed |= edge_zone
            else:
                decay = 1.0 - (abs_min - ic.t_start_min) / ic.duration_min  # 1→0 linearly
                inc += np.where(edge_zone, ic.magnitude * decay, 0.0)
        tm = tm * (1.0 + inc)
        return np.where(closed, np.inf, tm)  # closure overrides finite contributions (== time())

    def node_congestion(self, coords, at_minute: float = 0.0) -> np.ndarray:
        """max over incidents of the node contribution (0 without active incidents) — finite."""
        coords = np.asarray(coords, dtype=float)
        abs_min = self.offset_min + at_minute
        out = np.zeros(len(coords), dtype=np.float32)
        for inc in self.incidents:
            for j in range(len(coords)):
                lvl = inc.at_node(coords[j], abs_min)
                if lvl > out[j]:
                    out[j] = lvl
        return out
