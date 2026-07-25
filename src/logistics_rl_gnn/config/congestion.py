"""Congestion profile c(dow,h) + incident parameters (Phase 7, dec-0001 §4).

t(e,τ) = t0(e)·c(dow,h(τ))·(1 + Σ_k I_k(e,τ)). The road class is unavailable in the snapshot's
OD matrix (built with `with_graph=False`) → the class axis is collapsed into ONE urban profile;
the shape is calibrated against the TomTom Traffic Index curve (Augsburg): morning (08) and
evening (17) peaks, midday dip. Magnitudes are ASSUMED → tagged simulated-on-real. Everything
is deterministic (seed).
"""

from __future__ import annotations

CALIBRATION_TAG = "simulated-on-real"  # TomTom Augsburg shape is real, magnitudes synthetic

# c by hour of day (weekdays), 1.0 = free-flow. TomTom shape: two peaks (08, 17), midday dip.
# Outside the range — extrapolated by the edge values (clamped in congestion()).
_HOURLY_WEEKDAY: dict[int, float] = {
    6: 1.10, 7: 1.22, 8: 1.30, 9: 1.20, 10: 1.10, 11: 1.08,
    12: 1.12, 13: 1.10, 14: 1.08, 15: 1.12, 16: 1.22, 17: 1.32,
    18: 1.18, 19: 1.10, 20: 1.05,
}  # fmt: skip
# dow factor (Mon=0..Sun=6): weekdays get the full congestion amplitude, weekends are lighter.
_DOW_FACTOR: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0, 0.7, 0.5)

_HRS = sorted(_HOURLY_WEEKDAY)


def congestion(dow: int, hour: float) -> float:
    """Multiplier c(dow,h) ≥ 1. hour is clamped to the table range (edges extrapolate).

    The dow factor damps the AMPLITUDE (c−1), not c itself → weekends sit closer to free-flow,
    weekdays keep the full peak.
    """
    h = int(min(max(hour, _HRS[0]), _HRS[-1]))
    base = _HOURLY_WEEKDAY[h]
    f = _DOW_FACTOR[int(dow) % 7]
    return 1.0 + (base - 1.0) * f


# --- incidents I_k (dec-0001 §4): Poisson spawn, the rate rises during peaks ---
INCIDENT_BASE_RATE_PER_H = 0.6  # base λ (events/hour); scaled by (c−1) during a peak hour
INCIDENT_MAG_RANGE = (0.4, 1.8)  # δ: contribution (1+δ·decay) inside the zone
INCIDENT_DUR_RANGE_MIN = (15.0, 60.0)  # incident duration, minutes (linear decay)
INCIDENT_RADIUS_KM = 1.2  # zone radius from the centre node, km
INCIDENT_CLOSURE_PROB = 0.08  # δ=∞ (edge closure → removed from the graph)
