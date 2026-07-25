"""Single scorer for a solution's cost (single source of truth).

`evaluate_solution` closes the VRPEnv reward formula (dec-0001 §3): both the environment and the
baselines compute cost with ONE function → the before/after comparison shares a measure (#3).
Input — routes `[[0, i, j, 0], ...]` (depot = 0, every route starts and ends at the depot)
and an `Instance` (matrices in s/m/boxes). Internally converted to min/km — exactly as VRPEnv.

Route time = driving + waiting until e_i + service + return to the depot (same semantics as the
env: waiting is paid through c_t). Free-flow (Phase 3); Phase 7 swaps the travel model in.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from logistics_rl_gnn.config import instance as _im


@dataclass(frozen=True)
class CostConfig:
    """Costs/penalties of the reward formula. Defaults = the dec-0001 §2 calibration (im.*)."""

    c_f: float = _im.COST_PER_VEHICLE
    c_d: float = _im.COST_PER_KM
    c_t: float = _im.COST_PER_HOUR
    p_late: float = _im.PENALTY_LATE_PER_MIN
    p_unserved: float = _im.PENALTY_UNSERVED


def check_feasible(
    routes, instance, travel=None, *, t_max_min: float, vehicle_cap: float, fleet_size: int
) -> bool:
    """Hard feasibility of routes (cap/TW/T_max/fleet) — a MIRROR of the `evaluate_solution` walk.

    Does NOT compute cost (that is `evaluate_solution`), only constraints — the single source holds.
    Semantics strictly as in the ENV (`VRPEnv._feasible`, one source of feasibility): TW is HARD
    (arrival ≤ l_j; NOT via the p_late penalty), cap on demand sum ≤ Q, T_max per vehicle
    (return to depot ≤ T_max), non-empty routes ≤ fleet. Err strictly, never weaker than the env.
    The walk (arrival/wait/service) is identical to `evaluate_solution` — an agreement test guards.
    """
    time_m = np.asarray(instance.time_matrix, dtype=float) / 60.0
    win = np.asarray(instance.windows, dtype=float) / 60.0
    service = np.asarray(instance.service, dtype=float) / 60.0
    demand = np.asarray(instance.demand, dtype=float)
    eps = 1e-6
    vehicles_used = 0
    for route in routes:
        if len(route) < 2 or all(n == 0 for n in route):
            continue  # empty vehicle — as in evaluate_solution
        vehicles_used += 1
        load = t = 0.0
        for a, b in zip(route[:-1], route[1:], strict=False):
            dt = float(travel.time(a, b, t)) if travel is not None else float(time_m[a, b])
            arrival = t + dt
            if b == 0:  # return to depot: T_max per vehicle
                if arrival > t_max_min + eps:
                    return False
                t = arrival
                continue
            if arrival > win[b, 1] + eps:  # TW is hard (missed the late window)
                return False
            load += demand[b]
            if load > vehicle_cap + eps:  # cap
                return False
            t = max(arrival, win[b, 0]) + service[b]  # wait until e_b, then service
            back = float(travel.time(b, 0, t)) if travel is not None else float(time_m[b, 0])
            if t + back > t_max_min + eps:  # PER CUSTOMER: must make it back (== env._feasible)
                return False
    return vehicles_used <= fleet_size


def evaluate_solution(routes, instance, cfg: CostConfig, travel=None) -> dict:
    """Solution cost via the SINGLE VRPEnv formula.

    -> {distance_km, time_min, vehicles_used, on_time_pct, unserved, reward}.
    reward = −(c_f·vehicles + c_d·distance + c_t·time/60 + p_late·lateness + p_unserved·unserved).
    travel=None → free-flow (Phase 3, time_matrix). travel=TravelModel (Phase 7) → travel time
    depends on the departure moment t (congestion): dt = travel.time(a,b,t). Distance does NOT
    depend on congestion (distance = distance). t is in minutes from the tour start (as at_minute).
    """
    time_m = np.asarray(instance.time_matrix, dtype=float) / 60.0  # s→min (free-flow fallback)
    dist_km = np.asarray(instance.dist_matrix, dtype=float) / 1000.0  # m→km
    win = np.asarray(instance.windows, dtype=float) / 60.0  # s→min
    service = np.asarray(instance.service, dtype=float) / 60.0  # s→min
    n_customers = len(instance.demand) - 1  # depot = index 0

    served: set[int] = set()
    n_visits = on_time = vehicles_used = 0
    total_km = total_time = late_min = 0.0

    for route in routes:
        if len(route) < 2 or all(n == 0 for n in route):
            continue  # empty/skipped vehicle ([0] or [0,0])
        vehicles_used += 1  # a route with ≥1 pharmacy (== env: len(route) > 2)
        t = 0.0  # every vehicle starts at the depot with cur_time=0
        for a, b in zip(route[:-1], route[1:], strict=False):
            total_km += dist_km[a, b]
            dt = float(travel.time(a, b, t)) if travel is not None else float(time_m[a, b])
            arrival = t + dt
            if b == 0:  # return to depot: no window/service, but travel time is paid
                total_time += dt
                t = arrival
                continue
            wait = max(0.0, win[b, 0] - arrival)  # wait until the early window e_b
            late = max(0.0, arrival - win[b, 1])  # lateness past the late window l_b
            late_min += late
            n_visits += 1
            on_time += int(late <= 1e-9)
            total_time += dt + wait + service[b]
            t = arrival + wait + service[b]
            served.add(b)

    unserved = n_customers - len(served)
    on_time_pct = 100.0 if n_visits == 0 else 100.0 * on_time / n_visits
    reward = -(
        cfg.c_f * vehicles_used
        + cfg.c_d * total_km
        + cfg.c_t * total_time / 60.0
        + cfg.p_late * late_min
        + cfg.p_unserved * unserved
    )
    return {
        "distance_km": total_km,
        "time_min": total_time,
        "vehicles_used": vehicles_used,
        "on_time_pct": on_time_pct,
        "unserved": unserved,
        "reward": reward,
    }
