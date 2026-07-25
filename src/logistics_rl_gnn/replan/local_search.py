"""Local-search polish of decoded routes (Phase 6b Step 3.5). NO training.

`polish` improves routes with classic local search:
  • intra-route: 2-opt (reverse an inner segment) + Or-opt (move a 1–3 segment within a route);
  • inter-route: relocate (customer A→B) + swap (customer A ↔ customer B).
Cost comes ONLY from `evaluate_solution` (a full re-evaluation of the candidate → correct under
time-dependent congestion, WITHOUT delta-cost assumptions: a reversal/move shifts every downstream
time). Feasibility comes from `check_feasible` (hard as the env: cap/TW/T_max/fleet).
First-improvement, looping until convergence or `budget_ms`. **INVARIANT: the result is NEVER
worse than the input** (best starts as the input and is replaced only by a strictly better feasible
candidate). Operators redistribute over EXISTING slots — they never add a route (vehicles_used
never grows beyond fleet; empty [0,0] slots give relocate room for a vehicle).
"""

from __future__ import annotations

import time
from itertools import chain

from logistics_rl_gnn.config import instance as im
from logistics_rl_gnn.env.scoring import CostConfig, check_feasible, evaluate_solution


def _cost(routes, instance, cfg, travel) -> float:
    return -evaluate_solution(routes, instance, cfg, travel=travel)["reward"]


def _normalize(routes, fleet_size):
    """Every route → at least [0,0]; pad with empty slots up to fleet_size (room for relocate)."""
    norm = [list(r) if len(r) >= 2 else [0, 0] for r in routes]
    if fleet_size is not None:
        while len(norm) < fleet_size:
            norm.append([0, 0])
    return norm


def _two_opt(routes):
    """Reverse an inner segment of one route (depot ends untouched)."""
    for ri in range(len(routes)):
        r = routes[ri]
        for i in range(1, len(r) - 2):
            for j in range(i + 1, len(r) - 1):
                cand = [x[:] for x in routes]
                cand[ri][i : j + 1] = r[i : j + 1][::-1]
                yield cand


def _or_opt(routes):
    """Move a 1–3 segment to another position of the SAME route (identity → dropped by strict<)."""
    for ri in range(len(routes)):
        r = routes[ri]
        for seg_len in (1, 2, 3):
            for p in range(1, len(r) - seg_len):  # segment [p, p+seg_len) — inside the interior
                seg = r[p : p + seg_len]
                rest = r[:p] + r[p + seg_len :]
                for q in range(1, len(rest)):  # insertion between the depot ends of rest
                    cand = [x[:] for x in routes]
                    cand[ri] = rest[:q] + seg + rest[q:]
                    yield cand


def _relocate(routes):
    """Move a customer from route A to route B (B may be an empty [0,0] → a new vehicle)."""
    for ai in range(len(routes)):
        ra = routes[ai]
        for p in range(1, len(ra) - 1):
            cust = ra[p]
            src = ra[:p] + ra[p + 1 :]
            for bi in range(len(routes)):
                if bi == ai:
                    continue
                rb = routes[bi]
                for q in range(1, len(rb)):
                    cand = [x[:] for x in routes]
                    cand[ai] = src
                    cand[bi] = rb[:q] + [cust] + rb[q:]
                    yield cand


def _swap(routes):
    """Swap a customer of route A with a customer of route B (pairs ai<bi — no duplicates)."""
    for ai in range(len(routes)):
        ra = routes[ai]
        for pa in range(1, len(ra) - 1):
            for bi in range(ai + 1, len(routes)):
                rb = routes[bi]
                for pb in range(1, len(rb) - 1):
                    cand = [x[:] for x in routes]
                    cand[ai] = ra[:pa] + [rb[pb]] + ra[pa + 1 :]
                    cand[bi] = rb[:pb] + [ra[pa]] + rb[pb + 1 :]
                    yield cand


def _moves(routes):
    """Candidates in a FIXED order (determinism): 2-opt → Or-opt → relocate → swap."""
    return chain(_two_opt(routes), _or_opt(routes), _relocate(routes), _swap(routes))


def polish(
    routes,
    instance,
    travel=None,
    *,
    budget_ms: float = 200.0,
    t_max_min: float = im.T_MAX_MIN,
    vehicle_cap: float = im.VEHICLE_CAP,
    fleet_size: int | None = None,
    cfg: CostConfig | None = None,
) -> tuple[list, float]:
    """Polish routes with local search. -> (routes, cost€). Never worse than input; env feasibility.

    budget_ms — the wall-clock cap (set it generous when measuring convergence). fleet_size=None →
    inferred from the input (number of slots). Cost/feasibility — the single scorer/check_feasible.
    """
    cfg = cfg or CostConfig()
    best = _normalize(routes, fleet_size)
    fleet = fleet_size if fleet_size is not None else len(best)

    def feas(rs):
        return check_feasible(
            rs, instance, travel, t_max_min=t_max_min, vehicle_cap=vehicle_cap, fleet_size=fleet
        )

    best_cost = _cost(best, instance, cfg, travel)
    if not feas(best):  # infeasible input → return as-is (invariant: never worse than the input)
        return best, best_cost
    deadline = time.perf_counter() + budget_ms / 1000.0
    improved = True
    while improved and time.perf_counter() < deadline:
        improved = False
        for cand in _moves(best):
            if time.perf_counter() >= deadline:
                break
            c = _cost(cand, instance, cfg, travel)
            if c < best_cost - 1e-9 and feas(cand):  # strictly better AND feasible → take it
                best, best_cost, improved = cand, c, True
                break  # restart enumerate from the new best
    return best, best_cost
