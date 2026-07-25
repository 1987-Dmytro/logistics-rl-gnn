"""OR-Tools CVRPTW — the strong baseline (prohibition #3). Requires the `baselines` dep group.

The model works in native s/m/boxes (OR-Tools = integer domains):
  • transit time = travel + service(from), CEIL to seconds → conservative (OR-feasible ⇒ float
    on-time, no lateness from sub-second rounding);
  • Time-dim: windows [e_i, floor(l_i)], horizon/capacity = T_max (return to depot ≤ T_max),
    fix_start_cumul_to_zero (every vehicle starts at t=0);
  • TIME COST via the Time-dim span cost (span = driving+service+waiting = time_min from the
    single scorer) → OR-Tools minimises the scaled reward rather than its own proxy;
  • arc-cost = c_d·dist (the only per-arc term), fixed = c_f, Capacity-dim (demands, Q),
    Distance-dim (a cross-check on distance);
  • AddDisjunction penalty = p_unserved (the same trade-off as in the reward) → a feasible
    instance is served in full.
Routes are SCORED by the single evaluate_solution (NOT by the internal OR-Tools objective).
"""

from __future__ import annotations

import math

import numpy as np

from logistics_rl_gnn.config import instance as im
from logistics_rl_gnn.env.scoring import CostConfig

# integer scale for monetary values: the time coefficient = round(SCALE·c_t/3600); at SCALE=1e6
# the coefficient distortion is < 0.01% (int64 with room to spare).
_SCALE = 1_000_000


def ortools_routes(
    instance,
    cfg: CostConfig | None = None,
    *,
    fleet_size: int = im.FLEET_SIZE,
    vehicle_cap: float = im.VEHICLE_CAP,
    t_max_min: float = im.T_MAX_MIN,
    time_limit_s: float = 30,
) -> list[list[int]]:
    """CVRPTW solution by OR-Tools. Returns routes (env format `[[0,...,0], ...]`)."""
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2

    cfg = cfg or CostConfig()
    n = len(instance.demand)
    K = int(fleet_size)
    tm = np.asarray(instance.time_matrix, dtype=float)  # seconds
    dm = np.asarray(instance.dist_matrix, dtype=float)  # metres
    win = np.asarray(instance.windows, dtype=float)  # seconds
    sv = np.asarray(instance.service, dtype=float)  # seconds
    dem = [int(x) for x in instance.demand]
    tmax_s = int(t_max_min * 60)

    manager = pywrapcp.RoutingIndexManager(n, K, 0)
    routing = pywrapcp.RoutingModel(manager)

    def time_cb(i, j):  # travel + service(from), ceil → conservative towards windows/T_max
        a, b = manager.IndexToNode(i), manager.IndexToNode(j)
        return int(math.ceil(tm[a, b] + sv[a]))

    time_idx = routing.RegisterTransitCallback(time_cb)

    def dist_cost_cb(i, j):  # arc-cost = c_d·dist (scaled to int)
        a, b = manager.IndexToNode(i), manager.IndexToNode(j)
        return int(round(_SCALE * cfg.c_d * dm[a, b] / 1000.0))

    routing.SetArcCostEvaluatorOfAllVehicles(routing.RegisterTransitCallback(dist_cost_cb))
    routing.SetFixedCostOfAllVehicles(int(round(_SCALE * cfg.c_f)))  # c_f per vehicle

    # --- Time-dim: TW + T_max + the cost of time through the span ---
    routing.AddDimension(time_idx, tmax_s, tmax_s, True, "Time")
    time_dim = routing.GetDimensionOrDie("Time")
    time_dim.SetSpanCostCoefficientForAllVehicles(int(round(_SCALE * cfg.c_t / 3600.0)))
    for node in range(1, n):
        e = int(math.floor(win[node, 0]))
        hi = min(int(math.floor(win[node, 1])), tmax_s)  # floor(l) + clip to T_max
        time_dim.CumulVar(manager.NodeToIndex(node)).SetRange(e, hi)

    # --- Capacity-dim (demands, Q) ---
    dem_idx = routing.RegisterUnaryTransitCallback(lambda i: dem[manager.IndexToNode(i)])
    routing.AddDimensionWithVehicleCapacity(dem_idx, 0, [int(vehicle_cap)] * K, True, "Capacity")

    # --- Distance-dim (cross-check on distance; carries no constraint) ---
    def dist_cb(i, j):
        a, b = manager.IndexToNode(i), manager.IndexToNode(j)
        return int(round(dm[a, b]))

    routing.AddDimension(routing.RegisterTransitCallback(dist_cb), 0, 10**9, True, "Distance")

    # --- disjunction: a customer may be dropped for a penalty = p_unserved ---
    penalty = int(round(_SCALE * cfg.p_unserved))
    for node in range(1, n):
        routing.AddDisjunction([manager.NodeToIndex(node)], penalty)

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    # budget as a Duration (sec+nanos): sub-second ones (0.7s) for the anytime curve (task #15);
    # int-compatible (30 → 30s0ns == the old FromSeconds(30), 0002 parity holds). seconds/nanos is
    # more version-robust than FromMilliseconds. Best-so-far at budget end → cost is time-bound.
    params.time_limit.seconds = int(time_limit_s)
    params.time_limit.nanos = int(round((time_limit_s - int(time_limit_s)) * 1e9))
    params.log_search = False
    # the algorithm (PATH_CHEAPEST_ARC + GLS) is deterministic; only the number of iterations in
    # the budget varies → reproducibility via a fixed config (version+budget), not wall-clock.

    solution = routing.SolveWithParameters(params)
    if solution is None:
        return []

    routes: list[list[int]] = []
    for v in range(K):
        idx = routing.Start(v)
        route = [manager.IndexToNode(idx)]
        while not routing.IsEnd(idx):
            idx = solution.Value(routing.NextVar(idx))
            route.append(manager.IndexToNode(idx))
        if any(node != 0 for node in route):  # skip vehicles that were never used
            routes.append(route)
    return routes
