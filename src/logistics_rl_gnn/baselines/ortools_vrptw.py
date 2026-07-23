"""OR-Tools CVRPTW — сильный бейзлайн (запрет №3). Требует группу деп `baselines`.

Модель на нативных сек/м/боксах (OR-Tools = int-домены):
  • transit-время = travel + service(from), CEIL до сек → консервативно (OR-feasible ⇒ float
    on-time, без опозданий от суб-секундного округления);
  • Time-dim: окна [e_i, floor(l_i)], горизонт/капасити = T_max (возврат в депо ≤ T_max),
    fix_start_cumul_to_zero (каждая машина стартует в t=0);
  • СТОИМОСТЬ ВРЕМЕНИ через span-cost Time-dim (span = езда+сервис+ожидание = time_min из
    единого оценщика) → OR-Tools минимизирует масштабированный reward, а не свой прокси;
  • arc-cost = c_d·dist (единственный слагаемый на дуге), fixed = c_f, Capacity-dim (demands, Q),
    Distance-dim (кросс-чек пробега);
  • AddDisjunction penalty = p_unserved (тот же trade-off, что в reward) → feasible-инстанс
    обслуживается целиком.
Маршруты СЧИТАЕТ единый evaluate_solution (НЕ внутренний objective OR-Tools).
"""

from __future__ import annotations

import math

import numpy as np

from logistics_rl_gnn.config import instance as im
from logistics_rl_gnn.env.scoring import CostConfig

# int-масштаб денежных величин: coeff времени = round(SCALE·c_t/3600); при SCALE=1e6
# искажение коэффициента < 0.01% (int64 с запасом).
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
    """Решение CVRPTW OR-Tools. Возвращает routes (env-формат `[[0,...,0], ...]`)."""
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2

    cfg = cfg or CostConfig()
    n = len(instance.demand)
    K = int(fleet_size)
    tm = np.asarray(instance.time_matrix, dtype=float)  # сек
    dm = np.asarray(instance.dist_matrix, dtype=float)  # м
    win = np.asarray(instance.windows, dtype=float)  # сек
    sv = np.asarray(instance.service, dtype=float)  # сек
    dem = [int(x) for x in instance.demand]
    tmax_s = int(t_max_min * 60)

    manager = pywrapcp.RoutingIndexManager(n, K, 0)
    routing = pywrapcp.RoutingModel(manager)

    def time_cb(i, j):  # travel + service(from), ceil → консервативно к окнам/T_max
        a, b = manager.IndexToNode(i), manager.IndexToNode(j)
        return int(math.ceil(tm[a, b] + sv[a]))

    time_idx = routing.RegisterTransitCallback(time_cb)

    def dist_cost_cb(i, j):  # arc-cost = c_d·dist (масштаб в int)
        a, b = manager.IndexToNode(i), manager.IndexToNode(j)
        return int(round(_SCALE * cfg.c_d * dm[a, b] / 1000.0))

    routing.SetArcCostEvaluatorOfAllVehicles(routing.RegisterTransitCallback(dist_cost_cb))
    routing.SetFixedCostOfAllVehicles(int(round(_SCALE * cfg.c_f)))  # c_f за машину

    # --- Time-dim: TW + T_max + стоимость времени через span ---
    routing.AddDimension(time_idx, tmax_s, tmax_s, True, "Time")
    time_dim = routing.GetDimensionOrDie("Time")
    time_dim.SetSpanCostCoefficientForAllVehicles(int(round(_SCALE * cfg.c_t / 3600.0)))
    for node in range(1, n):
        e = int(math.floor(win[node, 0]))
        hi = min(int(math.floor(win[node, 1])), tmax_s)  # floor(l) + клип к T_max
        time_dim.CumulVar(manager.NodeToIndex(node)).SetRange(e, hi)

    # --- Capacity-dim (demands, Q) ---
    dem_idx = routing.RegisterUnaryTransitCallback(lambda i: dem[manager.IndexToNode(i)])
    routing.AddDimensionWithVehicleCapacity(dem_idx, 0, [int(vehicle_cap)] * K, True, "Capacity")

    # --- Distance-dim (кросс-чек пробега; ограничения не несёт) ---
    def dist_cb(i, j):
        a, b = manager.IndexToNode(i), manager.IndexToNode(j)
        return int(round(dm[a, b]))

    routing.AddDimension(routing.RegisterTransitCallback(dist_cb), 0, 10**9, True, "Distance")

    # --- disjunction: клиента можно бросить за штраф = p_unserved ---
    penalty = int(round(_SCALE * cfg.p_unserved))
    for node in range(1, n):
        routing.AddDisjunction([manager.NodeToIndex(node)], penalty)

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    # бюджет как Duration (sec+nanos): суб-секундные (0.7с) для anytime-кривой (задача #15);
    # int-совместимо (30 → 30s0ns == прежний FromSeconds(30), парити 0002 цела). seconds/nanos —
    # версия-робастнее FromMilliseconds. Best-so-far на истечении бюджета → cost wall-clock-bound.
    params.time_limit.seconds = int(time_limit_s)
    params.time_limit.nanos = int(round((time_limit_s - int(time_limit_s)) * 1e9))
    params.log_search = False
    # алгоритм (PATH_CHEAPEST_ARC + GLS) детерминистичен; варьирует лишь число итераций в
    # бюджете → воспроизводимость через фикс. config (версия+бюджет), не wall-clock (best-so-far).

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
        if any(node != 0 for node in route):  # пропускаем незадействованные машины
            routes.append(route)
    return routes
