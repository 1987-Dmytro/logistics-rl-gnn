"""Единый оценщик стоимости решения (single source of truth).

`evaluate_solution` замыкает reward-формулу VRPEnv (dec-0001 §3): и среда, и бейзлайны
считают стоимость ОДНОЙ функцией → сравнение «Было/Стало» на общей мере (запрет №3).
Вход — маршруты `[[0, i, j, 0], ...]` (депо = 0, каждый маршрут стартует/финиширует в депо)
и `Instance` (матрицы в сек/м/боксах). Внутри переводим в мин/км — как VRPEnv.

Время маршрута = езда + ожидание до e_i + сервис + возврат в депо (та же семантика, что в
среде: ожидание оплачивается через c_t). Free-flow (Phase 3); Phase 7 подменит travel-модель.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from logistics_rl_gnn.config import instance as _im


@dataclass(frozen=True)
class CostConfig:
    """Стоимости/штрафы reward-формулы. Дефолты = калибровка dec-0001 §2 (im.*)."""

    c_f: float = _im.COST_PER_VEHICLE
    c_d: float = _im.COST_PER_KM
    c_t: float = _im.COST_PER_HOUR
    p_late: float = _im.PENALTY_LATE_PER_MIN
    p_unserved: float = _im.PENALTY_UNSERVED


def evaluate_solution(routes, instance, cfg: CostConfig, travel=None) -> dict:
    """Стоимость решения ЕДИНОЙ формулой VRPEnv.

    -> {distance_km, time_min, vehicles_used, on_time_pct, unserved, reward}.
    reward = −(c_f·машин + c_d·пробег + c_t·время/60 + p_late·опоздание + p_unserved·необслуж).
    travel=None → free-flow (Phase 3, time_matrix). travel=TravelModel (Phase 7) → время в пути
    зависит от момента отправления t (congestion): dt = travel.time(a,b,t). Пробег НЕ зависит от
    congestion (distance = distance). Момент t в минутах от старта тура (как at_minute).
    """
    time_m = np.asarray(instance.time_matrix, dtype=float) / 60.0  # сек→мин (free-flow фолбэк)
    dist_km = np.asarray(instance.dist_matrix, dtype=float) / 1000.0  # м→км
    win = np.asarray(instance.windows, dtype=float) / 60.0  # сек→мин
    service = np.asarray(instance.service, dtype=float) / 60.0  # сек→мин
    n_customers = len(instance.demand) - 1  # депо = индекс 0

    served: set[int] = set()
    n_visits = on_time = vehicles_used = 0
    total_km = total_time = late_min = 0.0

    for route in routes:
        if len(route) < 2 or all(n == 0 for n in route):
            continue  # пустая/пропущенная машина ([0] или [0,0])
        vehicles_used += 1  # маршрут с ≥1 аптекой (== env: len(route) > 2)
        t = 0.0  # каждая машина стартует в депо при cur_time=0
        for a, b in zip(route[:-1], route[1:], strict=False):
            total_km += dist_km[a, b]
            dt = float(travel.time(a, b, t)) if travel is not None else float(time_m[a, b])
            arrival = t + dt
            if b == 0:  # возврат в депо: без окна/сервиса, но время в пути платим
                total_time += dt
                t = arrival
                continue
            wait = max(0.0, win[b, 0] - arrival)  # ждём до раннего окна e_b
            late = max(0.0, arrival - win[b, 1])  # опоздание за позднее окно l_b
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
