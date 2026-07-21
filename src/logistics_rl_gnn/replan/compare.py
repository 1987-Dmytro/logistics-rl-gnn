"""Реакция на событие из ТЕКУЩЕГО частичного состояния: RL vs OR-Tools vs greedy.

Мерим ДВЕ оси на ОДНОМ residual-инстансе:
  • латентность re-plan (wall-time, end-to-end per method — build+solve, не только kernel);
  • качество после re-plan (cost/on-time/unserved) единым evaluate_solution под congestion.
RL — forward-pass (мс, no_grad), OR-Tools — re-solve на РЕАЛИСТИЧНОМ дедлайне (не 30с; при
тесном бюджете качество OR-Tools падает к RL — там честный вопрос «RL≈OR при сравнимой цене»),
greedy — контроль. Латентная гигиена: warmup (torch lazy-init) + медиана реплик.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import replace

import numpy as np
import torch

from logistics_rl_gnn.baselines.greedy import greedy_routes
from logistics_rl_gnn.baselines.ortools_vrptw import ortools_routes
from logistics_rl_gnn.env.events import make_dynamic_env
from logistics_rl_gnn.env.scoring import CostConfig, evaluate_solution

_BIG_S = 1e7  # сек: закрытое ребро в static-snapshot OR-Tools (недостижимо в Time-dim=14400с)
_Q_KEYS = ("reward", "on_time_pct", "unserved", "vehicles_used", "distance_km", "time_min")


def _snapshot_matrix_s(travel, n: int) -> np.ndarray:
    """Static congestion-снапшот (сек) на момент старта тура (at=0): OR-Tools не видит
    time-dependency → замораживаем c на час события. Закрытие → _BIG_S (арк недоступен)."""
    m = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            t = travel.time(i, j, 0.0)  # мин; offset уже внутри travel
            m[i, j] = _BIG_S if not np.isfinite(t) else t * 60.0
    return m


def _bench(solve, *, reps: int, warmup: int):
    """(routes последнего прогона, медианная латентность мс). warmup гасит torch lazy-init."""
    for _ in range(warmup):
        solve()
    ts, routes = [], []
    for _ in range(reps):
        t0 = time.perf_counter()
        routes = solve()
        ts.append((time.perf_counter() - t0) * 1000.0)
    return routes, statistics.median(ts)


def compare_replan(
    residual,
    travel,
    policy,
    *,
    fleet_size: int,
    deadline_s: int = 2,
    rl_reps: int = 3,
    warmup: int = 1,
) -> dict:
    """Re-plan residual каждым методом → {method: {latency_ms, reward, on_time_pct, ...}}."""
    cfg = CostConfig()
    n = len(residual.demand)

    def rl():
        env = make_dynamic_env(residual, travel=travel, fleet_size=fleet_size)
        with torch.no_grad():  # инференс: без autograd-графа (честная латентность)
            return policy.rollout(env, mode="greedy")[0]

    def gd():
        env = make_dynamic_env(residual, travel=travel, fleet_size=fleet_size)
        return greedy_routes(env=env)

    snap = replace(residual, time_matrix=_snapshot_matrix_s(travel, n))

    def ort():
        return ortools_routes(snap, fleet_size=fleet_size, time_limit_s=deadline_s)

    out: dict = {}
    for name, fn, reps, wu in [
        ("rl", rl, rl_reps, warmup),
        ("greedy", gd, rl_reps, warmup),
        ("ortools", ort, 1, 0),  # OR-Tools крутит GLS до дедлайна → 1 реплика
    ]:
        routes, ms = _bench(fn, reps=reps, warmup=wu)
        q = evaluate_solution(routes, residual, cfg, travel=travel)  # качество под congestion
        out[name] = {"latency_ms": ms, **{k: q[k] for k in _Q_KEYS}}
    return out
