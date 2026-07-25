"""Reaction to an event from the CURRENT partial state: RL vs OR-Tools vs greedy.

Two axes measured on ONE residual instance:
  • re-plan latency (wall time, end-to-end per method — build+solve, not just the kernel);
  • post-re-plan quality (cost/on-time/unserved) via the single evaluate_solution under congestion.
RL is a forward pass (ms, no_grad), OR-Tools re-solves against a REALISTIC deadline (not 30s; on a
tight budget OR-Tools quality drops towards RL — that is the honest question "is RL≈OR at a
comparable price"), greedy is the control. Latency hygiene: warmup (torch lazy-init) + median.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import replace

import numpy as np
import torch

from logistics_rl_gnn.baselines.greedy import greedy_routes
from logistics_rl_gnn.baselines.ortools_vrptw import ortools_routes
from logistics_rl_gnn.config import instance as im
from logistics_rl_gnn.env.events import make_dynamic_env
from logistics_rl_gnn.env.scoring import CostConfig, evaluate_solution

_BIG_S = 1e7  # s: closed edge in the OR-Tools static snapshot (unreachable within Time-dim=14400s)
_Q_KEYS = ("reward", "on_time_pct", "unserved", "vehicles_used", "distance_km", "time_min")


def _snapshot_matrix_s(travel, n: int) -> np.ndarray:
    """Static congestion snapshot (seconds) at the tour start (at=0): OR-Tools cannot see
    time-dependency → c is frozen at the event hour. A closure → _BIG_S (arc unavailable)."""
    m = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            t = travel.time(i, j, 0.0)  # minutes; the offset already lives inside travel
            m[i, j] = _BIG_S if not np.isfinite(t) else t * 60.0
    return m


def _bench(solve, *, reps: int, warmup: int):
    """(routes of the last run, median latency in ms). warmup absorbs torch lazy-init."""
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
    rl_planner=None,
) -> dict:
    """Re-plan the residual with each method → {method: {latency_ms, reward, on_time_pct, ...}}.

    rl_planner=None → method "rl" = greedy decode (single forward, Phase 7). With rl_planner
    (Step 3) → "rl" = PortfolioPlanner.plan (sample-K ∪ RL-greedy ∪ greedy); the portfolio's
    end-to-end latency goes through the same `_bench`. The rl ≤ greedy guarantee holds: the
    greedy candidate in the portfolio is built exactly as the `greedy` method here (same
    instance+travel+fleet_size+scorer)."""
    cfg = CostConfig()
    n = len(residual.demand)

    def rl():
        if rl_planner is not None:  # portfolio (reps=1: the outer _bench takes the median)
            return rl_planner.plan(residual, travel, fleet_size=fleet_size)["routes"]
        env = make_dynamic_env(residual, travel=travel, fleet_size=fleet_size)
        with torch.no_grad():  # inference: no autograd graph (honest latency)
            return policy.rollout(env, mode="greedy")[0]

    def gd():
        env = make_dynamic_env(residual, travel=travel, fleet_size=fleet_size)
        return greedy_routes(env=env)

    snap = replace(residual, time_matrix=_snapshot_matrix_s(travel, n))
    _, cap = im.fleet_of(residual)  # scenario Q (K is a parameter) — else the def-time default

    def ort():
        return ortools_routes(snap, fleet_size=fleet_size, vehicle_cap=cap,
                              time_limit_s=deadline_s)

    out: dict = {}
    for name, fn, reps, wu in [
        ("rl", rl, rl_reps, warmup),
        ("greedy", gd, rl_reps, warmup),
        ("ortools", ort, 1, 0),  # OR-Tools runs GLS until the deadline → 1 replica
    ]:
        routes, ms = _bench(fn, reps=reps, warmup=wu)
        q = evaluate_solution(routes, residual, cfg, travel=travel)  # quality under congestion
        out[name] = {"latency_ms": ms, **{k: q[k] for k in _Q_KEYS}}
    return out
