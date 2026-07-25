"""Path B — residual-curriculum training (Phase 6b, pre-registration [[0011]]).

Fine-tunes congestion-best ON the residual distribution so that the policy's SINGLE greedy decode
beats greedy on re-plan (gate 0011). The key: `residual_instance` yields a FRESH CVRPTW (depot +
unserved, windows shifted) → POMO works UNCHANGED: `feasible_starts` on a residual env = "K
allowed NEXT nodes"; `_decode`/shared-baseline/`train_batch` are reused as they are.

A 50/50 mix: full congestion episodes (anti-catastrophic-forgetting) + residual episodes. Residual:
an InstanceSampler(62,62) instance → greedy prefix up to frac∈[0.2,0.8] of progress → one event
(traffic/urgent/breakdown) → residual_instance. Selection is best-by-val-residual (single decode =
the 0011 gate metric). The base is InstanceSampler(62,62) (cached; generate_instance reload ~4s per
call is unusable). Held out BY SEED (0011): train ≥3M, val 4M–4M+47, disjoint from gate 0–9.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from logistics_rl_gnn.baselines.greedy import greedy_routes
from logistics_rl_gnn.config import congestion as cg
from logistics_rl_gnn.config import instance as im
from logistics_rl_gnn.env.events import (
    BreakdownEvent,
    DynamicState,
    TrafficEvent,
    UrgentEvent,
    congestion_for,
    make_dynamic_env,
    residual_instance,
    served_by,
)
from logistics_rl_gnn.env.scoring import evaluate_solution
from logistics_rl_gnn.env.travel import Incident
from logistics_rl_gnn.train.instance_sampler import InstanceSampler
from logistics_rl_gnn.train.pomo import POMOTrainer, feasible_starts


@dataclass
class Residual:
    inst: object  # residual CVRPTW Instance (depot + unserved + urgent)
    travel: object  # CongestionTravel of the residual (offset=now, accumulated incidents)
    fleet: int  # vehicles left (base_k − broken)
    frac: float  # realised share of prefix progress
    kind: str  # event: traffic|urgent|breakdown


def _finish_times(routes, instance, travel) -> dict:
    """Service completion time of every customer when executing routes under travel (minutes).

    Mirrors the `served_by` time-walk but returns the finish times (to pick now_min for frac)."""
    win = np.asarray(instance.windows, dtype=float) / 60.0
    svc = np.asarray(instance.service, dtype=float) / 60.0
    fin: dict = {}
    for route in routes:
        t = 0.0
        for a, b in zip(route[:-1], route[1:], strict=False):
            arrival = t + float(travel.time(a, b, t))
            if b == 0:
                t = arrival
                continue
            finish = max(arrival, win[b, 0]) + svc[b]
            fin[int(b)] = finish
            t = finish
    return fin


def _now_for_progress(fin: dict, frac: float) -> float:
    """now_min at which exactly round(frac·n) customers are done (between finish k and k+1)."""
    times = sorted(fin.values())
    n = len(times)
    k = max(1, min(n - 1, int(round(frac * n))))  # ≥1 served AND ≥1 left
    return 0.5 * (times[k - 1] + times[k])


def _sample_event(rng, instance, at_min: float):
    """One trigger event (traffic|urgent|breakdown) at now_min — as in `event_stream`."""
    n = len(instance.demand)
    kind = str(rng.choice(["traffic", "urgent", "breakdown"]))
    if kind == "traffic":
        center = tuple(instance.coords[int(rng.integers(1, n))])
        closure = bool(rng.random() < cg.INCIDENT_CLOSURE_PROB)
        mag = np.inf if closure else float(rng.uniform(*cg.INCIDENT_MAG_RANGE))
        dur = float(rng.uniform(*cg.INCIDENT_DUR_RANGE_MIN))
        return TrafficEvent(at_min, Incident(center, cg.INCIDENT_RADIUS_KM, mag, at_min, dur))
    if kind == "urgent":
        order = {
            "idx": int(rng.integers(1, n)),
            "demand": int(rng.integers(im.DEMAND_RANGE[0], im.DEMAND_RANGE[1] + 1)),
            "delta_s": float(rng.uniform(30.0, 75.0) * 60.0),  # narrow window 30–75 min
        }
        return UrgentEvent(at_min, order)
    return BreakdownEvent(at_min)


def make_residual(sampler, seed, *, dow, base_k, frac_range, max_tries: int = 8) -> Residual:
    """residual state: instance → greedy prefix up to frac → event → `residual_instance`.

    Degenerate states (<3 pending or <2 allowed starts) are rejected by resampling frac/the event
    (the instance is fixed by seed). Deterministic per seed → a reproducible val pool.
    """
    inst = sampler.sample(int(seed))  # cached sampler: full-62, seed-varied demand (0011)
    exec_travel = congestion_for(inst, dow=dow)  # diurnal = execution timeline (as iter_events)
    exec_routes = greedy_routes(env=make_dynamic_env(inst, travel=exec_travel, fleet_size=base_k))
    fin = _finish_times(exec_routes, inst, exec_travel)
    if len(fin) < 4:
        raise RuntimeError(f"residual seed={seed}: greedy served {len(fin)}<4 — degenerate base")
    for t in range(max_tries):
        rng = np.random.default_rng(int(seed) * 131 + t + 999)
        frac = float(rng.uniform(*frac_range))
        now_min = _now_for_progress(fin, frac)
        state = DynamicState(inst, dow, now_min=now_min)
        state.served = served_by(exec_routes, inst, exec_travel, now_min)  # served as in the gate
        ev = _sample_event(rng, inst, now_min)
        ev.apply(state)
        pending = [i for i in range(1, len(inst.demand)) if i not in state.served]
        for u in state.urgent:
            if u["idx"] not in pending:
                pending.append(u["idx"])
        if len(pending) < 3:
            continue
        res = residual_instance(state)
        travel = congestion_for(res, dow=dow, offset_min=now_min, incidents=state.incidents)
        fleet = state.fleet(base_k)
        env = make_dynamic_env(res, travel=travel, fleet_size=fleet)
        obs, _ = env.reset(seed=0)
        if len(feasible_starts(env, obs, 2)) >= 2:  # POMO shared baseline needs ≥2 starts
            return Residual(res, travel, fleet, frac, ev.kind)
    raise RuntimeError(f"residual seed={seed}: no non-degenerate state in {max_tries} tries")


def single_decode_cost(policy, env) -> float:
    """Cost of a SINGLE greedy decode (== the rl_raw 0011 gate metric) under the env travel."""
    with torch.no_grad():
        routes = policy.rollout(env, mode="greedy")[0]
    return -evaluate_solution(routes, env._inst, env._cost_cfg, travel=env.travel)["reward"]


def greedy_cost(env) -> float:
    """Cost of the greedy heuristic (== the greedy_raw of the gate) under the env travel."""
    gr = greedy_routes(env=env)
    return -evaluate_solution(gr, env._inst, env._cost_cfg, travel=env.travel)["reward"]


class ResidualPOMOTrainer(POMOTrainer):
    """POMO with a residual curriculum (0011): 50/50 mix, best-by-val-residual single decode."""

    def __init__(self, policy, cfg, *, val_ort=None):
        super().__init__(policy, cfg, val_ort=val_ort)  # full val/probe (anti-forgetting monitor)
        self.base_k = im.FLEET_SIZE
        self.res_sampler = InstanceSampler(n_range=(62, 62))  # full-62, cached; residual base
        # fixed held-out residual val pool (the greedy prefix is seed-deterministic → reproducible)
        self.val_res = [self._make_res(s) for s in cfg.res_val_seeds()]
        self.val_res_envs = [
            make_dynamic_env(r.inst, travel=r.travel, fleet_size=r.fleet) for r in self.val_res
        ]
        self.val_res_heur = float(np.mean([greedy_cost(e) for e in self.val_res_envs]))

    def _make_res(self, seed) -> Residual:
        return make_residual(
            self.res_sampler, seed, dow=self.cfg.cong_dow, base_k=self.base_k,
            frac_range=self.cfg.res_frac_range,
        )

    def _batch_instance_env(self, seed):
        """50/50 (seeded): a residual episode (seed≥res_train_base) or a full congestion one."""
        rng = np.random.default_rng(int(seed) * 97 + 12345)
        if rng.random() < self.cfg.residual_frac:  # residual half (teaches the re-plan start)
            r = self._make_res(self.cfg.res_train_base + int(seed))
            env = make_dynamic_env(r.inst, travel=r.travel, fleet_size=r.fleet)
            return r.inst.node_ids, env
        return super()._batch_instance_env(seed)  # full congestion (anti-catastrophic-forgetting)

    def _validate(self) -> dict:
        rec = super()._validate()  # full congestion: val_cost(multistart), gap_greedy — MONITOR
        rec["val_full_cost"] = rec["val_cost"]  # anti-forgetting axis (not the selection criterion)
        res = [single_decode_cost(self.policy, e) for e in self.val_res_envs]  # == the gate metric
        m = float(np.mean(res))
        rec["val_res_cost"] = m
        rec["val_res_gap_greedy"] = m / self.val_res_heur - 1.0  # <0 → the RL start beats greedy
        rec["val_cost"] = m  # SELECTION best-by-val BY residual single decode (0011)
        return rec
