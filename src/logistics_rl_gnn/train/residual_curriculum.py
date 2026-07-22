"""Path B — residual-curriculum обучение (Phase 6b, предрегистрация [[0011]]).

Дообучение congestion-best НА residual-распределении, чтобы ОДИНОЧНЫЙ greedy-decode политики стал
лучше greedy на re-plan (гейт 0011). Ключ: `residual_instance` даёт СВЕЖИЙ CVRPTW (депо +
необслуженные, окна сдвинуты) → POMO работает БЕЗ изменений: `feasible_starts` на residual-env =
«K допустимых СЛЕДУЮЩИХ узлов»; `_decode`/shared-baseline/`train_batch` переиспользуются как есть.

Микс 50/50: полные congestion-эпизоды (anti-catastrophic-forgetting) + residual-эпизоды. Residual:
InstanceSampler(62,62)-инстанс → greedy-префикс до frac∈[0.2,0.8] прогресса → одно событие
(traffic/urgent/breakdown) → residual_instance. Отбор best-by-val-residual (single-decode = метрика
гейта 0011). База — InstanceSampler(62,62) (кэш; generate_instance reload ~4с/вызов непригоден).
Held-out по СИДУ (0011): train ≥3M, val 4M–4M+47, дизъюнктны с гейтом 0–9.
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
    inst: object  # residual CVRPTW Instance (депо + необслуженные + срочные)
    travel: object  # CongestionTravel residual'а (offset=now, накопленные инциденты)
    fleet: int  # машин осталось (base_k − broken)
    frac: float  # реализованная доля прогресса префикса
    kind: str  # событие: traffic|urgent|breakdown


def _finish_times(routes, instance, travel) -> dict:
    """Время завершения сервиса каждого клиента при исполнении routes под travel (мин).

    Зеркало time-walk `served_by`, но возвращает сами finish-таймы (для выбора now_min под frac)."""
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
    """now_min, при котором ровно round(frac·n) клиентов завершены (между k-м и k+1-м finish)."""
    times = sorted(fin.values())
    n = len(times)
    k = max(1, min(n - 1, int(round(frac * n))))  # ≥1 обслужен И ≥1 остался
    return 0.5 * (times[k - 1] + times[k])


def _sample_event(rng, instance, at_min: float):
    """Одно событие-триггер (traffic|urgent|breakdown) на now_min — как в `event_stream`."""
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
            "delta_s": float(rng.uniform(30.0, 75.0) * 60.0),  # узкое окно 30–75 мин
        }
        return UrgentEvent(at_min, order)
    return BreakdownEvent(at_min)


def make_residual(sampler, seed, *, dow, base_k, frac_range, max_tries: int = 8) -> Residual:
    """residual-состояние: инстанс → greedy-префикс до frac → событие → `residual_instance`.

    Отбраковывает вырожденные состояния (<3 pending или <2 допустимых стартов) ресэмплом
    frac/события (инстанс на seed фиксирован). Детерминирован по seed → воспроизводимый val-пул.
    """
    inst = sampler.sample(int(seed))  # кэш-сэмплер: full-62, seed-варьированный спрос (0011)
    exec_travel = congestion_for(inst, dow=dow)  # диурнал = timeline исполнения (как iter_events)
    exec_routes = greedy_routes(env=make_dynamic_env(inst, travel=exec_travel, fleet_size=base_k))
    fin = _finish_times(exec_routes, inst, exec_travel)
    if len(fin) < 4:
        raise RuntimeError(f"residual seed={seed}: greedy обслужил {len(fin)}<4 — база вырождена")
    for t in range(max_tries):
        rng = np.random.default_rng(int(seed) * 131 + t + 999)
        frac = float(rng.uniform(*frac_range))
        now_min = _now_for_progress(fin, frac)
        state = DynamicState(inst, dow, now_min=now_min)
        state.served = served_by(exec_routes, inst, exec_travel, now_min)  # served как гейт
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
        if len(feasible_starts(env, obs, 2)) >= 2:  # POMO shared baseline требует ≥2 старта
            return Residual(res, travel, fleet, frac, ev.kind)
    raise RuntimeError(f"residual seed={seed}: не построено не-вырожденное за {max_tries} попыток")


def single_decode_cost(policy, env) -> float:
    """Стоимость ОДИНОЧНОГО greedy-decode (== метрика гейта rl_raw 0011) под travel среды."""
    with torch.no_grad():
        routes = policy.rollout(env, mode="greedy")[0]
    return -evaluate_solution(routes, env._inst, env._cost_cfg, travel=env.travel)["reward"]


def greedy_cost(env) -> float:
    """Стоимость greedy-эвристики (== greedy_raw гейта) под travel среды."""
    gr = greedy_routes(env=env)
    return -evaluate_solution(gr, env._inst, env._cost_cfg, travel=env.travel)["reward"]


class ResidualPOMOTrainer(POMOTrainer):
    """POMO с residual-куррикулумом (0011): микс 50/50, отбор best-by-val-residual single-decode."""

    def __init__(self, policy, cfg, *, val_ort=None):
        super().__init__(policy, cfg, val_ort=val_ort)  # полн. val/probe (anti-forgetting монитор)
        self.base_k = im.FLEET_SIZE
        self.res_sampler = InstanceSampler(n_range=(62, 62))  # full-62, кэш снапшота; residual-база
        # фикс. held-out residual-val пул (greedy-префикс детерминирован по сиду → воспроизводимо)
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
        """50/50 (seeded): residual-эпизод (база seed≥res_train_base) либо полный congestion."""
        rng = np.random.default_rng(int(seed) * 97 + 12345)
        if rng.random() < self.cfg.residual_frac:  # residual-половина (учит re-plan-старт)
            r = self._make_res(self.cfg.res_train_base + int(seed))
            env = make_dynamic_env(r.inst, travel=r.travel, fleet_size=r.fleet)
            return r.inst.node_ids, env
        return super()._batch_instance_env(seed)  # полн. congestion (anti-catastrophic-forgetting)

    def _validate(self) -> dict:
        rec = super()._validate()  # полн. congestion: val_cost(multistart), gap_greedy — МОНИТОР
        rec["val_full_cost"] = rec["val_cost"]  # anti-forgetting ось (не критерий отбора)
        res = [single_decode_cost(self.policy, e) for e in self.val_res_envs]  # == метрика гейта
        m = float(np.mean(res))
        rec["val_res_cost"] = m
        rec["val_res_gap_greedy"] = m / self.val_res_heur - 1.0  # <0 → RL-старт бьёт greedy (гейт)
        rec["val_cost"] = m  # ОТБОР best-by-val ПО residual single-decode (0011)
        return rec
