"""События динамического эпизода + residual re-plan (Phase 7, dec-0001 §4).

Три события (триггеры re-plan): traffic (инцидент на зоне рёбер), breakdown (убрать машину →
её стопы остаются в пуле нераспределённых), urgent_order (существующая аптека с узким окном).
Плавный диурнал c(h) — НЕ триггер (уже в CongestionTravel). События seeded → воспроизводимо.

env НЕ переписываем: DynamicVRPEnv — тонкий сабкласс, переопределяет _load, чтобы CongestionTravel
пережил внутренний reset() (rollout/greedy сбрасывают среду). residual = переоптимизация
ОСТАВШИХСЯ стопов от депо в момент события (свежий T_max/машину; НЕ mid-route continuation —
стандартный periodic re-optimization dynamic-VRP). Окна сдвигаются в базу времени события.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from logistics_rl_gnn.config import congestion as cg
from logistics_rl_gnn.config import instance as im
from logistics_rl_gnn.env.travel import CongestionTravel, Incident
from logistics_rl_gnn.env.vrp_env import VRPEnv

# ---------- drop-in среда с congestion-travel ----------


class DynamicVRPEnv(VRPEnv):
    """VRPEnv с инъекцией travel-модели через фабрику (переживает reset). Base env не тронут."""

    def __init__(self, *, travel_factory=None, **kw):
        self._travel_factory = travel_factory  # (env) -> TravelModel; None → free-flow базы
        super().__init__(**kw)

    def _load(self, inst) -> None:
        super()._load(inst)
        if getattr(self, "_travel_factory", None) is not None:
            self.travel = self._travel_factory(self)


def make_dynamic_env(instance, *, travel=None, **kw) -> DynamicVRPEnv:
    """Среда на фикс. residual-инстансе с готовой travel-моделью (или free-flow при travel=None).

    K/Q/день недели берём ИЗ ИНСТАНСА (meta; дефолтный инстанс несёт глобальные значения) —
    иначе сценарный флот тихо потерялся бы в def-time дефолтах VRPEnv. Явный kwarg сильнее.
    """
    k, q = im.fleet_of(instance)
    kw.setdefault("fleet_size", k)
    kw.setdefault("vehicle_cap", q)
    kw.setdefault("delivery_weekday", int((getattr(instance, "meta", None) or {}).get(
        "delivery_weekday", im.DELIVERY_WEEKDAY)))
    fac = (lambda env: travel) if travel is not None else None
    return DynamicVRPEnv(instance_fn=lambda s: instance, travel_factory=fac, **kw)


def congestion_for(instance, *, dow: int, offset_min: float = 0.0, incidents=None, flat_c=False):
    """CongestionTravel на free-flow t0 инстанса (мин) и его координатах."""
    t0_min = np.asarray(instance.time_matrix, dtype=float) / 60.0
    return CongestionTravel(
        t0_min,
        instance.coords,
        dow=dow,
        offset_min=offset_min,
        incidents=incidents,
        flat_c=flat_c,
    )


# ---------- состояние динамического эпизода ----------


@dataclass
class DynamicState:
    instance: object  # Instance (полный, до срезов)
    dow: int
    served: set = field(default_factory=set)  # instance-local индексы уже обслуженных стопов
    broken: int = 0  # число выбывших машин
    urgent: list = field(default_factory=list)  # [{idx, demand, delta_s}]
    incidents: list = field(default_factory=list)  # накопленные Incident
    now_min: float = 0.0

    def fleet(self, base_k: int) -> int:
        return max(1, base_k - self.broken)


def served_by(routes, instance, travel, now_min: float) -> set[int]:
    """Стопы, ЗАВЕРШённые (конец сервиса) до now_min при прохождении routes с travel. Timeline
    исполняемого плана → «частичное состояние» привязано к реальному дисп-плану, не к рандому."""
    win = np.asarray(instance.windows, dtype=float) / 60.0
    svc = np.asarray(instance.service, dtype=float) / 60.0
    done: set[int] = set()
    for route in routes:
        t = 0.0
        for a, b in zip(route[:-1], route[1:], strict=False):
            arrival = t + float(travel.time(a, b, t))
            if b == 0:
                t = arrival
                continue
            finish = max(arrival, win[b, 0]) + svc[b]
            if finish <= now_min:
                done.add(int(b))
            t = finish
    return done


def residual_instance(state: DynamicState):
    """Инстанс оставшейся задачи: депо + необслуженные + срочные (окна сдвинуты в базу события).

    Тег meta.dynamic=True, tag='simulated-on-real' — возмущение реального инстанса (запрет №5).
    """
    inst = state.instance
    n = len(inst.demand)
    now_s = state.now_min * 60.0
    pending = [i for i in range(1, n) if i not in state.served]
    for u in state.urgent:  # срочные включаем даже если обслужены (повторная доставка)
        if u["idx"] not in pending:
            pending.append(u["idx"])
    idx = [0] + sorted(set(pending))
    ridx = np.array(idx, dtype=int)

    win = inst.windows[ridx].astype(float).copy()  # сек от episode_start
    dem = inst.demand[ridx].copy()
    svc = inst.service[ridx].astype(float).copy()
    # сдвиг окон: residual t=0 = момент события → [e-now, l-now]сек, e≥0; депо = [0, horizon]
    win[1:, 0] = np.maximum(0.0, win[1:, 0] - now_s)
    win[1:, 1] = win[1:, 1] - now_s
    win[0] = [0.0, float(inst.horizon_s)]
    pos = {orig: k for k, orig in enumerate(idx)}
    for u in state.urgent:  # срочное: узкое окно [0, delta] (event-relative) + свежий спрос
        k = pos[u["idx"]]
        win[k] = [0.0, float(u["delta_s"])]
        dem[k] = int(u["demand"])

    return replace(
        inst,
        node_ids=[inst.node_ids[i] for i in idx],
        snapshot_stops=[inst.snapshot_stops[i] for i in idx],
        kinds=[inst.kinds[i] for i in idx],
        tw_source=[inst.tw_source[i] for i in idx],
        time_matrix=inst.time_matrix[np.ix_(ridx, ridx)],
        dist_matrix=inst.dist_matrix[np.ix_(ridx, ridx)],
        coords=inst.coords[ridx],
        windows=win,
        demand=dem,
        service=svc,
        meta={
            **inst.meta,
            "dynamic": True,
            "tag": cg.CALIBRATION_TAG,
            "now_min": state.now_min,
            "n_pending": len(pending),
            "broken": state.broken,
            # флот ОСТАТКА: выбывшие машины недоступны. Иначе среда, созданная от residual без
            # явного fleet_size, молча планировала бы на исходный K (сейчас все вызовы явные).
            "fleet_size": state.fleet(int(inst.meta.get("fleet_size", im.FLEET_SIZE))),
        },
    )


# ---------- события ----------


@dataclass
class TrafficEvent:
    at_min: float
    incident: Incident
    kind: str = "traffic"

    def apply(self, state: DynamicState) -> bool:
        state.incidents.append(self.incident)
        return True  # инцидент = триггер re-plan


@dataclass
class BreakdownEvent:
    at_min: float
    kind: str = "breakdown"

    def apply(self, state: DynamicState) -> bool:
        state.broken += 1  # машина выбыла; её стопы уже в пуле необслуженных (residual их заберёт)
        return True


@dataclass
class UrgentEvent:
    at_min: float
    order: dict
    kind: str = "urgent"

    def apply(self, state: DynamicState) -> bool:
        state.urgent.append(self.order)
        return True


@dataclass
class DiurnalTick:
    at_min: float
    kind: str = "diurnal"

    def apply(self, state: DynamicState) -> bool:
        return False  # плавный c(h) уже в travel — НЕ триггер (dec-0001 §4)


def event_stream(seed: int, instance, dow: int, n_events: int = 6) -> list:
    """Seeded поток событий на эпизод. Порядок по времени; смесь triggers + диурнал-тики."""
    rng = np.random.default_rng(seed + 7_000_003)  # декоррелируем от instance-seed
    n = len(instance.demand)
    horizon_min = instance.horizon_s / 60.0
    # события в АКТИВНОЙ фазе исполнения: 8 машин идут параллельно от t=0, план ~2ч → окно
    # [~20, ~130]мин даёт содержательные частичные состояния (позже почти всё обслужено).
    times = np.sort(rng.uniform(0.03 * horizon_min, 0.22 * horizon_min, size=n_events))
    kinds = ["traffic", "urgent", "breakdown", "diurnal", "traffic", "urgent"]
    events: list = []
    for i, at in enumerate(times):
        kind = kinds[i % len(kinds)]
        if kind == "traffic":
            center = tuple(instance.coords[int(rng.integers(1, n))])
            closure = bool(rng.random() < cg.INCIDENT_CLOSURE_PROB)
            mag = np.inf if closure else float(rng.uniform(*cg.INCIDENT_MAG_RANGE))
            dur = float(rng.uniform(*cg.INCIDENT_DUR_RANGE_MIN))
            inc = Incident(center, cg.INCIDENT_RADIUS_KM, mag, float(at), dur)
            events.append(TrafficEvent(float(at), inc))
        elif kind == "urgent":
            order = {
                "idx": int(rng.integers(1, n)),
                "demand": int(rng.integers(im.DEMAND_RANGE[0], im.DEMAND_RANGE[1] + 1)),
                "delta_s": float(rng.uniform(30.0, 75.0) * 60.0),  # узкое окно 30–75 мин
            }
            events.append(UrgentEvent(float(at), order))
        elif kind == "breakdown":
            events.append(BreakdownEvent(float(at)))
        else:
            events.append(DiurnalTick(float(at)))
    return events
