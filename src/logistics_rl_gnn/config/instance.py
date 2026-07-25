"""Генератор инстанса CVRPTW из снапшота: реальные окна аптек из OSM opening_hours.

Окна: REAL из тега (клип к диспетч-окну депо), ASSUMED-синтетика при отсутствии/
непарсибельности, EXCLUDED — аптека закрыта в день доставки (стоп исключается).
Всё детерминировано: seed + delivery_weekday + снапшот. Дата якорится фиксированно
(НЕ now()) → воспроизводимость (dec-0001 запрет №4).

opening_hours_py устанавливается как пакет `opening_hours_py`, а импортируется как
модуль `opening_hours` (проверено на установленной версии).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
from opening_hours import OpeningHours

# --- конфиг инстанса (все [ASSUMED, config], dec-0001 §2) ---
DELIVERY_WEEKDAY = 1  # вторник (Mon=0)
DISPATCH_OPEN_H = 8  # депо: окно диспетчеризации (часы суток)
DISPATCH_CLOSE_H = 18
REFERENCE_DATE = date(2024, 1, 1)  # якорь → воспроизводимая будняя дата без PH
DEMAND_RANGE = (3, 12)  # боксов/стоп
SERVICE_MIN = 4.0  # мин обслуживания/стоп
DEFAULT_SEED = 0
REACH_MARGIN_S = 60.0  # запас достижимости при клипе синтетич. окна (под округление OR-Tools)

# --- флот / стоимость / штрафы (dec-0001 §2-3, [ASSUMED, config]; калибруется Phase 4) ---
FLEET_SIZE = 8  # K машин (калибровка: 6→8, чтобы K*Q покрывал спрос)
VEHICLE_CAP = 80  # Q боксов (калибровка: 60→80)
T_MAX_MIN = 240.0  # макс. тур, мин (4ч) — обоснование в dec-0001 §6
COST_PER_VEHICLE = 50.0  # c_f, € за задействованную машину
COST_PER_KM = 0.5  # c_d, €/км
COST_PER_HOUR = 20.0  # c_t, €/ч
PENALTY_LATE_PER_MIN = 2.0  # штраф опоздания, €/мин
PENALTY_UNSERVED = 200.0  # € за необслуженную аптеку
PENALTY_INVALID = 1000.0  # € за невалидное действие (терминал)
DENSE_REWARD = False  # dense-shaping по шагам (по умолчанию off)

SNAP_ROOT = Path("data/snapshots")


@dataclass
class Instance:
    """Инстанс CVRPTW. Все векторы длины k (включённые стопы, депо = индекс 0)."""

    node_ids: list[int]  # OSM node id на стоп
    snapshot_stops: list[int]  # индексы стопов снапшота, вошедшие в инстанс
    kinds: list[str]  # depot | pharmacy
    time_matrix: np.ndarray  # (k,k) под-матрица включённых стопов, сек
    dist_matrix: np.ndarray  # (k,k), м
    coords: np.ndarray  # (k,2) [x=lon, y=lat] на стоп
    windows: np.ndarray  # (k,2) [e_i,l_i] сек от start_datetime; депо=[0,horizon]
    demand: np.ndarray  # (k,) боксов; депо=0
    service: np.ndarray  # (k,) сек; депо=0
    tw_source: list[str]  # DEPOT | REAL | ASSUMED на стоп
    excluded_stops: list[int]  # индексы стопов снапшота, закрытых в день доставки
    start_datetime: datetime
    horizon_s: int
    meta: dict = field(default_factory=dict)


def _day_bounds(delivery_weekday: int, dispatch_open_h=None) -> tuple:
    """(day_start, day_end, episode_start, horizon_s) для дня доставки.

    dispatch_open_h — час открытия диспетч-окна (сценарий); None → DISPATCH_OPEN_H.
    """
    open_h = DISPATCH_OPEN_H if dispatch_open_h is None else float(dispatch_open_h)
    monday = REFERENCE_DATE - timedelta(days=REFERENCE_DATE.weekday())
    day = monday + timedelta(days=delivery_weekday)
    day_start = datetime(day.year, day.month, day.day)
    day_end = day_start + timedelta(days=1)
    episode_start = day_start + timedelta(hours=open_h)
    horizon_s = int((DISPATCH_CLOSE_H - open_h) * 3600)
    return day_start, day_end, episode_start, horizon_s


def fleet_of(instance) -> tuple[int, float]:
    """(K, Q) инстанса: оверрайд сценария из meta или глобальные дефолты. Единая точка —
    иначе сценарный флот тихо разъедется с дефолтом 8×80 в env/polish/OR-Tools."""
    m = getattr(instance, "meta", None) or {}
    return int(m.get("fleet_size", FLEET_SIZE)), float(m.get("vehicle_cap", VEHICLE_CAP))


def dispatch_offset_min(instance) -> float:
    """Сдвиг congestion-часов, мин: диспетч-окно сценария стартует не в DISPATCH_OPEN_H.

    CongestionTravel мапит abs_min → час как DISPATCH_OPEN_H + abs_min/60 → старт в 09:00
    требует offset_min=60, иначе профиль c(dow,h) сдвинут на час (тихая ошибка)."""
    m = getattr(instance, "meta", None) or {}
    return (float(m.get("dispatch_open_h", DISPATCH_OPEN_H)) - DISPATCH_OPEN_H) * 60.0


def real_day_window(oh_string, day_start: datetime, day_end: datetime):
    """(status, first_open, last_close) для дня [day_start, day_end).

    status: REAL — есть открытые интервалы; CLOSED — валидно, но закрыта (исключить);
            INVALID — пусто/не распарсилось/только unknown → синтетический фолбэк.
    """
    if oh_string is None:
        return "INVALID", None, None
    s = str(oh_string).strip()
    if not s or s.lower() == "nan":
        return "INVALID", None, None
    try:
        ivs = list(OpeningHours(s).intervals(day_start, day_end))
    except Exception:
        return "INVALID", None, None
    states = [str(iv[2]).lower() for iv in ivs]
    if any(st == "open" for st in states):
        opens = [(iv[0], iv[1]) for iv in ivs if str(iv[2]).lower() == "open"]
        return "REAL", min(a for a, _ in opens), max(b for _, b in opens)
    if any(st == "unknown" for st in states):
        return "INVALID", None, None  # неизвестно → фолбэк, НЕ исключаем
    return "CLOSED", None, None


def _synthetic_window(
    rng: np.random.Generator, horizon_s: int, e_max: float | None = None
) -> tuple[float, float]:
    """Сид-синтетическое окно [e,l] сек, e<l, в пределах горизонта (ASSUMED).

    e_max — верхний предел раннего окна по достижимости в T_max (клип e вниз, чтобы
    ASSUMED-плейсхолдер не сделал реальную аптеку структурно неохватной); None → без клипа.
    Клип пост-розыгрыша → rng-поток не меняется (feasible-окна остаются как были).
    """
    e = float(rng.uniform(0.0, 0.4 * horizon_s))
    width = float(rng.uniform(0.3 * horizon_s, 0.6 * horizon_s))
    if e_max is not None:
        e = min(e, max(0.0, e_max))  # не позже, чем успеть подождать/обслужить/вернуться
    return e, min(float(horizon_s), e + width)


def stop_window(oh_string, delivery_weekday: int, rng: np.random.Generator, e_max=None,
                dispatch_open_h=None):
    """(source, e_s, l_s) для одного стопа. source: REAL | ASSUMED | EXCLUDED.

    e_s,l_s — сек от episode_start (депо open); EXCLUDED → (None, None). rng тратится
    только на ASSUMED-ветку. e_max клипит ТОЛЬКО синтетику (реальные окна не трогаем —
    запрет №5); реальные окна снапшота достижимы (проверено стражем).
    """
    day_start, day_end, episode_start, horizon_s = _day_bounds(delivery_weekday, dispatch_open_h)
    status, fo, lc = real_day_window(oh_string, day_start, day_end)
    if status == "CLOSED":
        return "EXCLUDED", None, None
    if status == "REAL":
        e_s = max(0.0, (fo - episode_start).total_seconds())
        l_s = min(float(horizon_s), (lc - episode_start).total_seconds())
        if l_s > e_s:  # открыта в пределах диспетч-окна
            return "REAL", e_s, l_s
        # открыта, но вне диспетч-окна → синтетика
    e_s, l_s = _synthetic_window(rng, horizon_s, e_max)
    return "ASSUMED", e_s, l_s


def _latest_snapshot_dir():
    if not SNAP_ROOT.is_dir():
        return None
    dirs = sorted(p for p in SNAP_ROOT.glob("augsburg_*") if (p / "meta.json").is_file())
    return dirs[-1] if dirs else None


def _check_feasibility(demand, service, time_m, win_e, fleet_size=None, vehicle_cap=None) -> None:
    """Страж инстанса (пойманный баг Phase 3 → постоянная проверка).

    raise если суммарный спрос > K*Q (инфеасибл); warn если утилизация > 0.9;
    assert достижимости каждой аптеки С УЧЁТОМ ОЖИДАНИЯ до e_i: машина стартует в t=0,
    ждёт до e_i, обслуживает, возвращается — max(t[0,i], e_i)+service+t[i,0] <= T_max.
    demand/service/win_e — списки (депо = индекс 0); time_m — сек. fleet_size/vehicle_cap
    (сценарий) — None → ГЛОБАЛЬНЫЕ константы, читаются в теле (monkeypatch-тест жив).
    """
    k = FLEET_SIZE if fleet_size is None else int(fleet_size)
    q = VEHICLE_CAP if vehicle_cap is None else float(vehicle_cap)
    cap = k * q
    total = float(sum(demand))
    if total > cap:
        raise ValueError(
            f"инфеасибл: спрос {total:.0f} > K*Q={k}×{q:.0f}={cap:.0f} (мало машин/вместимости)"
        )
    heavy = [i for i in range(1, len(demand)) if float(demand[i]) > q]
    if heavy:  # оверрайд спроса больше вместимости машины → стоп неотвозим НИКОГДА (тихий unserved)
        raise ValueError(
            f"инфеасибл: спрос стопа #{heavy[0]} = {float(demand[heavy[0]]):.0f} > Q={q:.0f} "
            f"(одна машина такой стоп не увезёт)"
        )
    util = total / cap
    if util > 0.9:
        warnings.warn(
            f"высокая утилизация {util:.0%} (спрос {total:.0f}/{cap}) — риск unserved",
            stacklevel=2,
        )
    t_max_s = T_MAX_MIN * 60.0
    for i in range(1, len(demand)):  # 0 = депо
        reach = max(float(time_m[0, i]), float(win_e[i])) + float(service[i]) + float(time_m[i, 0])
        assert reach <= t_max_s + 1e-6, (
            f"аптека #{i} недостижима в T_max с ожиданием до e: "
            f"{reach / 60:.1f} > {T_MAX_MIN} мин (e={win_e[i] / 60:.1f})"
        )


def generate_instance(
    snapshot_dir=None,
    *,
    seed: int = DEFAULT_SEED,
    delivery_weekday: int = DELIVERY_WEEKDAY,
    include_stops=None,
    demand_overrides=None,
    fleet_size=None,
    vehicle_cap=None,
    dispatch_open_h=None,
) -> Instance:
    """Инстанс CVRPTW из снапшота с реальными окнами аптек на день доставки.

    Сценарные оверрайды (config.scenario; ВСЕ None → дефолтный путь байт-в-байт):
    include_stops — подмножество snapshot-стопов аптек (депо всегда); demand_overrides —
    {snapshot_stop: боксы} поверх розыгрыша (rng-поток НЕ меняется); fleet_size/vehicle_cap —
    K/Q сценария (в meta → `fleet_of`, оттуда env/polish/OR-Tools); dispatch_open_h — час
    открытия диспетч-окна.
    """
    from logistics_rl_gnn.data import snapshot as snap_mod

    d = snapshot_dir or _latest_snapshot_dir()
    if d is None:
        raise FileNotFoundError("нет снапшота — сначала `python scripts/build_snapshot.py`")
    snap = snap_mod.load_snapshot(d, with_graph=False)
    _, _, episode_start, horizon_s = _day_bounds(delivery_weekday, dispatch_open_h)
    want = None if include_stops is None else {int(s) for s in include_stops}
    over = {int(k): int(v) for k, v in (demand_overrides or {}).items()}
    rng = np.random.default_rng(seed)
    depot_stop = int(snap.nodes.loc[snap.nodes.kind == "depot", "stop"].iloc[0])
    t_max_s = T_MAX_MIN * 60.0

    included: list[int] = []
    excluded: list[int] = []
    node_ids: list[int] = []
    coords: list[list[float]] = []
    kinds: list[str] = []
    tw_source: list[str] = []
    win_e: list[float] = []
    win_l: list[float] = []
    demand: list[int] = []
    service: list[float] = []

    for row in snap.nodes.itertuples(index=False):
        stop_i = int(row.stop)
        if row.kind == "depot":
            included.append(stop_i)
            node_ids.append(int(row.node_id))
            coords.append([float(row.x), float(row.y)])
            kinds.append("depot")
            tw_source.append("DEPOT")
            win_e.append(0.0)
            win_l.append(float(horizon_s))
            demand.append(0)
            service.append(0.0)
            continue
        if want is not None and stop_i not in want:
            continue  # сценарное подмножество: стоп не запрошен (rng не тратим)
        # бюджет раннего окна: успеть подождать до e, обслужить и вернуться в депо в T_max
        ret_s = float(snap.time_matrix[stop_i, depot_stop])
        e_max = t_max_s - SERVICE_MIN * 60.0 - ret_s - REACH_MARGIN_S
        src, e_s, l_s = stop_window(row.opening_hours, delivery_weekday, rng, e_max=e_max,
                                    dispatch_open_h=dispatch_open_h)
        if src == "EXCLUDED":
            excluded.append(stop_i)
            continue
        included.append(stop_i)
        node_ids.append(int(row.node_id))
        coords.append([float(row.x), float(row.y)])
        kinds.append("pharmacy")
        tw_source.append(src)
        win_e.append(e_s)
        win_l.append(l_s)
        drawn = int(rng.integers(DEMAND_RANGE[0], DEMAND_RANGE[1] + 1))  # розыгрыш ВСЕГДА (rng)
        demand.append(over.get(stop_i, drawn))
        service.append(SERVICE_MIN * 60.0)

    idx = np.array(included)
    time_m = snap.time_matrix[np.ix_(idx, idx)]
    # страж (тот же, что для дефолта): обслуживаем ≤ K*Q и достижим в T_max
    _check_feasibility(demand, service, time_m, win_e, fleet_size, vehicle_cap)
    return Instance(
        node_ids=node_ids,
        snapshot_stops=included,
        kinds=kinds,
        time_matrix=time_m,
        dist_matrix=snap.dist_matrix[np.ix_(idx, idx)],
        coords=np.array(coords, dtype=float),
        windows=np.column_stack([win_e, win_l]),
        demand=np.array(demand),
        service=np.array(service),
        tw_source=tw_source,
        excluded_stops=excluded,
        start_datetime=episode_start,
        horizon_s=horizon_s,
        meta={
            "seed": seed,
            "delivery_weekday": delivery_weekday,
            "n_real": tw_source.count("REAL"),
            "n_fallback": tw_source.count("ASSUMED"),
            "n_excluded": len(excluded),
            "n_included_stops": len(included),
            # эффективные K/Q/час-открытия (== дефолты, если сценарий не задал) → fleet_of
            "fleet_size": FLEET_SIZE if fleet_size is None else int(fleet_size),
            "vehicle_cap": VEHICLE_CAP if vehicle_cap is None else float(vehicle_cap),
            "dispatch_open_h": DISPATCH_OPEN_H if dispatch_open_h is None else float(
                dispatch_open_h
            ),
        },
    )


if __name__ == "__main__":
    inst = generate_instance()
    m = inst.meta
    print(
        f"instance: включено стопов={m['n_included_stops']} (депо+аптеки); "
        f"окна REAL={m['n_real']} ASSUMED={m['n_fallback']} excluded={m['n_excluded']}; "
        f"weekday={m['delivery_weekday']} seed={m['seed']} горизонт={inst.horizon_s}с"
    )
