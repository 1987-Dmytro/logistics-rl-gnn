"""Генератор инстанса CVRPTW из снапшота: реальные окна аптек из OSM opening_hours.

Окна: REAL из тега (клип к диспетч-окну депо), ASSUMED-синтетика при отсутствии/
непарсибельности, EXCLUDED — аптека закрыта в день доставки (стоп исключается).
Всё детерминировано: seed + delivery_weekday + снапшот. Дата якорится фиксированно
(НЕ now()) → воспроизводимость (dec-0001 запрет №4).

opening_hours_py устанавливается как пакет `opening_hours_py`, а импортируется как
модуль `opening_hours` (проверено на установленной версии).
"""

from __future__ import annotations

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

SNAP_ROOT = Path("data/snapshots")


@dataclass
class Instance:
    """Инстанс CVRPTW. Все векторы длины k (включённые стопы, депо = индекс 0)."""

    node_ids: list[int]  # OSM node id на стоп
    snapshot_stops: list[int]  # индексы стопов снапшота, вошедшие в инстанс
    kinds: list[str]  # depot | pharmacy
    time_matrix: np.ndarray  # (k,k) под-матрица включённых стопов, сек
    dist_matrix: np.ndarray  # (k,k), м
    windows: np.ndarray  # (k,2) [e_i,l_i] сек от start_datetime; депо=[0,horizon]
    demand: np.ndarray  # (k,) боксов; депо=0
    service: np.ndarray  # (k,) сек; депо=0
    tw_source: list[str]  # DEPOT | REAL | ASSUMED на стоп
    excluded_stops: list[int]  # индексы стопов снапшота, закрытых в день доставки
    start_datetime: datetime
    horizon_s: int
    meta: dict = field(default_factory=dict)


def _day_bounds(delivery_weekday: int) -> tuple[datetime, datetime, datetime, int]:
    """(day_start, day_end, episode_start, horizon_s) для дня доставки."""
    monday = REFERENCE_DATE - timedelta(days=REFERENCE_DATE.weekday())
    day = monday + timedelta(days=delivery_weekday)
    day_start = datetime(day.year, day.month, day.day)
    day_end = day_start + timedelta(days=1)
    episode_start = day_start + timedelta(hours=DISPATCH_OPEN_H)
    horizon_s = (DISPATCH_CLOSE_H - DISPATCH_OPEN_H) * 3600
    return day_start, day_end, episode_start, horizon_s


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


def _synthetic_window(rng: np.random.Generator, horizon_s: int) -> tuple[float, float]:
    """Сид-синтетическое окно [e,l] сек, e<l, в пределах горизонта (ASSUMED)."""
    e = float(rng.uniform(0.0, 0.4 * horizon_s))
    width = float(rng.uniform(0.3 * horizon_s, 0.6 * horizon_s))
    return e, min(float(horizon_s), e + width)


def stop_window(oh_string, delivery_weekday: int, rng: np.random.Generator):
    """(source, e_s, l_s) для одного стопа. source: REAL | ASSUMED | EXCLUDED.

    e_s,l_s — сек от episode_start (депо open); EXCLUDED → (None, None). rng тратится
    только на ASSUMED-ветку.
    """
    day_start, day_end, episode_start, horizon_s = _day_bounds(delivery_weekday)
    status, fo, lc = real_day_window(oh_string, day_start, day_end)
    if status == "CLOSED":
        return "EXCLUDED", None, None
    if status == "REAL":
        e_s = max(0.0, (fo - episode_start).total_seconds())
        l_s = min(float(horizon_s), (lc - episode_start).total_seconds())
        if l_s > e_s:  # открыта в пределах диспетч-окна
            return "REAL", e_s, l_s
        # открыта, но вне диспетч-окна → синтетика
    e_s, l_s = _synthetic_window(rng, horizon_s)
    return "ASSUMED", e_s, l_s


def _latest_snapshot_dir():
    if not SNAP_ROOT.is_dir():
        return None
    dirs = sorted(p for p in SNAP_ROOT.glob("augsburg_*") if (p / "meta.json").is_file())
    return dirs[-1] if dirs else None


def generate_instance(
    snapshot_dir=None, *, seed: int = DEFAULT_SEED, delivery_weekday: int = DELIVERY_WEEKDAY
) -> Instance:
    """Инстанс CVRPTW из снапшота с реальными окнами аптек на день доставки."""
    from logistics_rl_gnn.data import snapshot as snap_mod

    d = snapshot_dir or _latest_snapshot_dir()
    if d is None:
        raise FileNotFoundError("нет снапшота — сначала `python scripts/build_snapshot.py`")
    snap = snap_mod.load_snapshot(d, with_graph=False)
    _, _, episode_start, horizon_s = _day_bounds(delivery_weekday)
    rng = np.random.default_rng(seed)

    included: list[int] = []
    excluded: list[int] = []
    node_ids: list[int] = []
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
            kinds.append("depot")
            tw_source.append("DEPOT")
            win_e.append(0.0)
            win_l.append(float(horizon_s))
            demand.append(0)
            service.append(0.0)
            continue
        src, e_s, l_s = stop_window(row.opening_hours, delivery_weekday, rng)
        if src == "EXCLUDED":
            excluded.append(stop_i)
            continue
        included.append(stop_i)
        node_ids.append(int(row.node_id))
        kinds.append("pharmacy")
        tw_source.append(src)
        win_e.append(e_s)
        win_l.append(l_s)
        demand.append(int(rng.integers(DEMAND_RANGE[0], DEMAND_RANGE[1] + 1)))
        service.append(SERVICE_MIN * 60.0)

    idx = np.array(included)
    return Instance(
        node_ids=node_ids,
        snapshot_stops=included,
        kinds=kinds,
        time_matrix=snap.time_matrix[np.ix_(idx, idx)],
        dist_matrix=snap.dist_matrix[np.ix_(idx, idx)],
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
