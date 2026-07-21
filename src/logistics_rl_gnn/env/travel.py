"""Модель времени в пути. Интерфейс TravelModel + FreeFlowTravel (Phase 3) +
CongestionTravel (Phase 7).

CongestionTravel — тот же интерфейс, но at_minute влияет на время: диурнальный множитель
c(dow,h) × вклад инцидентов. Drop-in: подменяется через env.travel (в dynamic-env
переживает reset). Единица — минуты.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from logistics_rl_gnn.config import congestion as cg
from logistics_rl_gnn.config.instance import DISPATCH_OPEN_H


class TravelModel:
    """Интерфейс: время в пути i→j (мин) при отправлении в at_minute (мин от старта)."""

    def time(self, i: int, j: int, at_minute: float) -> float:
        raise NotImplementedError


class FreeFlowTravel(TravelModel):
    """Free-flow: время не зависит от at_minute (Phase 3) → time_m[i, j]."""

    def __init__(self, time_m_min: np.ndarray):
        self.time_m = np.asarray(time_m_min, dtype=float)

    def time(self, i: int, j: int, at_minute: float = 0.0) -> float:
        return float(self.time_m[i, j])


def _km(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Приближённое расстояние lon/lat → км (локальная плоская аппроксимация, Augsburg ~48.4°).

    ponytail: flat-earth ок для зоны радиусом 1–2 км; для города точнее не нужно.
    """
    lat = math.radians((a[1] + b[1]) / 2.0)
    dlon = (a[0] - b[0]) * 111.320 * math.cos(lat)
    dlat = (a[1] - b[1]) * 110.574
    return math.hypot(dlon, dlat)


@dataclass
class Incident:
    """Локальный инцидент: зона (центр-коорд + радиус) × окно времени × магнитуда.

    Геометрия по координатам (не по индексу узла) → переживает срез инстанса при re-plan.
    magnitude=inf → закрытие (ребро удаляется). Затухание линейное к концу длительности.
    Время в АБСОЛЮТНЫХ минутах от dispatch-open (как at_minute+offset в CongestionTravel).
    """

    center: tuple[float, float]  # (lon, lat)
    radius_km: float
    magnitude: float  # δ; inf = закрытие ребра
    t_start_min: float
    duration_min: float

    def contrib(self, coord_i, coord_j, abs_min: float) -> float:
        """Вклад в (1+Σ) на ребре i→j в момент abs_min: δ·decay, inf при закрытии, иначе 0."""
        if not (self.t_start_min <= abs_min <= self.t_start_min + self.duration_min):
            return 0.0
        in_zone = (
            _km(self.center, tuple(coord_i)) <= self.radius_km
            or _km(self.center, tuple(coord_j)) <= self.radius_km
        )
        if not in_zone:
            return 0.0
        if math.isinf(self.magnitude):
            return math.inf
        decay = 1.0 - (abs_min - self.t_start_min) / self.duration_min  # 1→0 линейно
        return self.magnitude * decay


class CongestionTravel(TravelModel):
    """t(i,j,at) = t0[i,j]·c(dow, h) · (1 + Σ_k I_k), h = DISPATCH_OPEN_H + (offset+at)/60.

    offset_min — сдвиг времени старта тура (для residual re-plan: событие в середине дня →
    туры начинаются в реальный час). incidents — список Incident (геометрия по coords).
    Паритет: c≡1 (передан flat_c=True) И нет инцидентов ⇒ ровно FreeFlow (t0[i,j]).
    """

    def __init__(
        self,
        t0_min: np.ndarray,
        coords: np.ndarray,
        *,
        dow: int,
        offset_min: float = 0.0,
        incidents: list[Incident] | None = None,
        flat_c: bool = False,
    ):
        self.time_m = np.asarray(t0_min, dtype=float)
        self.coords = np.asarray(coords, dtype=float)
        self.dow = int(dow)
        self.offset_min = float(offset_min)
        self.incidents = list(incidents or [])
        self.flat_c = bool(flat_c)  # тест-паритет: отключить диурнал (c≡1)

    def time(self, i: int, j: int, at_minute: float = 0.0) -> float:
        t0 = float(self.time_m[i, j])
        abs_min = self.offset_min + at_minute
        c = 1.0 if self.flat_c else cg.congestion(self.dow, DISPATCH_OPEN_H + abs_min / 60.0)
        inc = 0.0
        for k in self.incidents:
            d = k.contrib(self.coords[i], self.coords[j], abs_min)
            if math.isinf(d):
                return math.inf  # закрытие ребра → недостижимо (маскируется в env)
            inc += d
        return t0 * c * (1.0 + inc)
