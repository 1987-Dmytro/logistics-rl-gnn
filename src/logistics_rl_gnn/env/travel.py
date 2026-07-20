"""Модель времени в пути. Интерфейс TravelModel + FreeFlowTravel (Phase 3).

Phase 7 подменит FreeFlow на CongestionTravel с ТЕМ ЖЕ интерфейсом (at_minute начнёт
влиять на время). Единица — минуты.
"""

from __future__ import annotations

import numpy as np


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
