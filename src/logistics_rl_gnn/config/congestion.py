"""Congestion-профиль c(dow,h) + параметры инцидентов (Phase 7, dec-0001 §4).

t(e,τ) = t0(e)·c(dow,h(τ))·(1 + Σ_k I_k(e,τ)). Класс дороги в OD-матрице снапшота
недоступен (сборка `with_graph=False`) → ось class схлопнута в ОДИН городской профиль;
форма калибрована по кривой TomTom Traffic Index (Augsburg): утренний (08) и вечерний (17)
пики, дневной провал. Магнитуды ASSUMED → тег simulated-on-real. Всё детерминировано (seed).
"""

from __future__ import annotations

CALIBRATION_TAG = "simulated-on-real"  # форма TomTom Augsburg реальна, магнитуды синтетические

# c по часу суток (будни), 1.0 = free-flow. Форма TomTom: два пика (08, 17), провал днём.
# Вне диапазона — экстраполяция крайними значениями (clamp в congestion()).
_HOURLY_WEEKDAY: dict[int, float] = {
    6: 1.10, 7: 1.22, 8: 1.30, 9: 1.20, 10: 1.10, 11: 1.08,
    12: 1.12, 13: 1.10, 14: 1.08, 15: 1.12, 16: 1.22, 17: 1.32,
    18: 1.18, 19: 1.10, 20: 1.05,
}  # fmt: skip
# dow-фактор (Mon=0..Sun=6): будни — полная амплитуда congestion, выходные легче.
_DOW_FACTOR: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0, 0.7, 0.5)

_HRS = sorted(_HOURLY_WEEKDAY)


def congestion(dow: int, hour: float) -> float:
    """Множитель c(dow,h) ≥ 1. hour клампится к диапазону таблицы (экстраполяция краями).

    dow-фактор гасит АМПЛИТУДУ (c−1), не сам c → выходные ближе к free-flow, будни — полный пик.
    """
    h = int(min(max(hour, _HRS[0]), _HRS[-1]))
    base = _HOURLY_WEEKDAY[h]
    f = _DOW_FACTOR[int(dow) % 7]
    return 1.0 + (base - 1.0) * f


# --- инциденты I_k (dec-0001 §4): пуассон-спавн, ставка растёт в пики ---
INCIDENT_BASE_RATE_PER_H = 0.6  # λ базовая (событий/час); масштабируется на (c−1) в час пика
INCIDENT_MAG_RANGE = (0.4, 1.8)  # δ: вклад (1+δ·decay) на зоне
INCIDENT_DUR_RANGE_MIN = (15.0, 60.0)  # длительность инцидента, мин (линейное затухание)
INCIDENT_RADIUS_KM = 1.2  # радиус зоны от центра-узла, км
INCIDENT_CLOSURE_PROB = 0.08  # δ=∞ (закрытие ребра → удаление из графа)
