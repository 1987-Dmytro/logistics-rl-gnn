"""Конфиг данных Phase 2: место, теги, депо, немецкая таблица фолбэка скоростей.

Все данные — реальный OSM ([REAL] по dec-0001). Синтетики здесь нет.
"""

# Место и объекты.
PLACE = "Augsburg, Germany"
PHARMACY_TAGS = {"amenity": "pharmacy"}
DEPOT_ADDR = "Benzstraße 10, 86391 Stadtbergen"  # PHOENIX VZ

# Немецкий фолбэк скоростей (км/ч) для рёбер без тега maxspeed.
# dec-0001 §4: primary/secondary 50, residential/Tempo-30 → 30, living_street 7.
GERMAN_HWY_SPEEDS = {
    "living_street": 7,
    "residential": 30,
    "primary": 50,
    "secondary": 50,
}
# Типы вне таблицы и без maxspeed → немецкий городской дефолт.
FALLBACK_SPEED_KMH = 50

# Санити-гейт числа аптек (dec-0001 eval-план).
PHARMACY_COUNT_RANGE = (40, 120)
