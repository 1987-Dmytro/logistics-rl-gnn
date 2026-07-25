"""Phase 2 data config: place, tags, depot, German speed-fallback table.

All data is real OSM ([REAL] per dec-0001). Nothing synthetic here.
"""

# Place and objects.
PLACE = "Augsburg, Germany"
PHARMACY_TAGS = {"amenity": "pharmacy"}
DEPOT_ADDR = "Benzstraße 10, 86391 Stadtbergen"  # PHOENIX VZ

# German speed fallback (km/h) for edges without a maxspeed tag.
# dec-0001 §4: primary/secondary 50, residential/Tempo-30 → 30, living_street 7.
GERMAN_HWY_SPEEDS = {
    "living_street": 7,
    "residential": 30,
    "primary": 50,
    "secondary": 50,
}
# Types outside the table and without maxspeed → German urban default.
FALLBACK_SPEED_KMH = 50

# Sanity gate on the pharmacy count (dec-0001 eval plan).
PHARMACY_COUNT_RANGE = (40, 120)
