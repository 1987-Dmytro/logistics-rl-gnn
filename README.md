# logistics-rl-gnn

Динамическая маршрутизация доставки (CVRPTW) через Graph Neural Networks + Reinforcement Learning.

## Миссия

GNN+RL-агент динамической маршрутизации доставки медикаментов по всем аптекам Аугсбурга
на реальной OSM-сети, с онлайн-перестроением при пробках / поломках / срочных заказах.
Цель — воспроизводимый кейс со снижением времени и пробега против OR-Tools.

Полная постановка (MDP, congestion-модель, eval-план):
[`knowledge/decisions/0001-mdp-spec.md`](knowledge/decisions/0001-mdp-spec.md).

## Dependencies

Рантайм-зависимости **не пиннятся** до фиксации целевого железа (CPU vs ROCm).
Целевое железо — CPU-only. Пиннинг делается по фазам, по мере появления кода:

| Фаза | Пакеты (план) |
|------|---------------|
| Phase 2 · data | `osmnx`, `networkx`, `geopandas`, `shapely` |
| Phase 3 · env | `gymnasium`, `numpy` |
| Phase 4 · baselines | `ortools` |
| Phase 5 · models | `torch`, `torch-geometric` (CPU-сборка) |
| Phase 6 · train | `+ wandb` (опц.) |

### CPU-установка torch / PyG

```bash
# PyTorch — CPU-only wheels
pip install torch --index-url https://download.pytorch.org/whl/cpu

# PyG core (чистый Python)
pip install torch-geometric

# Опциональные C++-расширения PyG (pyg-lib/scatter/sparse) — из PyG-индекса под версию torch:
# pip install pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-${TORCH_VERSION}+cpu.html
```

## Dev

```bash
just install   # pip install -e ".[dev]"
just check     # ruff check . + pytest -q
```

Воспроизводимость обязательна: любая метрика — только с зафиксированным seed, сохранённым
конфигом и снапшотом данных (см. запреты в `CLAUDE.md`).
