<!-- AUTO-GEN START (refreshed by scripts/refresh-hot-cache.py) -->

# Hot Cache

**Auto-refreshed:** 2026-07-20 13:17:54 (every SessionStart)
**Branch:** `main`

## 📅 Recent daily logs

- `2026-07-20.md`

<!-- AUTO-GEN END (everything below preserved across refreshes) -->

# Hot Cache — curated

**Last update:** 2026-07-20 13:16 (правится руками / `/close`; секция выше — авто, маркер НЕ трогать)

## 🔥 What's Hot
- Проект только заскаффолжен (второй мозг поставлен). Постановка RL-задачи (CVRPTW) ещё не зафиксирована.
- Рантайм-зависимости (torch/torch-geometric/gymnasium/ortools) НЕ пинятся до фиксации задачи и железа (CPU vs ROCm).
- CLAUDE.md: dev-команды заполнены (`pip install -e ".[dev]"`, pytest, ruff); код-карта — later, когда будет `src/`.

## ⏭️ Next
- Зафиксировать постановку RL-задачи (state/action/reward, CVRPTW-инстанс).
- Затем — запинить рантайм-зависимости под выбранное железо.
- Запустить `/init` (code-discovery) когда появится код в `src/`.

## 🚧 Blockers
- нет
