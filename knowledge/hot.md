<!-- AUTO-GEN START (refreshed by scripts/refresh-hot-cache.py) -->

# Hot Cache

**Auto-refreshed:** 2026-07-20 17:48:53 (every SessionStart)
**Branch:** `main`

## 🔀 Recent commits (top 5)

```
835b43c chore: initial scaffold + MDP spec (0001)
```

## 📋 Recent decisions

- `0001-mdp-spec.md` — 0001 — Динамический CVRPTW: доставка по аптекам Аугсбурга

## 📅 Recent daily logs

- `2026-07-20.md`

<!-- AUTO-GEN END (everything below preserved across refreshes) -->

# Hot Cache — curated

**Last update:** 2026-07-20 17:25 (правится руками / `/close`; секция выше — авто, маркер НЕ трогать)

## 🔥 What's Hot
- Постановка RL-задачи ЗАФИКСИРОВАНА → `decisions/0001-mdp-spec.md`: динамический CVRPTW,
  доставка по аптекам Аугсбурга на реальной OSM-сети; входы помечены `[REAL]`/`[ASSUMED]`.
- Репо инициализирован: первый коммит `835b43c` (scaffold + spec), есть `.gitignore` (запрет №1).
- Рантайм-зависимости (torch/torch-geometric/gymnasium/ortools) всё ещё НЕ пинятся — до фиксации железа (CPU vs ROCm).

## ⏭️ Next
- Запинить рантайм-зависимости под выбранное железо (CPU vs ROCm) — теперь постановка есть.
- Начать код в `src/`: сбор OSM Augsburg + депо/аптеки (Phase 2, verify-гейт = % покрытия maxspeed).
- Запустить `/init` (code-discovery) когда появится код в `src/`.

## 🚧 Blockers
- нет
