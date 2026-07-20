<!-- AUTO-GEN START (refreshed by scripts/refresh-hot-cache.py) -->

# Hot Cache

**Auto-refreshed:** 2026-07-20 19:20:15 (every SessionStart)
**Branch:** `main`

## 🔀 Recent commits (top 5)

```
d7498df feat: Phase 2 — OSMnx-пайплайн Аугсбурга (data)
e711c37 feat: Phase 1 — скелет репо + verifier-петля
835b43c chore: initial scaffold + MDP spec (0001)
```

## 📋 Recent decisions

- `0001-mdp-spec.md` — 0001 — Динамический CVRPTW: доставка по аптекам Аугсбурга

## 📅 Recent daily logs

- `2026-07-20.md`

<!-- AUTO-GEN END (everything below preserved across refreshes) -->

# Hot Cache — curated

**Last update:** 2026-07-20 19:15 (правится руками / `/close`; секция выше — авто, маркер НЕ трогать)

## 🔥 What's Hot
- Phase 2 ЗАКРЫТА (коммит `d7498df`): OSMnx-пайплайн Аугсбурга. Снапшот строится
  `python scripts/build_snapshot.py` → `data/snapshots/augsburg_<YYYYMMDD>/` (вне git).
  Реальные числа: 4819 узлов, maxspeed-покрытие 89.7%, 62 аптеки; матрицы 63×63 ПО СТОПАМ.
- Ключевое в матрицах: индексация ПО СТОПАМ (депо + каждая аптека = строка), НЕ по уникальным
  узлам; со-узловые аптеки → Δ=0, но отдельные строки. dim = 1 + n_pharmacies.
- Депы: группа `[data]` (osmnx 2.1/CPU). `.gitignore` анкорит `/data/` и `/cache/` — НЕ вернуть
  к `data/` (тихо игнорит пакет кода `src/.../data/`).

## ⏭️ Next
- Phase 3 (env): `DynamicVRPEnv` (MDP dec-0001 §3) поверх снапшота — state/action(masking)/reward,
  transition с travel-time из матриц + congestion-профиль. Пинить `gymnasium`/`numpy`.
- Phase 4 baseline (OR-Tools + greedy) — до RL, для честного «Было/Стало» (запрет №3).

## 🚧 Blockers
- нет
