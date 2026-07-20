<!-- AUTO-GEN START (refreshed by scripts/refresh-hot-cache.py) -->

# Hot Cache

**Auto-refreshed:** 2026-07-20 18:12:09 (every SessionStart)
**Branch:** `main`

## 🔀 Recent commits (top 5)

```
e711c37 feat: Phase 1 — скелет репо + verifier-петля
835b43c chore: initial scaffold + MDP spec (0001)
```

## 📋 Recent decisions

- `0001-mdp-spec.md` — 0001 — Динамический CVRPTW: доставка по аптекам Аугсбурга

## 📅 Recent daily logs

- `2026-07-20.md`

<!-- AUTO-GEN END (everything below preserved across refreshes) -->

# Hot Cache — curated

**Last update:** 2026-07-20 18:11 (правится руками / `/close`; секция выше — авто, маркер НЕ трогать)

## 🔥 What's Hot
- Phase 1 ЗАКРЫТА (коммит `e711c37`): скелет `src/logistics_rl_gnn/` (пакеты docstring-only),
  smoke-тест, ruff pre-commit, justfile, README. Верификатор зелёный. Рантайм-депсы ещё пустые.
- Постановка RL-задачи зафиксирована → `decisions/0001-mdp-spec.md` (CVRPTW, аптеки Аугсбурга, OSM).
- Верификатор-петля: `pip install -e ".[dev]" && pytest -q && ruff check . && ruff format --check .`
  (или `just check`); ruff гоняется по всему репо, включая `scripts/`.

## ⏭️ Next
- Phase 2 (data): OSMnx-пайплайн — граф Augsburg drive + депо (PHOENIX VZ) + аптеки. Тогда
  запинить `osmnx`/`networkx`/`shapely` под CPU. Verify-гейт = % покрытия maxspeed.
- `/init` (code-discovery) наполнит fold-in в CLAUDE.md, когда `src/` обрастёт логикой.

## 🚧 Blockers
- нет
