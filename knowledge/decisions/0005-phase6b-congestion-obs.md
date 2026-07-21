---
type: decision
id: dec-2026-07-21-phase6b-obs
date: 2026-07-21
status: accepted
tags: [decision, congestion, observability, gnn, phase6b]
---

# 0005 — Наблюдаемость congestion (Phase 6b · Шаг 0)

**Контекст:** [[0004-dynamics]] показал RL OOD под congestion (free-flow-trained, реагирует лишь
через feasibility-маску). Шаг 0 — пламбинг: политика ВИДИТ время/пробки. БЕЗ смены обучения;
переобучение (POMO) — Шаг 1.

## Что сделано (3 канала congestion в модель)

1. **edge_attr энкодера** — `travel_time(i,j, cur_time)` АКТИВНОЙ travel-модели (снимок в cur_time),
   а не фикс. free-flow. Под FreeFlow == прежнее значение (паритет). Закрытие (inf) → крупный
   конечный «очень медленно» перед нормировкой (без NaN). Размерность ребра та же (1).
   **Точно:** per-instance max-нормировка `tm/tm.max()` СОКРАЩАЕТ диурнальный множитель (c
   одинаков по городу для одного снимка → `t0·c/(t0.max()·c)=t0/t0.max()`), поэтому edge_attr
   несёт **инциденты** (локальные, переживают отношение), а rush-hour доходит до модели через
   time-context (канал 3). Диурнал В рёбра = смена нормировки (фикс. free-flow reference) — это
   feature-решение Шага 1, НЕ Шага 0 (сломало бы чистый паритет).
2. **node_congestion** — новый признак узла (столбец 8): max по активным инцидентам вклада на узел
   (`Incident.at_node`, конечный сентинел `_CLOSED_LEVEL` при закрытии). 0 под free-flow. Энкодер
   `in_dim` 7→8.
3. **time-context** — `[sin_h, cos_h, sin_dow, cos_dow]` (фаза congestion) в контекст декодера.
   Декодер `ctx_extra` 2→6. Под free-flow — постоянный вход без congestion-сигнала.

obs дополнен: `node_features (k,9)` + ключ `time_context (4)`. `TravelModel.offset_min`,
`env.abs_minute`, `time_context()`, `node_congestion()` — общие хелперы (obs и `build_graph`
считают одно и то же). FreeFlow → всё нейтрально.

## Совместимость чекпойнтов (важно)

Старый `results/policy_best.pt` (Phase 6, in_dim=7 / ctx_extra=2) **несовместим по размерности** —
это ОЖИДАЕМО. Переобучение в Шаге 1 (POMO) даст новые веса. **Не грузить старый чекпойнт**;
`scripts/run_dynamic.py` / eval заработают только после Шага 1 (гард не добавляем — вне scope).

## Гейт-регрессия

Под FreeFlow congestion-признаки нейтральны: `build_graph` edge_attr бит-в-бит == прежней
free-flow-норме, node_congestion≡0. overfit-tiny сходится к **78.9€** (== optimum `[0,1,2,0]`) —
пламбинг не сломал статику. no-NaN forward проверен на CongestionTravel С ЗАКРЫТИЕМ (единственный
новый risky-путь: inf→сентинел). Тесты: `tests/test_congestion_obs.py` +
`test_model.test_overfit_tiny_cost_drops` (гоняет free-flow сквозь новый код).
