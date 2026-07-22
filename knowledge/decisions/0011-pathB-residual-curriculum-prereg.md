---
type: decision
id: dec-2026-07-22-pathB-residual-prereg
date: 2026-07-22
status: accepted
tags: [decision, prereg, path-b, residual, curriculum, rl, cvrptw, phase6b]
---

# 0011 — Path B: residual-curriculum, ПРЕДРЕГИСТРАЦИЯ (Phase 6b)

**Это предрегистрация: гейт, правило отбора и kill-критерии зафиксированы ДО кода и ДО прогона.**
Коммитится ОТДЕЛЬНЫМ коммитом ПЕРЕД обучающим кодом — git-таймстамп = гарантия, что цель не
подгонялась под результат. Пост-фактум правки этого файла запрещены (только новый decision).

**Контекст:** [[0010-phase6b-ablation-latency-niche]] закрылся отрицательно — достижимый RL-старт
(одиночный greedy-decode) на re-plan ХУЖЕ greedy (865.5 vs 851.5, выигрывает 6/25). Path B —
единственный путь к «RL по качеству»: дообучить политику НА residual-распределении, чтобы её
**одиночный decode** стал реально лучше greedy на re-plan. Time-boxed, один заход, честный гейт.

## ГЕЙТ (первичный критерий успеха — pre-registered)

На **25 событиях харнесса 0004** (`run_dynamic.iter_events`, `generate_instance` сиды 0–4 × 6
событий) **одиночный greedy-decode** `rl_raw` бьёт `greedy_raw`, парно по событиям:

> **median Δ (rl_raw − greedy_raw) < 0 € И wins(rl_raw) > 12/25** (т.е. ≥ 13 из 25).

Считается `python scripts/run_ablation.py --ckpt results/policy_pomo_residual.pt` — чтением
`analysis.start_rl_vs_greedy`: `median_delta_eur < 0` И `a_wins > 12`. Метрика ДЕТЕРМИНИРОВАНА
(raw greedy-decode, без wall-clock-конфаунда) → воспроизводимый pass/fail. `run_ablation` НЕ
меняется — гейт это чтение уже существующей строки.

## Правило отбора чекпойнта (pre-registered)

Чекпойнт, подаваемый в гейт, = **best-by-val-residual**, где val-residual cost = **одиночный
greedy-decode** (та же величина, что читает гейт) на ФИКСИРОВАННОМ held-out пуле residual-состояний
(≥48, сиды дизъюнктны с гейтом). Отбор БЕЗ единой ссылки на сиды 0–4. Anti-forgetting-ось (полная
congestion-статика) — только МОНИТОРИНГ в логе, НЕ критерий отбора. Warm-start-floor: деплой ≥
одиночного decode congestion-best на том же пуле (нулевой исход = warm-start).

## Kill-критерии (pre-registered)

- **patience = 15** эпох без улучшения val-residual → early-stop (best-by-val сохранён).
- **wall-time ≤ 36 ч** (сервер NJ, по [[train-on-server]], `TRAIN=scripts/train_residual.py`).
- **ОДИН заход.** Ни второй попытки, ни retune-and-rerun после взгляда на гейт. Провал = валидный
  исход «Path B не берёт гейт».

## Seed-дизъюнкция (валидность гейта — enforced тестом)

**Held-out — по СИДУ (реализация спроса/окон), не по node_id.** Все инстансы тянутся из ОДНОГО
реального пула аптек Аугсбурга (запрет №5 — реальные данные; `generate_instance` по построению
берёт полный снапшот, node_ids идентичны у всех сидов, разнятся demand/окна). Node_id-дизъюнкция
поэтому физически невозможна и не является held-out'ом — как и во всём проекте (Шаг1/refit:
train/val/test = сид-диапазоны, не срезы узлов). Пиновка сид-диапазонов (пересечений НЕТ):

| набор | источник | сиды |
|-------|----------|------|
| ГЕЙТ / деплой full-static | `generate_instance` | 0–9 (гейт: 0–4) |
| full-static val (anti-forget, монитор) | `InstanceSampler` | 1_000_000–1_000_063 |
| residual-train (база) | `InstanceSampler(62,62)` | ≥ 3_000_000 |
| residual-val (пул отбора) | `InstanceSampler(62,62)` | 4_000_000–4_000_047 |

Residual-база = `InstanceSampler(n_range=(62,62))` (полный набор 62 аптек, held-out по сиду через
спрос), НЕ `generate_instance` напрямую: последний перезагружает снапшот с диска (~4 с/вызов) →
непригоден для тысяч residual-построений за прогон. Тот же снапшот, что гейт (кэш загружается 1 раз);
геометрия/окна общие, разнятся demand-реализации по сиду. Гейт остаётся на `generate_instance(0–4)`.

Residual-обучение НЕ касается сидов 0–9. Тест `test_residual_seed_disjoint` ассертит: (а) сид-
диапазоны train/val-residual не пересекают {0–9}; (б) demand-векторы `generate_instance(residual)`
≠ `generate_instance(гейт)` (реализации held-out, геометрия общая — реальный город).

## Дизайн (раскрытые заранее выборы)

- **Residual = свежий CVRPTW** (`residual_instance`: депо + необслуженные + срочные, окна сдвинуты).
  POMO работает БЕЗ изменений — `feasible_starts` на residual-env = «K допустимых СЛЕДУЮЩИХ узлов»;
  `_decode`/shared-baseline/`train_batch` переиспользуются как есть.
- **Prefix-rollout = GREEDY** (не политика) — намеренно: `iter_events` берёт served из greedy-
  исполнения, greedy-префикс совпадает с served-распределением гейта; policy-префикс учил бы на
  ДРУГОМ распределении и мог провалить гейт «не за то».
- **Прогресс frac ∈ [0.2, 0.8]** — now_min выбирается так, что ровно round(frac·n_cust) обслужено
  (по finish-таймам greedy-исполнения). Затем ОДНО событие (traffic/urgent/breakdown) на now_min.
- **База residual = full-62** (`InstanceSampler(62,62)`, кэш; см. seed-таблицу) — совпадает с
  размером гейта (нет size-gap).
- **Микс 50/50**: 50% полных congestion-эпизодов (`InstanceSampler`, распределение congestion-best
  — anti-catastrophic-forgetting) + 50% residual-эпизодов. Warm-start = `policy_pomo_congestion.pt`;
  новый файл `results/policy_pomo_residual.pt` (best.pt/congestion/refit НЕ трогаем).
- Стражи как refit: |g|>0 (freeze-guard), энтропия, no-NaN, mem-gap, freshness-хеш; residual-
  генератор отбраковывает вырожденные состояния (< 2 допустимых стартов → ресэмпл).

## Раскрытые gap'ы train↔test (pre-committed, НЕ пост-фактум оправдания)

1. Train-residual = **одно** событие в случайной точке прогресса; гейт (0004) включает
   **накопленные многособытийные** состояния (поток из 6). Distribution shift известен заранее.
2. `frac ∈ [0.2, 0.8]` — предполагаемый диапазон обслуженной доли; гейт не привязан к frac.

## Диспозиция

- **Pass** (гейт взят) → decision 0012: промоут `policy_pomo_residual.pt`, перегон ablation
  (0010-харнесс) с новым чекпойнтом (вторично: сдвинулась ли tight-budget-картина).
- **Fail** → decision 0012: «Path B не берёт гейт» — честное закрытие тезиса «RL по качеству».
  Без ретрая. В обоих исходах перегоняем ablation для полной картины.

## Тесты (enforce предрегистрацию)

seed-дизъюнкция (train/val-residual ∩ {0–9} = ∅); residual feasible + непустой пул pending +
≥2 стартов; POMO-multistart работает на residual; val-residual = single-decode (== метрика гейта);
микс ~50/50 (seeded); обе оси в логе (residual + full); smoke: residual-cost↓ и full НЕ деградирует.
Связи: [[0010-phase6b-ablation-latency-niche]] · [[0007-phase6b-congestion-training]] · [[0006-pomo-static]].
