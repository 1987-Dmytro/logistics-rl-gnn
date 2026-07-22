---
type: decision
id: dec-2026-07-22-congestion-training
date: 2026-07-22
status: accepted
tags: [decision, congestion, training, pomo, dynamic, cvrptw, phase6b]
---

# 0007 — Обучение под congestion (Phase 6b · Шаг 2)

**Контекст:** [[0004-dynamics]] вскрыл корень — RL реагирует на события ЧЕРЕЗ feasibility-маску,
а не через congestion-фичи (обучен free-flow → под congestion OOD, хуже на больших residual).
[[0005-phase6b-congestion-obs]] дал наблюдаемость (Шаг 0), [[0006-pomo-static]] — POMO на статике
(Шаг 1, free-flow). Шаг 2: **обучить POMO под активной congestion**, чтобы политика ИСПОЛЬЗОВАЛА
сигнал. Выбор оператора: **Path A (congestion-static) + warm-start от 770.4€** (residual-обучение
Path B — в резерве, если A не закроет dynamic-гэп).

## Что построено

1. **Векторизация `build_graph`** (`TravelModel.matrix()`): k² Python-вызовов `travel.time()` душили
   ретрейн при k=62 → один шот (FreeFlow→t0; Congestion→`t0·c·(1+Σинц)`, зоны outer-OR, closure union).
   Free-flow **бит-в-бит** (регрессия-safe, parity-тест vs поэлементный `time()`).
2. **Congestion-обучение** (`POMOConfig.for_congestion` + `POMOTrainer`): warm-start 770.4€, lr=3e-4,
   β=0.03; congestion-env фабрика (dow=delivery, offset+инциденты на **t0**, `t_start=offset` →
   активны с диспетчеризации, **coverage 100%** узлов); reward И greedy congestion-aware
   (`travel=env.travel`; free-flow → паритет). Веса → `policy_pomo_congestion.pt` (best.pt/refit целы).
3. **Warm-start floor**: стартовый val = планка, warm-start сохранён как ckpt → **деплой НЕ хуже
   warm-start** (нулевой исход = чистая находка, не тихая регрессия).
4. **Было/Стало под congestion** (`eval_congestion`) + **переоценка таблицы 0004** (`run_dynamic --ckpt`).

## Ключевой дизайн (осознанные упрощения)

- **encode ОДИН раз в t0** (congestion-снимок на диспетчеризации) + decoder `time_context` бежит по
  шагам; reward через `evaluate_solution(travel=)` — полное time-dependent время (корректно). НЕ
  per-step re-encode (как snapshot в 0004).
- **Диурнал почти невидим энкодеру** (математика): канал 0 max-норма сокращает равномерный `c`
  (`t0·c/max = t0/max`), канал 1 = `c` — константа, дублирует `time_context`. Значит **инциденты
  на t0 — весь сигнал** (локальны, переживают max-норму). Отсюда richness инцидентов критична
  (advisor); `≥1` инцидент в узле + долгоживущий → coverage 100%.

## Результат (early-stop ep30, best-by-val ep15, TRAIN_DONE=0)

**Планка** (free-flow-best под congestion): val 721.1, gap_greedy **+0.8%** — OOD съел выигрыш
(было −6.7% под free-flow → чуть ХУЖЕ greedy под congestion).

| Ось | Стало (RL-cong) | RL-cong vs RL-ff | vs greedy | vs OR |
|-----|-----------------|------------------|-----------|-------|
| **Статика** (Было/Стало, 32 held-out) | 712.2€ | **−1.7%** | −0.3% | +16.4% (snap-пессимизм) |
| **Динамика** (0004 re-plan, 5×6) | 865.5€ | **−0.4%** | +1.6% | ≈паритет |

- **Обобщение** (gap-to-greedy): train −0.2% · val −0.5% · TEST −0.9% — согласованы (train≈val≈test)
  → **memorization нет**; выигрыш переносится на held-out.
- **Латентность ×134 цела** (RL 15мс vs OR 2001мс — тот же forward-pass).
- Динамика −0.4% — **слабый, но консистентный сигнал, НЕ выброс** (парно 25 событий: median −4.84€,
  16/23 разошедшихся маршрутов за congestion vs 7 против; крупнейший |d|=+33€ — ПРОТИВ cong, т.е.
  итог не одним outlier'ом). congestion реально сместил поведение (23/25 маршрутов иные).
- Динамика: 5 сидов (0–4) vs 0004's 2 → **абс.числа иные** (сиды 2–4 тяжелее, unserved 2.0);
  сравнение RL-vs-RL/greedy — на ТЕХ ЖЕ сидах. Provenance/sha — `results/pomo_congestion_summary.json`.

## Вывод — честно

- **Congestion-обучение помогло, но СКРОМНО, и статика > динамика.** Статический выигрыш −1.7%
  (RL вернул greedy-паритет+ под congestion) **слабо переносится** на residual re-plan (−0.4%).
- **0004-гэп «RL хуже greedy на больших residual» почти не сдвинут** (+2.1%→+1.6%): residual
  (депо+необслуженные+срочные, окна сдвинуты, congestion в момент события) отличается от
  static-congestion достаточно, что полный перенос требует **Path B (обучение на residual)**.
  Линковка advisor «Path A → 0004» держится лишь направленно.
- **RL под динамикой НЕ обгоняет greedy** (event-dependent, как в 0004) — заявлять победу RL по
  качеству нельзя. Хедлайн остаётся латентность (×134); congestion-обучение — маргинальный плюс.
- **floor гарантировал отсутствие регрессии**; `policy_pomo_best.pt` (770.4€ static) не тронут.

## Дальше

Path B (residual-дообучение на распределении re-plan) — прямой лом под dynamic-гэп; либо per-step
re-encode (дороже) — если нужна победа RL над greedy на больших residual. Иначе Шаг 2 закрыт как
«congestion-обучение даёт модест-плюс, dynamic-качество остаётся ≈greedy».

## Тесты

matrix-parity (vs поэлементный `time()`, вкл. closure+граница зоны), congestion-train+coverage,
sampler-детерминизм, **warm-start floor не перезаписан худшей эпохой**. Все зелёные (pytest 70).
Связи: [[0004-dynamics]] · [[0005-phase6b-congestion-obs]] · [[0006-pomo-static]].
