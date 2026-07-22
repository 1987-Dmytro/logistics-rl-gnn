---
type: decision
id: dec-2026-07-22-inference-search
date: 2026-07-22
status: accepted
tags: [decision, inference, search, pomo, portfolio, dynamic, cvrptw, phase6b]
---

# 0008 — Инференс-поиск (Phase 6b · Шаг 3)

**Контекст:** [[0007-phase6b-congestion-training]] закрыл обучение под congestion, но с честным
потолком — **RL под динамикой НЕ обгоняет greedy** (0004-re-plan: RL +1.6% ХУЖЕ greedy,
event-dependent), а статический выигрыш слабо переносится на residual. Шаг 3: **БЕЗ обучения,
только decode** — инференс-поиск на congestion-best чекпойнте (sha `24c8cfb0607235f8`, тот же, что
0007). Цель: срезать static gap и **закрыть динамический пол** (сделать RL ≥ greedy).

## Что построено

1. **Батчевый decode** (`decoder.logits_batch`, `policy.sample_k`): `sample_k(K, temperature)` —
   K стохастических роллаутов, **один encode + батчевый декод по K** (нейронка векторно, один
   forward на шаг). Свой `torch.Generator` → детерминизм по seed (глобальный RNG не трогаем);
   `temperature>0`. Форс-старты НЕ навязываем (POMO multistart-greedy = отдельный кандидат).
2. **`take_best`** — лучший кандидат ЕДИНЫМ `evaluate_solution` (под тем же travel).
3. **`PortfolioPlanner`** (`replan/portfolio.py`): кандидаты = { sample-K ∪ RL-multistart-greedy ∪
   greedy-эвристика } → best + латентность end-to-end. **Гарантия ПО ПОСТРОЕНИЮ: результат ≤
   greedy** — greedy-кандидат байт-идентичен методу `greedy` в таблице (тот же
   instance+travel+fleet+scorer → `min(...) ≤ greedy` тождественно).
4. **Проброс в харнесс 0004** (`compare_replan(rl_planner=)`, `run_dynamic.run`) + `run_search.py`
   (3 замера + provenance → `results/search_summary.json`).

## Ключевой дизайн

- **Латентность инференс-поиска env-bound, НЕ neural-bound**: батч-декод держит нейронку ~плоской
  по K, растёт только K× `env.step`/шаг → латентность ~линейна по K (K=256 пробивает 1с). Отсюда
  динамика на **K≤32**, статика (офлайн, без гейта) на K=128.
- **Гарантия держится на байт-идентичности greedy-кандидата** — не на «RL умный».
- **sample_k — чистый temperature-сэмплинг** (multistart-greedy — pre-existing, отдельный кандидат).

## Результат (congestion-best, free-flow static + congestion dynamic)

**K-таблица** (sample-K take-best, full-62 free-flow, **3 сида** — латентно-качественная развёртка):

| K | best,€ | vs greedy | лат,мс |
|---|--------|-----------|--------|
| 16 | 830.0 | −0.1% | 114 |
| 128 | 783.4 | −5.7% | 811 |
| 256 | 782.3 | −5.8% | 1588 |

**Static** (full-62, seeds 0–9, free-flow, K=128; congestion-фичи нейтральны — mult≡1, node_cong=0):

| метод | € | vs greedy | vs OR 611 |
|-------|---|-----------|-----------|
| greedy | 825.4 | — | +35.1% |
| RL multistart-only (pre-existing) | 785.3 | −4.9% | +28.5% |
| sample-K take-best (новый, standalone) | 789.8 | −4.3% | +29.2% |
| **PortfolioPlanner** | **766.1** | **−7.2%** | **+25.4%** |

**Dynamic** (0004 harness, 5×6, K=32): RL-portfolio **843.9€ vs greedy 851.5 = −0.9%** (в 0007 было
**+1.6% ХУЖЕ**); **гарантия 0/25 нарушений, худшая Δ +0.00€**; латентность **430мс <1с** (OR-Tools
2001мс, ×5). unserved rl 2.0 = greedy 2.0 (OR 2.48).

## Вывод — честно (асимметрия)

- **Динамический пол ЗАКРЫТ (реально, ново — главный выигрыш Шага 3):** портфель ≥ greedy на КАЖДОМ
  из 25 событий по построению; было +1.6% хуже → стало −0.9% (берёт RL, где выигрывает, greedy —
  иначе). Латентность цела (430мс <1с). Это прямой лом под 0004-гэп «RL хуже greedy на residual».
- **Статика — скромно, тянет pre-existing multistart:** дискриминатор (multistart-only 785.3 vs
  portfolio 766.1) показывает, что **новый рычаг sample-K добавил −2.45% сверх multistart** (не ноль —
  гипотеза «бесполезен» опровергнута), и портфель обходит прежний деплой best.pt (770.4). Но тяжёлую
  работу делает **pre-existing** multistart-greedy, не инференс-поиск. Это маргинальный плюс, не прорыв.
- **OR-Tools по cost ниже RL только из-за 2с дедлайна** (душит OR: unserved 2.48 vs 2.0) — хедлайн
  держим на **гарантии + латентности**, не на «RL бьёт OR».

## Дальше

Динамический пол закрыт → RL-развёртка (портфель) безопасна к greedy. Если нужна победа RL над
greedy ПО КАЧЕСТВУ (не только ≥) на больших residual — Path B (residual-дообучение, [[0007-phase6b-congestion-training]]).
Иначе Phase 6b закрыта: congestion-обучение (модест) + инференс-поиск (пол закрыт, static срезан скромно).

## Тесты

parity батч-vs-одиночный decode, детерминизм sample_k по seed, `temperature>0`, гарантия
portfolio ≤ greedy на каждом инстансе, take_best пропускает None-кандидатов, латентность логируется,
планер не мутирует вход-инстанс. pytest **77 passed**. Provenance/sha — `results/search_summary.json`
(вне git, запрет №1; платформа локальная — качество RL/greedy детерминировано seed+config+весами).
Связи: [[0004-dynamics]] · [[0006-pomo-static]] · [[0007-phase6b-congestion-training]].
