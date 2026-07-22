---
type: decision
id: dec-2026-07-22-local-search-polish
date: 2026-07-22
status: accepted
tags: [decision, local-search, polish, portfolio, dynamic, cvrptw, phase6b]
---

# 0009 — Local-search polish (Phase 6b · Шаг 3.5)

**Контекст:** [[0008-phase6b-inference-search]] закрыл динамический пол (portfolio ≥ greedy) и
скромно срезал static (sample-K −2.45% сверх multistart). Шаг 3.5: **БЕЗ обучения** — классический
нейроконструктор + полировка декодированных маршрутов local search'ем. Гипотеза: polish дотянет
качество к OR-Tools. Чекпойнт congestion-best (sha `24c8cfb0607235f8`, тот же, что 0007/0008).

## Что построено

1. **`env/scoring.py::check_feasible`** — жёсткая feasibility (cap/TW/T_max/fleet) **зеркалом**
   time-walk `evaluate_solution` (без второй логики стоимости). Строго как `env._feasible` —
   **включая пер-клиентскую проверку возврата** `t_after + travel(j,0) ≤ T_max` (см. «Корректность»).
2. **`replan/local_search.py::polish`** — 2-opt + Or-opt(1–3) intra, relocate + swap inter.
   Стоимость/feasibility — ТОЛЬКО `evaluate_solution`/`check_feasible`. **Full-eval каждого
   кандидата** (переоценка целиком → корректно под time-dependent congestion: реверс/перенос сдвигает
   все downstream-времена, delta-cost неверен). First-improvement, циклы до сходимости или `budget_ms`.
   **Инвариант: результат ≤ входа**. Операторы — по существующим слотам (vehicles_used не растёт).
3. **Интеграция**: `PortfolioPlanner` полирует топ-M в общем бюджете (исходные кандидаты в пуле →
   гарантия ≤ greedy цела); static-eval полирует каждый старт до сходимости. `run_polish.py`.

## Корректность (adversarial review, Ultracode)

5-линзовый воркфлоу (never-worse · feasibility-looser-than-env · операторы · детерминизм ·
edge-cases) → verify. **1 подтверждённый баг**: `check_feasible` проверял T_max-возврат только на
финальном ребре, а env — после КАЖДОГО клиента. Под **асимметричным** (не-метрич.) OSM-travel
(односторонние, geometry) polish мог принять дешёвый 2-opt-сосед, env-инфеасибл (из промежуточного
узла не успеть вернуться), и **вернуть его**. Пофикшен (пер-клиент возврат = env) + 2 регресс-теста
(оракул + money-path). Операторы/детерминизм/copy-safety — чисто. pytest **99 passed**.

## Результат (static до сходимости, budget 30000мс; dynamic in-budget 400мс)

**Разложение polish** (full-62, seeds 0–9, free-flow):

| старт | raw € | polished € | Δ polish |
|-------|-------|-----------|----------|
| greedy | 825.4 | **652.2** | −21.0% |
| RL-multistart | 785.3 | **658.9** | −16.1% |
| sample-K | 789.8 | **650.9** | −17.6% |

**Static gap**: polished-portfolio **631.6€** → vs Step-3 766.1 **−17.6%**, vs OR-Tools 611.1 **+3.4%**
(было +25.4% в 0008). **Dynamic** (0004, 5×6): RL-portfolio+polish **827.3€ vs greedy 851.5 (−2.8%)**,
**гарантия 0/25 нарушений**, латентность **689мс <1с** (OR 2001мс, ×3).

## Вывод — честный (веду с главного)

- **Polish СТИРАЕТ преимущество нейроконструктора.** greedy→652.2, RL→658.9, sample-K→650.9 — **все
  старты сходятся в окно ~650–659**, RL-edge **+1.0%** (RL после polish даже чуть ХУЖЕ greedy).
  Классический local search — доминирующий рычаг; RL-старт выигрыша не даёт. Это асимметрия
  [[0008-phase6b-inference-search]] на ступень глубже (там тянул pre-existing multistart, не sample-K;
  тут — polish, не политика).
- **+3.4% к OR-Tools — достижение POLISH, не RL.** Тот же результат берётся из greedy-старта; НЕ
  доказывает GNN+RL-тезис. Абсолютно, впрочем, крупнейший сдвиг фазы: gap к оптимизатору +25%→+3.4%.
- **Динамика −2.8% — конфаунд** (polished-portfolio vs **непополированный** greedy). Честная рамка:
  деплой-сравнение (портфель+polish vs дешёвый greedy) + гарантия по построению 0/25 + латентность
  <1с. НЕ приписываю −2.8% политике. 689мс — медиана локальной машины (хвост под нагрузкой может >1с).
- **Сходимость проверена**: пробой на тяжёлом seed1 (greedy→634.7 за ~16с, плоско при 20/40/80с);
  8000мс-прогон был budget-bound (portfolio 643.2) → перегнал на 30000мс (631.6, сошлось).

## Дальше

Static почти у OR-Tools (+3.4%) — потолок конструктив+polish близок. Открытый вопрос для тезиса
проекта: **есть ли у GNN+RL ниша, где polish НЕ выравнивает** (напр. очень большие residual /
жёсткий realtime-бюджет, где сходимость polish не успевает). Иначе честный итог Phase 6b: латентность
(×3–134) — реальный вклад RL; по качеству классические методы (multistart, local search) доминируют.

## Тесты

check_feasible (env-parity + границы cap/TW/T_max/fleet + agreement + **асимметрия** + money-path),
операторы (консервация клиентов + depot-концы), polish (не хуже входа ×6 сидов, feasibility,
детерминизм при сходимости, бюджет, parity-оптимум brute-force), portfolio+polish гарантия ≤ greedy.
Provenance/sha — `results/polish_summary.json` (вне git, запрет №1).
Связи: [[0004-dynamics]] · [[0006-pomo-static]] · [[0008-phase6b-inference-search]].
