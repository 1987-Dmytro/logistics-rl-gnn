---
type: decision
id: dec-2026-07-21-pomo-static
date: 2026-07-21
status: accepted
tags: [decision, pomo, reinforce, training, cvrptw, phase6b]
---

# 0006 — POMO на статике (Phase 6b · Шаг 1)

**Контекст:** [[0003-phase6-training]] дал REINFORCE + Kool rollout-baseline (804€, бьёт greedy).
Шаг 1 — заменить его на POMO (multi-start + shared baseline): проще (нет frozen-copy/t-test),
устойчивее к вырождению baseline. Free-flow статика; динамика — Шаг 2. Реюз [[0005-phase6b-congestion-obs]].

## Что построено

1. **train/pomo.py — POMOTrainer.** На инстанс: encode ОДИН раз → N траекторий из N РАЗНЫХ
   первых узлов (sample). shared baseline `b = mean_N(cost_i)`; `advantage_i = cost_i − b`
   (центрирован, БЕЗ std-норм — POMO baseline не вырожден, all-depot-патологии Phase 6 нет).
   `loss = mean(adv·Σlogπ)`, знак `+Σlogπ` → спуск ↓cost (== фикс Phase 6; флип → cost растёт).
   Убраны rollout-baseline / paired t-test / frozen-copy (shared baseline их заменяет, `p=nan` уходит).
   Adam + grad-clip@1 (clip нормирует ‖grad‖ → шаг ~lr; биндит каждый шаг — это ОК, не баг).
2. **Старты — только допустимые на шаге 0** (`feasible_starts`: mask==1, прорежены до max_starts).
   Форс инфеасибл-старта дал бы `log_prob=−inf → NaN` — гард. `<2` допустимых стартов → инстанс skip.
   **Форс-шаг ИСКЛЮЧЁН из градиента** (prob=1 у навязанного действия, канон POMO — Kwon; adversarial-
   ревью поймал отклонение). На инференсе `multistart_greedy` форсит все старты → π(·|s₀) всё равно
   не выбирает старт, так что член был бы бесполезным. Держим оценщик несмещённым/каноничным.
3. **Градиент-аккумуляция:** `(loss/b).backward()` на инстанс → память O(1 инстанс), не O(batch)
   (важно для full-62). encode переживает reset (статический граф; instance_fn игнорит seed).
4. **Инференс — multi-start greedy** (`multistart_greedy`): greedy-decode из N стартов, лучший.
   Быстро/параллельно. Валидация и финальный eval — им.
5. **config/pomo.py** (N/epochs/batch/lr/clip), **scripts/train_pomo.py** (`--smoke`/full;
   OR-Tools val-референс инъекцией из скрипта, не в `__init__` → тесты без ortools).
6. **Часть 0 — edge-канал `congestion_multiplier`** (см. [[0005-phase6b-congestion-obs]], закрывает
   его нюанс): `edge_attr [E,2]` = [travel-норма (топология, стирает равномерный диурнал),
   `travel/free_flow=c·(1+ΣI)` (диурнал/инцидент ВИДНЫ по-рёберно)]. Под free-flow канал1≡1
   (нейтрален для статики), но делает диурнал видимым для Шага 2. diag→1, закрытие inf→cap=10.

## Почему НЕ 8x-augmentation

POMO-статья добавляет ×8 евклидовых аугментаций (отражения/повороты координат) — у нас **реальная
travel-матрица Аугсбурга** (не евклидова: асимметрии дорог, OD из OSRM), поворот координат НЕ
сохраняет времена. Аугментация была бы неверной. Multi-start (N стартов) — единственный источник
диверсификации; этого достаточно для shared baseline.

## Совместимость / метрика

- Веса — новый `results/policy_pomo_best.pt` (вне git). Старые чекпойнты Phase 6/6b-Шаг0
  несовместимы (edge_dim 1→2, in_dim/ctx уже росли в 0005) — переобучение здесь и есть цель.
- **Было/Стало валиден:** «Стало» = multi-start greedy на full-62/seeds 0–9 (ТЕ ЖЕ инстансы, что
  [[0002-baselines]]), единый `evaluate_solution`. Сравнение с greedy 825€ / OR-Tools 611€.
- Цель Шага 1: статический gap к OR-Tools ≤ прежних +31.6% (REINFORCE 804€). **Полный прогон —
  на сервере**; smoke (мак) лишь демонстрирует механизм (cost↓, |g|>0, разброс стартов).

## Smoke (мак, 5 эпох, иллюстративно)

`train 333.8→249 · |g|>0 (clip биндит) · start_std 50→12 (baseline жив) · H 1.87→0.79 (не коллапс)`.
val насыщен ≈эвристикой на малых инстансах → learning-сигнал берём с train. «Стало» смоука
(842€) — не показатель; реальное число даст полный прогон.

## Тесты (tests/test_pomo.py)

shared-baseline advantage не вырожден (RAW cost std>0), |g|>0, cost↓ (train), детерминизм по seed,
энтропия не коллапс, no-NaN на РЕАЛЬНОМ сэмплированном инстансе. Часть 0: паритет free-flow
(канал0 бит-в-бит, канал1≡1), оба канала конечны под закрытием, диурнал виден в канале1.
