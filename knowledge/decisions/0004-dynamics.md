---
type: decision
id: dec-2026-07-21-dynamics
date: 2026-07-21
status: accepted
tags: [decision, dynamic, congestion, replan, cvrptw, latency, phase7]
---

# 0004 — Динамика + re-plan: «Стало по скорости реакции» (Phase 7)

**Контекст:** dec-0001 §4 — онлайн-перестроение при пробках/поломках/срочных заказах. Цель
Phase 7: латентность re-plan ≪ re-solve OR-Tools **при сравнимой стоимости** (dec-0001 §5).
Реюз travel-интерфейса Phase 3 (drop-in) — env НЕ переписан.

## Что построено

1. **CongestionTravel** (`env/travel.py`, тот же интерфейс `time(i,j,at)`):
   `t = t0·c(dow,h) · (1 + Σ_k I_k)`, `h = 08 + (offset+at)/60`. Инциденты **геометрические**
   (центр-коорд + радиус) → переживают срез инстанса. Паритет: `c≡1` + 0 инцидентов ⇒ ровно
   FreeFlow (тест). `c(dow,h)` — городской профиль, калибровка по форме TomTom Augsburg
   (пики 08/17), тег `simulated-on-real` (`config/congestion.py`).
2. **События** (`env/events.py`): traffic (инцидент), breakdown (−машина, стопы → пул),
   urgent (аптека с узким окном); плавный диурнал — НЕ триггер. Seeded-поток.
3. **Drop-in без переписывания env**: `DynamicVRPEnv(VRPEnv)` переопределяет `_load` → travel
   переживает внутренний `reset()` (rollout/greedy сбрасывают среду). Base env не тронут.
4. **evaluate_solution(travel=…)**: congestion-aware время (backward-compatible, дефолт free-flow).
5. **replan/compare.py**: re-plan residual (депо + необслуженные + срочные, окна сдвинуты в базу
   события) каждым методом; латентность (warmup + медиана, end-to-end) + качество (единый оценщик).

## Таблица «Стало по скорости реакции» (2 сида × 6 событий, deadline OR-Tools 2с)

`residual re-plan, mean по событиям · results/dynamic.json (вне git)`

| метод    | латентность (медиана) | cost,€ | on-time,% | unserved | × медленнее RL |
|----------|-----------------------|--------|-----------|----------|----------------|
| greedy   | 7.8 мс                | 446.1  | 100.0     | 0.00     | 0×             |
| RL       | 20.3 мс               | 450.6  | 100.0     | 0.00     | 1×             |
| OR-Tools | 2001 мс               | 487.3  | 100.0     | 0.60     | **98×**        |

## Вывод (таблица + честная нюансировка — НЕ переклаиваем)

- **ГЛАВНЫЙ ГЕЙТ взят и робастен: RL реагирует ×98 быстрее OR-Tools (20мс vs 2с), суб-100мс
  абсолютно.** Это forward-pass без поиска — целевое свойство динамики (dec-0001 §5).
- **Качество — event-dependent, НЕ победа RL.** OR-Tools остаётся сильнейшим оптимизатором на
  БОЛЬШИХ residual (n=56: OR 548€ < RL 673€ — RL хуже). Агрегатный проигрыш OR-Tools в таблице
  идёт от **static-snapshot-пессимизма**, НЕ от дедлайна (проверено: 2с→8с меняет OR 548→542€,
  уже сошёлся): OR-Tools не видит time-dependency → snapshot замораживает congestion на пиковом
  часе события (ratio ×1.46 сред., ×2.86 макс, без затухания инцидента/спада диурнала) → иногда
  роняет достижимый под истинным временем стоп (+200€), раздувая среднюю стоимость. RL ≈ greedy.
- **RL реагирует ЧЕРЕЗ feasibility, не через congestion-фичи**: `build_graph` кормит free-flow
  рёбра, политика обучена free-flow (Phase 6) → под congestion RL **вне распределения (OOD)**.
  Он не «объезжает пробки», а лишь соблюдает time-dependent маску. Отсюда RL хуже на больших
  residual. Congestion-фичи + дообучение на residual → Phase 8.

## Упрощения (ponytail, явные)

- **class(e) схлопнут в один городской профиль** — OD-матрица снапшота без per-segment класса
  (`with_graph=False`); TomTom Traffic Index и есть city-level. Upgrade: per-edge на графе.
- **re-plan = переоптимизация оставшихся стопов от депо** со свежим T_max/машину от времени
  события (стандартный periodic re-optimization dynamic-VRP; env не поддерживает mid-route
  continuation). Не «продолжение с середины леги».
- **static snapshot OR-Tools = congestion на момент re-plan** (реалистично для статик-решателя:
  дисп берёт текущий трафик), но пессимистично (не видит спад) → см. нюанс выше.
- **частичное состояние привязано к timeline исполняемого greedy-плана** (`served_by`), не
  к рандому и не к пере-симуляции после каждого re-plan.

## Альтернативы отклонены

- **переписать VRPEnv под параллельные машины в wall-clock** — большая переделка ядра ради
  mid-route continuation; periodic re-opt даёт тот же латентность/качество-замер дешевле.
- **snapshot по среднему/прогнозному congestion** — дал бы OR-Tools лучший static-шанс, но это
  уже форкаст (Phase 8 backlog), не честный статик-бейзлайн «здесь и сейчас».
- **заявить «RL бьёт OR-Tools по качеству»** — неверно (event-dependent, OR сильнее на больших
  n; агрегат искажён snapshot-пессимизмом). Публикуем латентность как хедлайн, качество — как есть.

**Воспроизводимость (запрет №4):** **латентность И качество OR-Tools — wall-clock-dependent**:
GLS крутит соседства до дедлайна → быстрее/свободнее железо укладывает больше итераций → ДРУГОЙ
маршрут → другой cost/on-time/unserved (проверено adversarial-ревью: тот же seed+config, 5 повторов
→ reward −589.14 vs −587.45). Детерминировано seed+config ТОЛЬКО качество RL/greedy (+ фикс. hash
чекпойнта). Все числа — median/mean на фикс. железе; provenance (hash `policy_best.pt` + версии
torch/numpy/ortools + platform) в `results/dynamic.json` (запрет №4 для трассируемости артефактов).
Тесты: паритет с FreeFlow, env_checker под congestion, события (traffic/breakdown/urgent), re-plan
feasible, ГЛАВНЫЙ ГЕЙТ (RL-латентность на порядки < OR).
