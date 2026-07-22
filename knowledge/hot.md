<!-- AUTO-GEN START (refreshed by scripts/refresh-hot-cache.py) -->

# Hot Cache

**Auto-refreshed:** 2026-07-22 19:29:46 (every SessionStart)
**Branch:** `main`

## 🔀 Recent commits (top 5)

```
0bf3051 docs(decision): 0008 inference-search (Phase 6b Шаг 3) + hot/daily
be79d27 feat(replan): inference-search — sample-K batched decode + PortfolioPlanner (Phase 6b Шаг 3)
63c3fd3 docs(decision): 0007 congestion-training (Path A) — статика −1.7%, динамика −0.4%
e0d21cc feat(scripts): run_dynamic --ckpt/--out (переоценка таблицы 0004, Шаг 2 Piece 5)
f0dc54d feat(scripts): congestion Было/Стало eval + --congestion wiring + tests
```

## 📋 Recent decisions

- `0009-phase6b-local-search-polish.md` — 0009 — Local-search polish (Phase 6b · Шаг 3.5)
- `0008-phase6b-inference-search.md` — 0008 — Инференс-поиск (Phase 6b · Шаг 3)
- `0007-phase6b-congestion-training.md` — 0007 — Обучение под congestion (Phase 6b · Шаг 2)

## 📅 Recent daily logs

- `2026-07-22.md`
- `2026-07-21.md`
- `2026-07-20.md`

<!-- AUTO-GEN END (everything below preserved across refreshes) -->

# Hot Cache — curated

**Last update:** 2026-07-22 21:30 (правится руками / `/close`; секция выше — авто, маркер НЕ трогать)

## 🔥 What's Hot
- **Phase 6b Шаг 3.5 ЗАКРЫТ — local-search polish (БЕЗ обучения):** polish (2-opt+Or-opt intra,
  relocate+swap inter; full-eval per move → корректно под time-dependent) — **доминирующий рычаг**:
  polished-portfolio **631.6€**, gap к OR-Tools **+3.4%** (было +25.4% в Шаге 3), крупнейший сдвиг фазы.
  **НО polish СТИРАЕТ преимущество нейроконструктора:** greedy→652.2, RL→658.9, sample-K→650.9 — все
  старты сходятся в ~650-659, **RL-edge +1.0% (RL после polish даже ХУЖЕ greedy)**. Выигрыш =
  классический local search, достижим из greedy-старта → **НЕ доказывает GNN+RL-тезис**. Динамика −2.8%
  vs greedy — **конфаунд** (polished-portfolio vs непополир. greedy); честно = гарантия **0/25** +
  латентность **689мс<1с**. **Урок:** feasibility строго как env (пер-клиент возврат ≤ T_max — иначе
  под асимметр. OSM polish вернёт env-инфеасибл; поймал adversarial-воркфлоу). [[0009-phase6b-local-search-polish]].
- **Phase 6b Шаг 3 — инференс-поиск (БЕЗ обучения, PortfolioPlanner):** **динамический пол
  ЗАКРЫТ** — портфель { sample-K ∪ RL-multistart ∪ greedy } ≥ greedy на **25/25 событиях** по
  построению (байт-идентичный greedy-кандидат), 0004 re-plan **−0.9%** vs greedy (в 0007 было **+1.6%
  ХУЖЕ**), латентность **430мс <1с** (OR ×5). Статика: portfolio **766.1€** (−7.2% vs greedy, +25.4%
  vs OR); дискриминатор — sample-K дал **−2.45% сверх** pre-existing multistart (785.3→766.1, обходит
  best.pt 770.4). **Честно: динамика — реальный ново-выигрыш; статику тянет multistart, sample-K
  скромен.** Латентность **env-bound** (~линейна по K; нейронка батчится). [[0008-phase6b-inference-search]].
- **Phase 6b Шаг 2 — congestion-обучение (Path A + warm-start от 770.4€):** статика **−1.7%**
  (RL-cong vs RL-ff под congestion, вернул greedy-паритет+; free-flow-best под congestion был +0.8% =
  OOD), динамика (0004 re-plan) **−0.4%** (слабо, но не выброс: 16:7 событий). **RL под динамикой НЕ
  обгоняет greedy** (event-dependent); латентность **×134** цела; **floor → деплой ≥ warm-start**.
  Модест-плюс, статика > динамика. `results/policy_pomo_congestion.pt` (вне git). [[0007-phase6b-congestion-training]].
  **Урок:** диурнал ~невидим энкодеру (max-норма стирает равномерный c) → **инциденты на t0 — весь сигнал**.
- **Шаг 1·refit — анти-оверфит POMO валидирован:** «Стало» **783.2€** (не побил прошлый 770.4€, +1.7%),
  НО обобщение чисто (train/val/test/deploy согласованы → memorization нет) → выигрыш RL реален, не оверфит.
  `policy_pomo_best.pt` (770.4€) = деплой-модель (refit/congestion — отдельные файлы). [[0006-pomo-static]].
- **Phase 7 (динамика/латентность):** RL реагирует **×134** быстрее OR-Tools (forward-pass без поиска);
  качество event-dependent, OR сильнее на больших residual. env НЕ переписан. [[0004-dynamics]], `run_dynamic.py`.
- **КРИТИЧНО — коллапс обучения (Phase 6):** decoder БЕЗ `C·tanh` (насыщение зануляло grad), advantage
  НОРМИРУЕТСЯ; freeze-guard `|g|≈0` 3 эпохи → обрыв. НЕ возвращать tanh-clip / ненорм. adv.
- «Было» Phase 4: greedy **−825€**, OR-Tools **−611€** ([[0002-baselines]], `results/baselines.json`).
  Единый `env/scoring.py:evaluate_solution` — одна reward-формула для среды И бейзлайнов/политики.

## ⏭️ Next
- **Открытый вопрос тезиса**: есть ли ниша, где polish НЕ выравнивает старты (очень большой residual /
  жёсткий realtime-бюджет, где сходимость polish не успевает)? Иначе честный итог Phase 6b: **вклад RL =
  латентность (×3–134), по КАЧЕСТВУ классика (multistart, local search) доминирует**.
- **Path B (residual-дообучение)** — только если нужна победа RL над greedy ПО КАЧЕСТВУ на больших
  residual: обучение на распределении re-plan (депо+необслуженные+срочные, окна сдвинуты). Резерв.
- Опц.: закоммитить висящие vault-правки прошлых /save (0006-refit-секция, daily 21/22, index).

## 🚧 Blockers
- нет
