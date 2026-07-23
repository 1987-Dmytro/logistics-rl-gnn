<!-- AUTO-GEN START (refreshed by scripts/refresh-hot-cache.py) -->

# Hot Cache

**Auto-refreshed:** 2026-07-23 09:56:18 (every SessionStart)
**Branch:** `main`

## 🔀 Recent commits (top 5)

```
a614509 feat(viz): Phase 8 — визуализация кейса + финальные метрики Было/Стало
a174489 docs(decision): 0012 Path B residual-curriculum — гейт FAIL (Phase 6b закрыт)
c26f6ae feat(ablation): латентная ниша RL — ниши нет (Phase 6b, decision 0010)
d169002 feat(train): Path B residual-curriculum обучение (Phase 6b, 0011-prereg)
e93ceb5 docs(decision): 0011 residual-база = InstanceSampler(62,62) (кэш, не reload)
```

## 📋 Recent decisions

- `0012-pathB-residual-verdict.md` — 0012 — Path B: вердикт по предрегистрации (Phase 6b)
- `0011-pathB-residual-curriculum-prereg.md` — 0011 — Path B: residual-curriculum, ПРЕДРЕГИСТРАЦИЯ (Phase 6b)
- `0010-phase6b-ablation-latency-niche.md` — 0010 — Ablation: латентная ниша RL (Phase 6b)

## 📅 Recent daily logs

- `2026-07-23.md`
- `2026-07-22.md`
- `2026-07-21.md`

<!-- AUTO-GEN END (everything below preserved across refreshes) -->

# Hot Cache — curated

**Last update:** 2026-07-23 09:46 (правится руками / `/close`; секция выше — авто, маркер НЕ трогать)

## 🔥 What's Hot
- **Phase 8 — визуализация + финальные метрики (LinkedIn-кейс, ЗАКРЫТ):** карта Аугсбурга Было/Стало
  (+folium), 2 GIF re-plan (пробка+поломка, задетые рёбра красным), кривые обучения всех фаз,
  `final_metrics` (сводка из durable-json, **парити-страж 631.6€=0009**). **Честная таблица:** издержки
  **−23.5%** к greedy, **машино-часы наряда −39.6%** (простой окон, НЕ вождение — пробег ~flat −1%,
  верифицировано travel-декомпозицией), **+3.4%** к OR-Tools, реакция **×2.9** (деплой-система 689мс).
  **Латентность разделена (advisor):** нейро-floor ~15мс = потолок скорости, но качество-инфериор
  (0010); деплой-качество даёт portfolio+polish 689мс. Скрипты+малые PNG/GIF в git (исключ. №1),
  веса/folium/results — вне. `docs/final_metrics.md`. Коммит `a614509`.
- **Phase 6b Path B — residual-curriculum: ГЕЙТ FAIL, тезис ЗАКРЫТ ([[0012-pathB-residual-verdict]]):**
  предрегистрированный гейт 0011 НЕ взят — одиночный decode `rl_raw` **863.4 vs greedy_raw 851.5**
  (median Δ **+14.84€**, **7/25**; нужно <0 И >12/25). Обучение здоровое (early-stop ep48, отбор ep33,
  val-RES +1.6%, val-FULL стабилен → anti-forget держал), но выигрыш на single-event val-пуле НЕ
  перенёсся на многособытийный гейт (раскрытый в 0011 gap #1). **Один заход, без ретрая.** residual
  sha `dfe8401cc40d519c` (вне git). **Итог Phase 6b:** RL по КАЧЕСТВУ классику не бьёт (A→0008→0009→
  0010→0012 — у сходимости сравнялся, под бюджетом проиграл); вклад RL = латентность vs OR-Tools.
  [[0011-pathB-residual-curriculum-prereg]].
- **Phase 6b Ablation — латентная ниша RL: НЕТ ([[0010-phase6b-ablation-latency-niche]]):** на 25
  событиях 0004 под бюджетом {50,100,200,500}мс — достижимый RL-старт (**одиночный decode**) ХУЖЕ
  greedy (865.5 vs 851.5, **6/25**) И медленнее (**18 vs 7мс**); под тесным бюджетом `rl_polish`
  ПРОИГРЫВАЕТ `greedy_polish` (@50мс Δ̃+10.3€, 18/25) — анти-ниша; polish выравнивает с ростом бюджета
  (200/500мс монетка 12/11). **Открытый вопрос 0009 закрыт отрицательно.** Итог: RL по качеству НЕ
  бьёт классику (у сходимости сравнялся, под бюджетом проиграл); латентный выигрыш — только vs OR-Tools.
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
- **Phase 6b ЗАКРЫТ** — все пути исчерпаны (Path A congestion 0007 · inference-search 0008 · polish
  0009 · ablation 0010 · Path B residual 0012). Честный итог: **RL по качеству классику не бьёт;
  устойчивый вклад GNN+RL — латентность мгновенного ответа vs OR-Tools**, не качество vs greedy.
- **Phase 8 кейс готов** — карты/GIF/кривые/таблица (`docs/`). Дальше (по запросу): writeup/публикация
  LinkedIn-кейса; опц. деплой-слой (RL-старт как anytime-кандидат в портфеле, качество тянут
  multistart+polish). Новой RL-«качество»-ветки НЕ открывать без нового рычага (тезис закрыт честно).
- Опц.: закоммитить висящий vault-housekeeping (`.vault-state.json`, `index.md`).

## 🚧 Blockers
- нет
