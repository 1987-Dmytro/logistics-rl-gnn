---
type: decision
id: dec-2026-07-23-pathB-residual-verdict
date: 2026-07-23
status: accepted
tags: [decision, path-b, residual, curriculum, rl, cvrptw, phase6b, verdict]
---

# 0012 — Path B: вердикт по предрегистрации (Phase 6b)

**Исход предрегистрации [[0011-pathB-residual-curriculum-prereg]]: гейт НЕ взят (FAIL). Один заход,
без ретрая — как зафиксировано ДО прогона.** Path B — последняя проверка тезиса «GNN+RL даёт
качество»; residual-curriculum должен был сделать ОДИНОЧНЫЙ decode политики лучше greedy на re-plan.
Не сделал.

## Прогон (сервер base-node, warm-start congestion-best sha `24c8cfb0607235f8`)

Микс 50/50 (full-congestion + residual), отбор best-by-val-residual (single-decode = метрика гейта).
Здоров весь путь: |g|>0 (400–780), энтропия жива (~0.06–0.10), start_std>0, freshness ротирует,
no-NaN, val-FULL стабилен ~715 (**anti-forgetting держал**, g≈0 — не разменял полную статику).
**Early-stop эпоха 48** (15 эпох без улучшения val-residual). **Отбор — эпоха 33:** val-RES **364.2
(g+1.6%)**, val-FULL 713.8 (g−0.2%). residual-ckpt sha `dfe8401cc40d519c` (вне git, запрет №1).

val-residual сполз лишь с +3.07% (warm-start планка) до **+1.6%** — RL-старт на single-event пуле
всё ещё ХУЖЕ greedy, лишь чуть менее.

## ГЕЙТ (pre-registered, 25 событий 0004, generate_instance 0–4)

`start_rl_vs_greedy` — одиночный greedy-decode `rl_raw` vs `greedy_raw`, парно по событиям:

| метрика | требование гейта | факт | итог |
|---------|------------------|------|------|
| median Δ (rl−greedy) | **< 0 €** | **+14.84 €** | ✗ |
| rl-wins | **> 12 / 25** | **7 / 25** (greedy 16, ties 2) | ✗ |

(mean Δ +11.98€.) **Обе половины провалены → FAIL.** Для сравнения — та же величина на warm-start
(congestion-best, [[0010-phase6b-ablation-latency-niche]]): median +9.6€, 6/25. Residual-обучение
СДВИНУЛО rl_raw ровно на −2.1€ (865.5→863.4) — в пределах шума, а по парной медиане даже ХУЖЕ.

## Вывод — честный (веду с реального рычага)

- **Тренировка улучшила НЕ то распределение.** val-residual (single-event пул) сполз +3.07%→+1.6%,
  но на гейте (**накопленные многособытийные** состояния) перенос НЕ случился: rl_raw остался 863.4,
  парно 7/25. Сработал ровно **раскрытый в 0011 gap #1** (train = одно событие в точке прогресса;
  гейт = поток из 6). Селектор был направленно валиден (pre-flight: концордантен по знаку), но малый
  выигрыш на лёгком прокси не дотянул до тяжёлого гейта.
- **Не оверфит и не забывание:** val-FULL держался ~715 (anti-forgetting-микс работал), обучение
  здорово (|g|, энтропия, freshness). Path B честно обучился — просто одиночный RL-старт на re-plan
  так и не обошёл greedy.
- **Перегон ablation (вторично, тот же ckpt):** картина 0010 не сдвинулась — **ниши нет** (rl_raw
  863.4 vs greedy_raw 851.5; `rl_polish` проигрывает `greedy_polish` @50мс med+8.6€ 8/15, монетка на
  100/200/500). Новый чекпойнт tight-budget-вывод не меняет.

**Итог Phase 6b — закрыт.** Через Path A (0007), инференс-поиск (0008), polish (0009), ablation
(0010) и теперь residual-обучение (0012): **RL по КАЧЕСТВУ классику не бьёт** — у сходимости
сравнивается, под бюджетом/одиночным decode проигрывает. Единственный устойчивый вклад GNN+RL —
**латентность мгновенного ответа vs OR-Tools** (2001мс→14мс), не качество vs greedy. Тезис «GNN+RL
даёт качество» на реальном Аугсбурге не подтверждён — честно, с зафиксированным ДО прогона гейтом.

## Диспозиция

**Без ретрая** (0011). Гейт не взят по предрегистрации — валидное отрицательное закрытие, не повод
для retune-and-rerun. `policy_pomo_residual.pt` НЕ промоутится; деплой-модели прежние
([[0006-pomo-static]] `policy_pomo_best.pt`; congestion `policy_pomo_congestion.pt`). Дальше — если
и продолжать, то не «RL по качеству», а инженерия латентного слоя (RL-мгновенный старт как anytime-
кандидат в портфеле, где качество тянут multistart+polish — что уже есть в [[0008-phase6b-inference-search]]/[[0009-phase6b-local-search-polish]]).

## Провенанс

residual-ckpt sha `dfe8401cc40d519c` · warm-start `24c8cfb0607235f8` · гейт-результат
`results/ablation_residual_gate.json` (400 записей, tag `simulated-on-real`, вне git). Прогон:
early-stop ep48, отбор ep33, сервер base-node. Связи: [[0011-pathB-residual-curriculum-prereg]] ·
[[0010-phase6b-ablation-latency-niche]] · [[0007-phase6b-congestion-training]].
