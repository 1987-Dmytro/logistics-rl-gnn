---
type: decision
id: dec-2026-07-21-phase6-training
date: 2026-07-21
status: accepted
tags: [decision, rl, reinforce, cvrptw, gnn, collapse, phase5, phase6]
---

# 0003 — Обучение RL (REINFORCE) + «Стало» 804€ (Phase 5+6)

**Контекст:** после «Было» ([[0002-baselines]]) — обучить GNN+attention политику и получить
честное «Стало» на ИДЕНТИЧНЫХ инстансах/сидах (запрет №3). Сеть: GATEncoder (GATv2 ×3, d=128,
encode-once) + Kool-style AttentionDecoder + VRPPolicy (автогрегрессивный rollout, feasibility
из `env.action_mask`). Обучение: REINFORCE + rollout-baseline (Kool).

## Таблица «Стало» (full-62, seeds 0–9, greedy-decode)

`K=8 Q=80 T_max=240мин · снапшот augsburg_20260720 · == «Было» Phase 4`

| метод       | reward,€ | gap vs OR-Tools | против greedy |
|-------------|----------|-----------------|---------------|
| OR-Tools    | −611.14  | —               | −26%          |
| RL «Стало»  | −804.1   | +31.6%          | −2.6%         |
| greedy      | −825.38  | +35.0%          | —             |

**RL «Стало» −804€ БЬЁТ greedy −825€ (−2.6%)** — первый валидный выигрыш RL vs эвристика. Обучение здорово 100 эпох (`|g|` 13–20, `H`→0.24
уверенная, freeze-guard не сработал). Веса `results/policy_best.pt` (вне git). Конфиг —
`config/train.py:TrainConfig` (100 эпох, 50 шагов, batch 32, lr=1e-3, grad-clip 1.0, n∈[40,62]).

## Решение

1. **REINFORCE + rollout-baseline** (Kool): baseline = замороженная копия политики (greedy на
   ТЕХ ЖЕ инстансах, paired). Обновление baseline — paired t-test (`ttest_rel`, one-sided p/2 <
   0.05) на val-сидах (непересекающихся с train). Adam lr=1e-3, grad-clip=1.0 (находка Phase 5).
2. **Знак loss = +Σlogπ**, `mean((cost−b)·Σlogπ)`: ∇L=E[(cost−b)∇Σlogπ]=∇E[cost] → спуск ↓cost.
   Спек давал `−Σlogπ` (подняло бы cost) — отклонено, проверено эмпирически (cost падал).
3. **advantage НОРМИРУЕТСЯ** (mean0/std1, detach) — см. коллапс ниже.
4. **decoder БЕЗ `C·tanh`-клипа** логитов (сырой `q·k/√d`) — см. коллапс ниже.

## Критический коллапс обучения (корень = Phase-5 lr-коллапс)

Полный прогон #1 замёрз на epoch 1: `|g|→0`, «Стало» = all-depot 12400€. Диагностика a/b
(инструментация одного шага): код здоров при init (loss.requires_grad=True, Σlogπ.grad_fn≠None,
model.training=True) → не (a)/(b), а **(c) коллапс насыщения**. Три сцепленных причины:

- **tanh-clip насыщается** — `C·tanh(logits)` при |logits|≫1 даёт grad≈0 → сеть не учится.
- **ненормированный advantage** — all-depot baseline = n·200 (гигантский масштаб) вгонял
  softmax в насыщение за 1 эпоху ещё до полезного сигнала.
- **baseline-deadlock** — при all-depot greedy(current)==greedy(baseline) → `ttest_rel`
  даёт `p=nan`, baseline не обновляется, deadlock самоподдерживается.

**Ключевая связь:** тот же tanh-clip был корнем Phase-5 «lr=1e-2 → uniform-коллапс». Тогда
списали на lr; на самом деле насыщение клипа. Один корень, снятый здесь окончательно.

**Фикс:** (1) снять tanh-clip; (2) нормализовать advantage; (3) runtime freeze-guard —
`|g|<1e-6` 3 эпохи подряд → обрыв прогона (не 100 впустую); (4) smoke-стражи (grad>0,
val-двигается, baseline p не nan, энтропия ∈[0.05, log k]) — коллапс раньше прятался до 100
эпох; (5) entropy-бонус `+β·H` в резерве (`entropy_beta=0`, вкл. если softmax переострится
без клипа). Прогон #2 (фиксед) — здоров, дал «Стало» 804€.

## Альтернативы отклонены

- **Оставить tanh-clip, лечить lr** — не корень (Phase 5 показал), насыщение вернётся.
- **Не нормировать advantage, снизить lr** — маскирует масштаб, коллапс медленнее но тот же.
- **Показать «Стало» с прогона #1 (all-depot)** — недействительно (запрет №3/№4).

**Воспроизводимость:** seed+config в `TrainConfig`; «Стало» — greedy-decode на `eval_seeds`
0–9 (== «Было»). Стражи гарантируют: тихий коллапс больше не пройдёт в опубликованное число.
