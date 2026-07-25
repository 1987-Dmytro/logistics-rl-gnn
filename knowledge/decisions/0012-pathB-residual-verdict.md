---
type: decision
id: dec-2026-07-23-pathB-residual-verdict
date: 2026-07-23
status: accepted
tags: [decision, path-b, residual, curriculum, rl, cvrptw, phase6b, verdict]
---

# 0012 — Path B: the verdict on the pre-registration (Phase 6b)

**The outcome of the pre-registration [[0011-pathB-residual-curriculum-prereg]]: the gate was NOT taken
(FAIL). One attempt, no retry — as fixed BEFORE the run.** Path B was the last test of the thesis "GNN+RL
delivers quality"; the residual curriculum was supposed to make the policy's SINGLE decode better than
greedy on a re-plan. It did not.

## The run (server base-node, warm start congestion-best sha `24c8cfb0607235f8`)

A 50/50 mix (full-congestion + residual), selection best-by-val-residual (a single decode = the gate metric).
Healthy throughout: |g|>0 (400–780), entropy alive (~0.06–0.10), start_std>0, freshness rotating,
no NaN, val-FULL stable at ~715 (**anti-forgetting held**, g≈0 — the full statics were not traded away).
**Early-stop at epoch 48** (15 epochs without a val-residual improvement). **Selection — epoch 33:** val-RES
**364.2 (g+1.6%)**, val-FULL 713.8 (g−0.2%). The residual ckpt sha `dfe8401cc40d519c` (outside git, #1).

val-residual only slid from +3.07% (the warm-start bar) to **+1.6%** — the RL start on the single-event pool
is still WORSE than greedy, merely a little less so.

## THE GATE (pre-registered, the 25 events of 0004, generate_instance 0–4)

`start_rl_vs_greedy` — a single greedy decode `rl_raw` vs `greedy_raw`, paired over events:

| metric | the gate's requirement | actual | outcome |
|--------|------------------------|--------|---------|
| median Δ (rl−greedy) | **< 0 €** | **+14.84 €** | ✗ |
| rl wins | **> 12 / 25** | **7 / 25** (greedy 16, ties 2) | ✗ |

(mean Δ +11.98€.) **Both halves failed → FAIL.** For comparison — the same quantity on the warm start
(congestion-best, [[0010-phase6b-ablation-latency-niche]]): median +9.6€, 6/25. Residual training
SHIFTED rl_raw by exactly −2.1€ (865.5→863.4) — within the noise, and by the paired median even WORSE.

## Conclusion — honest (leading with the real lever)

- **The training improved the WRONG distribution.** val-residual (the single-event pool) slid +3.07%→+1.6%,
  but on the gate (**accumulated multi-event** states) the transfer did NOT happen: rl_raw stayed at 863.4,
  7/25 paired. Exactly **gap #1 disclosed in 0011** fired (train = one event at a point of progress; the
  gate = a stream of 6). The selector was directionally valid (pre-flight: concordant in sign), but the small
  win on an easy proxy did not reach the hard gate.
- **Not overfitting and not forgetting:** val-FULL held at ~715 (the anti-forgetting mix worked), training was
  healthy (|g|, entropy, freshness). Path B trained honestly — the single RL start on a re-plan simply never
  overtook greedy.
- **The ablation re-run (secondary, the same ckpt):** the 0010 picture did not move — **there is no niche**
  (rl_raw 863.4 vs greedy_raw 851.5; `rl_polish` loses to `greedy_polish` @50ms med+8.6€ 8/15, a coin flip at
  100/200/500). The new checkpoint does not change the tight-budget conclusion.

**Phase 6b outcome — closed.** Through Path A (0007), inference search (0008), polish (0009), the ablation
(0010) and now residual training (0012): **on QUALITY RL does not beat the classics** — at convergence it
draws, under a budget/a single decode it loses. The only durable contribution of GNN+RL is
**the latency of an instant answer vs OR-Tools** (2001ms→14ms), not quality vs greedy. The thesis "GNN+RL
delivers quality" is not confirmed on the real Augsburg — honestly, with a gate fixed BEFORE the run.

## Disposition

**No retry** (0011). The gate was not taken per the pre-registration — a valid negative closure, not grounds
for retune-and-rerun. `policy_pomo_residual.pt` is NOT promoted; the deployment models stay as they were
([[0006-pomo-static]] `policy_pomo_best.pt`; congestion `policy_pomo_congestion.pt`). Next — if anything is
continued, it is not "RL on quality" but engineering of the latency layer (the instant RL start as an anytime
candidate in a portfolio where quality is carried by multistart+polish — which already exists in
[[0008-phase6b-inference-search]]/[[0009-phase6b-local-search-polish]]).

## Provenance

residual ckpt sha `dfe8401cc40d519c` · warm start `24c8cfb0607235f8` · the gate result
`results/ablation_residual_gate.json` (400 records, tag `simulated-on-real`, outside git). The run:
early-stop ep48, selection ep33, server base-node. Links: [[0011-pathB-residual-curriculum-prereg]] ·
[[0010-phase6b-ablation-latency-niche]] · [[0007-phase6b-congestion-training]].
