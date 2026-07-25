---
type: decision
id: dec-2026-07-22-congestion-training
date: 2026-07-22
status: accepted
tags: [decision, congestion, training, pomo, dynamic, cvrptw, phase6b]
---

# 0007 — Training under congestion (Phase 6b · Step 2)

**Context:** [[0004-dynamics]] exposed the root cause — RL reacts to events THROUGH the feasibility mask
rather than through congestion features (trained free-flow → OOD under congestion, worse on large residuals).
[[0005-phase6b-congestion-obs]] gave observability (Step 0), [[0006-pomo-static]] gave POMO on statics
(Step 1, free-flow). Step 2: **train POMO under active congestion** so the policy USES the
signal. The operator's choice: **Path A (congestion-static) + a warm start from 770.4€** (residual training,
Path B, is held in reserve should A fail to close the dynamic gap).

## What was built

1. **Vectorising `build_graph`** (`TravelModel.matrix()`): k² Python calls to `travel.time()` choked the
   retrain at k=62 → one shot (FreeFlow→t0; Congestion→`t0·c·(1+Σinc)`, zones by outer-OR, closure union).
   Free-flow is **bit for bit** (regression-safe, a parity test vs element-wise `time()`).
2. **Congestion training** (`POMOConfig.for_congestion` + `POMOTrainer`): warm start 770.4€, lr=3e-4,
   β=0.03; a congestion env factory (dow=delivery, offset+incidents on **t0**, `t_start=offset` →
   active from dispatch, **coverage 100%** of nodes); both reward AND greedy are congestion-aware
   (`travel=env.travel`; free-flow → parity). Weights → `policy_pomo_congestion.pt` (best.pt/refit intact).
3. **A warm-start floor**: the starting val is the bar and the warm start is kept as the ckpt → **the
   deployment is NEVER worse than the warm start** (a zero outcome is a clean finding, not a silent regression).
4. **Before/after under congestion** (`eval_congestion`) + **re-evaluating the 0004 table** (`run_dynamic --ckpt`).

## The key design (deliberate simplifications)

- **encode ONCE at t0** (a congestion snapshot at dispatch) + the decoder's `time_context` runs over the
  steps; reward through `evaluate_solution(travel=)` — full time-dependent time (correct). NOT a
  per-step re-encode (like the snapshot in 0004).
- **The diurnal is nearly invisible to the encoder** (the maths): channel 0's max normalisation cancels a
  uniform `c` (`t0·c/max = t0/max`), channel 1 = `c` is a constant that duplicates `time_context`. So
  **the incidents on t0 are the whole signal** (local, they survive the max norm). Hence the richness of
  incidents is critical (advisor); `≥1` incident on a node + long-lived → coverage 100%.

## Result (early-stop ep30, best-by-val ep15, TRAIN_DONE=0)

**The bar** (free-flow-best under congestion): val 721.1, gap_greedy **+0.8%** — OOD ate the win
(it was −6.7% under free-flow → slightly WORSE than greedy under congestion).

| Axis | After (RL-cong) | RL-cong vs RL-ff | vs greedy | vs OR |
|------|-----------------|------------------|-----------|-------|
| **Statics** (before/after, 32 held-out) | 712.2€ | **−1.7%** | −0.3% | +16.4% (snapshot pessimism) |
| **Dynamics** (0004 re-plan, 5×6) | 865.5€ | **−0.4%** | +1.6% | ≈parity |

- **Generalisation** (gap-to-greedy): train −0.2% · val −0.5% · TEST −0.9% — consistent (train≈val≈test)
  → **no memorisation**; the win transfers to held-out.
- **The ×134 latency is intact** (RL 15ms vs OR 2001ms — the same forward pass).
- Dynamics −0.4% is a **weak but consistent signal, NOT an outlier** (paired over 25 events: median −4.84€,
  16/23 diverging routes in favour of congestion vs 7 against; the largest |d|=+33€ is AGAINST cong, i.e.
  the result is not driven by one outlier). Congestion genuinely shifted behaviour (23/25 routes differ).
- Dynamics: 5 seeds (0–4) vs 0004's 2 → **the absolute numbers differ** (seeds 2–4 are heavier, unserved 2.0);
  the RL-vs-RL/greedy comparison is on THE SAME seeds. Provenance/sha — `results/pomo_congestion_summary.json`.

## Conclusion — honestly

- **Congestion training helped, but MODESTLY, and statics > dynamics.** The static win of −1.7%
  (RL restored greedy parity+ under congestion) **transfers weakly** to a residual re-plan (−0.4%).
- **The 0004 gap "RL worse than greedy on large residuals" barely moved** (+2.1%→+1.6%): a residual
  (depot+unserved+urgent, windows shifted, congestion at the moment of the event) differs from
  static congestion enough that a full transfer requires **Path B (training on residuals)**.
  The advisor's link "Path A → 0004" only holds directionally.
- **Under dynamics RL does NOT overtake greedy** (event-dependent, as in 0004) — no RL win on quality
  may be claimed. The headline stays latency (×134); congestion training is a marginal plus.
- **The floor guaranteed no regression**; `policy_pomo_best.pt` (770.4€ static) is untouched.

## Next

Path B (residual fine-tuning on the re-plan distribution) is the direct lever against the dynamic gap; or a
per-step re-encode (more expensive) — if an RL win over greedy on large residuals is required. Otherwise
Step 2 closes as "congestion training gives a modest plus, dynamic quality stays ≈greedy".

## Tests

matrix parity (vs element-wise `time()`, including closure+the zone boundary), congestion train+coverage,
sampler determinism, **the warm-start floor is not overwritten by a worse epoch**. All green (pytest 70).
Links: [[0004-dynamics]] · [[0005-phase6b-congestion-obs]] · [[0006-pomo-static]].
