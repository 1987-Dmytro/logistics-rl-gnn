---
type: decision
id: dec-2026-07-21-pomo-static
date: 2026-07-21
status: accepted
tags: [decision, pomo, reinforce, training, cvrptw, phase6b]
---

# 0006 — POMO on statics (Phase 6b · Step 1)

**Context:** [[0003-phase6-training]] delivered REINFORCE + the Kool rollout baseline (804€, beats greedy).
Step 1 replaces it with POMO (multi-start + a shared baseline): simpler (no frozen copy/t-test),
more robust to a degenerate baseline. Free-flow statics; dynamics are Step 2. Reuses [[0005-phase6b-congestion-obs]].

## What was built

1. **train/pomo.py — POMOTrainer.** Per instance: encode ONCE → N trajectories from N DIFFERENT
   first nodes (sample). The shared baseline `b = mean_N(cost_i)`; `advantage_i = cost_i − b`
   (centred, WITHOUT std normalisation — the POMO baseline is not degenerate, the all-depot pathology
   of Phase 6 is gone). `loss = mean(adv·Σlogπ)`, sign `+Σlogπ` → descent ↓cost (== the Phase 6 fix;
   a flip → cost grows). The rollout baseline / paired t-test / frozen copy are removed (the shared
   baseline replaces them, `p=nan` disappears). Adam + grad-clip@1 (the clip normalises ‖grad‖ → the
   step stays ~lr; it binds at every step — that is OK, not a bug).
2. **Starts — only those allowed at step 0** (`feasible_starts`: mask==1, thinned to max_starts).
   Forcing an infeasible start would give `log_prob=−inf → NaN` — hence the guard. `<2` allowed starts
   → the instance is skipped. **The forced step is EXCLUDED from the gradient** (prob=1 for an imposed
   action, the POMO canon — Kwon; an adversarial review caught the deviation). At inference
   `multistart_greedy` forces every start → π(·|s₀) never picks the start anyway, so the term would have
   been useless. We keep the estimator unbiased/canonical.
3. **Gradient accumulation:** `(loss/b).backward()` per instance → memory O(1 instance), not O(batch)
   (important for full-62). encode survives reset (a static graph; instance_fn ignores the seed).
4. **Inference — multi-start greedy** (`multistart_greedy`): a greedy decode from N starts, take the best.
   Fast/parallel. Validation and the final eval use it.
5. **config/pomo.py** (N/epochs/batch/lr/clip), **scripts/train_pomo.py** (`--smoke`/full;
   the OR-Tools val reference is injected from the script, not in `__init__` → tests run without ortools).
6. **Part 0 — the `congestion_multiplier` edge channel** (see [[0005-phase6b-congestion-obs]], it closes
   that nuance): `edge_attr [E,2]` = [the travel norm (topology, erases a uniform diurnal),
   `travel/free_flow=c·(1+ΣI)` (the diurnal/incident are VISIBLE per edge)]. Under free-flow channel 1 ≡1
   (neutral for statics) yet it makes the diurnal visible for Step 2. diag→1, a closure inf→cap=10.

## Why NOT 8x augmentation

The POMO paper adds ×8 euclidean augmentations (reflections/rotations of coordinates) — we have a **real
Augsburg travel matrix** (non-euclidean: road asymmetries, OD from OSRM), and rotating coordinates does NOT
preserve the times. The augmentation would be invalid. Multi-start (N starts) is the only source of
diversification; that is enough for a shared baseline.

## Compatibility / the metric

- The weights are a new `results/policy_pomo_best.pt` (outside git). The old Phase 6/6b-Step 0 checkpoints
  are incompatible (edge_dim 1→2, in_dim/ctx already grew in 0005) — retraining is the very point here.
- **The before/after stays valid:** the "after" = multi-start greedy on full-62/seeds 0–9 (THE SAME
  instances as [[0002-baselines]]), the single `evaluate_solution`. Compared against greedy 825€ /
  OR-Tools 611€.
- The Step 1 goal: a static gap to OR-Tools ≤ the previous +31.6% (REINFORCE 804€). **The full run happens
  on the server**; the smoke run (Mac) only demonstrates the mechanism (cost↓, |g|>0, start spread).

## Result of the full run (server base-node, 100 epochs, seed=0)

**The "after" = 770.4€** (full-62/seeds 0–9, multi-start greedy). It beats greedy 825€ by **−6.7%**,
the gap to OR-Tools 611€ = **+26.1%**. The goal is met: **+31.6% → +26.1%** (−5.5 pp vs REINFORCE
804€; and the lead over greedy is deeper: −6.7% against −2.6%). The run took 6.3h on a Ryzen 9 9900X
(CPU, OMP=12), 100 epochs × 3.8 min, healthy throughout (start_std alive, H 1.76→0.12 not 0, |g| binds
the clip, no NaN). Config: batch=16, starts=16, steps/ep=30, lr=1e-3, clip=1.0, n=40–62. Artefacts/
provenance (torch/tg/ortools versions + hash `79e6ffb4…`) — `results/pomo_summary.json` (outside git).
Honestly: val (n=40–62 subsets) stayed at the heuristic (gap_greedy ~0), but full-62 multi-start
greedy gives −6.7% vs greedy — the quality shows on full instances. OR-Tools 611€ is not taken yet;
congestion/dynamics (Step 2) + longer/wider training is the way forward.

## Refit (the anti-overfit protocol, Step 1·refit — 2026-07-22)

The previous run (100 epochs, β=0, no early-stop) gave 770.4€, but without a held-out val/test one could
not rule out memorisation. The refit added the protocol: a seed split train(0–1M)/val(64)/test(64), disjoint,
an entropy bonus β=0.01, early-stop best-by-val (patience=15), an instance freshness hash (the RNG is fresh
every epoch — test+log), a train probe (a train-side gap apples-to-apples with val). Weights →
`policy_pomo_refit.pt` (770.4€ `policy_pomo_best.pt` is NOT touched). The run was on base-node, early-stop
at epoch 23 (best=epoch 8), `TRAIN_DONE=0` (a clean exit; `PIPESTATUS[0]` instead of `$?` — otherwise a
false 0 as in the previous session).

**Generalisation (all gap-to-greedy):** train −3.7% · val −3.1% · TEST −1.6% · **deployment (full-62) −5.1%**.
All of the same sign/order, train≈val (Δ0.6pp) → there is **NO memorisation**; deployment (−5.1%) is even
BETTER than val/test → the size extrapolation (n=40–62 → full-62) went the favourable way (the advisor's
nuance is resolved). Val is discriminative (a peak of −3.1%, not "≈the heuristic" as before) → the
`val_n_range` lever was not needed.

**Deployment:** the "after" is **783.2€** (gap to greedy −5.1%, **OR-Tools +28.2%**). Provenance (seed+config+
versions+sha256 `9a06ee7f…`) — `results/pomo_refit_summary.json` (outside git).

**Verdict — honestly:** the refit **did NOT beat** the previous run (783.2€ vs 770.4€, +1.7%). The early peak
(epoch 8), early-stop@15 and β cut off the depth that the 100-epoch run gained over its extra 77 epochs.
**The value of the refit is not a new number but the VALIDATION:** the RL win over greedy (−5%) is real and
generalises (train/val/test/deploy agree), not an overfitting artefact. The previous `policy_pomo_best.pt`
(770.4€) remains the deployment model (NOT promoted). To beat 770.4 under the protocol — a wider patience /
a lower lr (val wobbles 605–630, |g| spikes 914/1039 → the step is large) / longer training; or accept the
validation and move to Step 2 (dynamics during training).

## Smoke (Mac, 5 epochs, illustrative)

`train 333.8→249 · |g|>0 (the clip binds) · start_std 50→12 (the baseline is alive) · H 1.87→0.79 (no collapse)`.
val saturates ≈the heuristic on small instances → the learning signal is taken from train. The smoke run's
"after" (842€) means nothing; the real number comes from a full run.

## Tests (tests/test_pomo.py)

The shared-baseline advantage is not degenerate (RAW cost std>0), |g|>0, cost↓ (train), determinism per seed,
no entropy collapse, no NaN on a REAL sampled instance. Part 0: free-flow parity
(channel 0 bit for bit, channel 1 ≡1), both channels finite under a closure, the diurnal visible in channel 1.
