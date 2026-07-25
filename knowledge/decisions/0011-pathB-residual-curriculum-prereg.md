---
type: decision
id: dec-2026-07-22-pathB-residual-prereg
date: 2026-07-22
status: accepted
tags: [decision, prereg, path-b, residual, curriculum, rl, cvrptw, phase6b]
---

# 0011 — Path B: residual curriculum, PRE-REGISTRATION (Phase 6b)

**This is a pre-registration: the gate, the selection rule and the kill criteria are fixed BEFORE the code
and BEFORE the run.** It is committed as a SEPARATE commit BEFORE the training code — the git timestamp is
the guarantee that the target was not tuned to the result. Post-hoc edits of this file are forbidden (only a
new decision).

**Context:** [[0010-phase6b-ablation-latency-niche]] closed negatively — the reachable RL start
(a single greedy decode) on a re-plan is WORSE than greedy (865.5 vs 851.5, winning 6/25). Path B is
the only route to "RL on quality": fine-tune the policy ON the residual distribution so that its
**single decode** genuinely beats greedy on a re-plan. Time-boxed, one attempt, an honest gate.

## THE GATE (the primary success criterion — pre-registered)

On the **25 events of the 0004 harness** (`run_dynamic.iter_events`, `generate_instance` seeds 0–4 × 6
events) the **single greedy decode** `rl_raw` beats `greedy_raw`, paired over events:

> **median Δ (rl_raw − greedy_raw) < 0 € AND wins(rl_raw) > 12/25** (i.e. ≥ 13 of 25).

It is computed by `python scripts/run_ablation.py --ckpt results/policy_pomo_residual.pt` — by reading
`analysis.start_rl_vs_greedy`: `median_delta_eur < 0` AND `a_wins > 12`. The metric is DETERMINISTIC
(a raw greedy decode, without a wall-clock confound) → a reproducible pass/fail. `run_ablation` is NOT
modified — the gate is a read of an already existing line.

## The checkpoint selection rule (pre-registered)

The checkpoint submitted to the gate is **best-by-val-residual**, where the val-residual cost = the **single
greedy decode** (the very quantity the gate reads) on a FIXED held-out pool of residual states
(≥48, seeds disjoint from the gate). Selection makes NO reference to seeds 0–4. The anti-forgetting axis
(full congestion statics) is MONITORING in the log only, NOT a selection criterion. A warm-start floor: the
deployment ≥ the single decode of congestion-best on the same pool (a zero outcome = the warm start).

## Kill criteria (pre-registered)

- **patience = 15** epochs without a val-residual improvement → early-stop (best-by-val is kept).
- **wall time ≤ 36 h** (the NJ server, per [[train-on-server]], `TRAIN=scripts/train_residual.py`).
- **ONE attempt.** No second try, no retune-and-rerun after looking at the gate. A failure is the valid
  outcome "Path B does not take the gate".

## Seed disjointness (gate validity — enforced by a test)

**Held out BY SEED (the realisation of demand/windows), not by node_id.** All instances come from ONE
real pool of Augsburg pharmacies (prohibition #5 — real data; `generate_instance` by construction
takes the full snapshot, node_ids are identical across seeds, demand/windows differ). Node-id disjointness
is therefore physically impossible and is not a held-out — as everywhere in this project (Step 1/refit:
train/val/test = seed ranges, not node slices). The seed ranges are pinned (there are NO intersections):

| set | source | seeds |
|-----|--------|-------|
| GATE / deployment full-static | `generate_instance` | 0–9 (gate: 0–4) |
| full-static val (anti-forget, monitor) | `InstanceSampler` | 1_000_000–1_000_063 |
| residual-train (base) | `InstanceSampler(62,62)` | ≥ 3_000_000 |
| residual-val (the selection pool) | `InstanceSampler(62,62)` | 4_000_000–4_000_047 |

The residual base = `InstanceSampler(n_range=(62,62))` (the full set of 62 pharmacies, held out by seed via
demand), NOT `generate_instance` directly: the latter reloads the snapshot from disk (~4 s per call) →
unusable for thousands of residual constructions per run. It is the same snapshot as the gate (the cache
is loaded 1 time); geometry/windows are shared, the demand realisations differ per seed. The gate stays on
`generate_instance(0–4)`.

Residual training never touches seeds 0–9. The test `test_residual_seed_disjoint` asserts: (a) the
train/val-residual seed ranges do not intersect {0–9}; (b) the demand vectors of `generate_instance(residual)`
≠ `generate_instance(gate)` (the realisations are held out, the geometry is shared — a real city).

## Design (choices disclosed in advance)

- **A residual = a fresh CVRPTW** (`residual_instance`: depot + unserved + urgent, windows shifted).
  POMO works UNCHANGED — `feasible_starts` on a residual env = "K allowed NEXT nodes";
  `_decode`/shared-baseline/`train_batch` are reused as they are.
- **The prefix rollout is GREEDY** (not the policy) — deliberately: `iter_events` takes served from the greedy
  execution, so the greedy prefix matches the gate's served distribution; a policy prefix would train on a
  DIFFERENT distribution and could fail the gate "for the wrong reason".
- **Progress frac ∈ [0.2, 0.8]** — now_min is chosen so that exactly round(frac·n_cust) are served
  (by the finish times of the greedy execution). Then ONE event (traffic/urgent/breakdown) at now_min.
- **The residual base is full-62** (`InstanceSampler(62,62)`, cached; see the seed table) — it matches the
  gate's size (no size gap).
- **A 50/50 mix**: 50% full congestion episodes (`InstanceSampler`, the congestion-best distribution
  — anti-catastrophic-forgetting) + 50% residual episodes. The warm start = `policy_pomo_congestion.pt`;
  a new file `results/policy_pomo_residual.pt` (best.pt/congestion/refit are NOT touched).
- Guards as in the refit: |g|>0 (a freeze guard), entropy, no NaN, the mem gap, a freshness hash; the residual
  generator rejects degenerate states (< 2 allowed starts → resample).

## Disclosed train↔test gaps (pre-committed, NOT post-hoc excuses)

1. Train-residual = **one** event at a random point of progress; the gate (0004) includes
   **accumulated multi-event** states (a stream of 6). The distribution shift is known in advance.
2. `frac ∈ [0.2, 0.8]` is the assumed range of the served share; the gate is not tied to frac.

## Disposition

- **Pass** (the gate is taken) → decision 0012: promote `policy_pomo_residual.pt`, re-run the ablation
  (the 0010 harness) with the new checkpoint (secondary: did the tight-budget picture move).
- **Fail** → decision 0012: "Path B does not take the gate" — an honest closure of the "RL on quality" thesis.
  No retry. In either outcome we re-run the ablation for the full picture.

## Tests (enforcing the pre-registration)

seed disjointness (train/val-residual ∩ {0–9} = ∅); the residual is feasible + a non-empty pending pool +
≥2 starts; POMO multistart works on a residual; val-residual = a single decode (== the gate metric);
the ~50/50 mix (seeded); both axes in the log (residual + full); smoke: residual cost↓ and full does NOT degrade.
Links: [[0010-phase6b-ablation-latency-niche]] · [[0007-phase6b-congestion-training]] · [[0006-pomo-static]].
