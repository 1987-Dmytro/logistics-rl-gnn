---
type: decision
id: dec-2026-07-21-baselines
date: 2026-07-21
status: accepted
tags: [decision, baseline, cvrptw, ortools, greedy, phase4]
---

# 0002 — The "before" baselines: OR-Tools + greedy (Phase 4)

**Context:** before RL we need an honest baseline (prohibition #3) on IDENTICAL instances and
fixed seeds — the "before" number to compare "after" against. Metrics from
dec-0001 §5: distance, time, vehicles, on-time, unserved.

## The "before" table (10 seeds, mean±std)

`K=8 Q=80 T_max=240min · snapshot augsburg_20260720 · OR-Tools 9.15.6755 @ 30s/seed · seeds 0–9`

| method   | distance,km   | time,min        | vehicles    | on-time,% | unserved | reward,€        |
|----------|---------------|-----------------|-------------|-----------|----------|-----------------|
| greedy   | 158.82 ±10.30 | 1157.91 ±108.59 | 7.20 ±0.60  | 100.0     | 0.0      | −825.38 ±62.78  |
| OR-Tools | 144.75 ± 6.15 |  671.29 ± 71.40 | 6.30 ±0.46  | 100.0     | 0.0      | −611.14 ±39.65  |

OR-Tools is better on every axis: −9% distance, −42% time (it minimises waiting), −0.9
vehicles, reward +26%. Both serve 62/62 with no lateness. The summary (config+seeds+versions):
`results/baselines.json` (outside git).

**Decision:**
1. **A single scorer** `env/scoring.py:evaluate_solution(routes, instance, cfg)` — one
   reward formula for the environment AND the baselines (VRPEnv switched to it, the regression
   is green). Comparison on a shared measure.
2. **greedy** = VRPEnv under the nearest-feasible policy (cap/TW/T_max masking comes FROM
   the environment — one source of feasibility, no divergence). Deterministic per seed.
3. **OR-Tools CVRPTW**: arc-cost = c_d·dist; the cost of TIME via the Time-dim span cost
   (span = driving+service+waiting = the time from the scorer) → OR-Tools minimises the
   SCALED reward rather than a proxy; TW windows [e, floor(l)]∩T_max, transit=ceil
   (conservative towards late), Capacity Q, disjunction penalty = p_unserved (the same
   trade-off). Scoring uses the single evaluate_solution, NOT the OR-Tools objective.

**Why span-cost and not "arc=c_d·dist+c_t·time":** 38/62 stops have e>0 (up to 207 min), so
waiting is a large reward term. A travel-only proxy does not see it → OR-Tools could
scatter the high-e stops and lose to greedy under the true reward. span-cost reproduces the
time term exactly → "OR-Tools ≥ greedy" holds robustly rather than by luck.

**Side finding (fixed):** the synthetic ASSUMED windows (the fallback for ~9 stops without
OSM hours) drew e up to T_max → ~1 real pharmacy became structurally unreachable on seed 1/2/4
(the vehicle starts at t=0, waits until e and cannot return within T_max). The old guard checked
reachability WITHOUT the wait and let this through. Fix: `_synthetic_window` clips e to the budget
`T_max − service − travel(i→depot) − 60s` (after the draw → the rng stream and the feasible windows
do not change; seed 0 is identical); the `_check_feasibility` guard is now wait-aware. Real windows
are untouched (prohibition #5).

**Alternatives rejected:**
- a travel-only OR-Tools objective — it underestimates waiting (see above);
- cherry-picking feasible seeds to get "62/62" — it hides real infeasibility (dishonest);
- excluding unreachable stops as CLOSED — that changes the instance size; clipping the synthetics
  is more minimal and does not cut coverage.

**Reproducibility:** seeds+config+versions in `results/baselines.json`. OR-Tools
(PATH_CHEAPEST_ARC+GLS) is algorithmically deterministic; only the number of iterations inside the
`time_limit` budget varies — the "before" number is tied to the fixed budget, not to wall-clock.
