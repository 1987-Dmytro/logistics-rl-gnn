---
type: decision
id: dec-2026-07-21-phase6b-obs
date: 2026-07-21
status: accepted
tags: [decision, congestion, observability, gnn, phase6b]
---

# 0005 — Congestion observability (Phase 6b · Step 0)

**Context:** [[0004-dynamics]] showed RL is OOD under congestion (free-flow-trained, it reacts only
through the feasibility mask). Step 0 is plumbing: the policy SEES time/traffic. WITHOUT changing
training; retraining (POMO) is Step 1.

## What was done (3 congestion channels into the model)

1. **the encoder's edge_attr** — `travel_time(i,j, cur_time)` of the ACTIVE travel model (a snapshot at
   cur_time) instead of a fixed free-flow value. Under FreeFlow == the previous value (parity). A closure
   (inf) → a large finite "very slow" before normalisation (no NaN). The edge dimension is unchanged (1).
   **Precisely:** the per-instance max normalisation `tm/tm.max()` CANCELS the diurnal multiplier (c
   is the same across the city for one snapshot → `t0·c/(t0.max()·c)=t0/t0.max()`), so edge_attr
   carries **incidents** (local, they survive the ratio), while rush hour reaches the model through the
   time-context (channel 3). Putting the diurnal INTO the edges = changing the normalisation (a fixed
   free-flow reference) — that is a Step 1 feature decision, NOT Step 0 (it would break clean parity).
2. **node_congestion** — a new node feature (column 8): the max over active incidents of the node
   contribution (`Incident.at_node`, the finite sentinel `_CLOSED_LEVEL` on a closure). 0 under free-flow.
   The encoder's `in_dim` goes 7→8.
3. **time-context** — `[sin_h, cos_h, sin_dow, cos_dow]` (the congestion phase) into the decoder context.
   The decoder's `ctx_extra` goes 2→6. Under free-flow it is a constant input without a congestion signal.

obs is extended: `node_features (k,9)` + the key `time_context (4)`. `TravelModel.offset_min`,
`env.abs_minute`, `time_context()`, `node_congestion()` are shared helpers (obs and `build_graph`
compute the same thing). FreeFlow → everything stays neutral.

## Checkpoint compatibility (important)

The old `results/policy_best.pt` (Phase 6, in_dim=7 / ctx_extra=2) is **dimensionally incompatible** —
that is EXPECTED. Retraining in Step 1 (POMO) produces new weights. **Do not load the old checkpoint**;
`scripts/run_dynamic.py` / eval will only work after Step 1 (no guard added — out of scope).

## The regression gate

Under FreeFlow the congestion features are neutral: `build_graph` edge_attr is bit for bit == the old
free-flow norm, node_congestion≡0. overfit-tiny converges to **78.9€** (== the optimum `[0,1,2,0]`) —
the plumbing did not break the statics. A no-NaN forward is verified on CongestionTravel WITH A CLOSURE
(the only new risky path: inf→sentinel). Tests: `tests/test_congestion_obs.py` +
`test_model.test_overfit_tiny_cost_drops` (which runs free-flow through the new code).
