---
type: decision
id: dec-2026-07-21-dynamics
date: 2026-07-21
status: accepted
tags: [decision, dynamic, congestion, replan, cvrptw, latency, phase7]
---

# 0004 — Dynamics + re-plan: the "after" on reaction speed (Phase 7)

**Context:** dec-0001 §4 — online re-planning on jams/breakdowns/urgent orders. The Phase 7 goal:
re-plan latency ≪ an OR-Tools re-solve **at comparable cost** (dec-0001 §5).
The Phase 3 travel interface is reused (drop-in) — the env was NOT rewritten.

## What was built

1. **CongestionTravel** (`env/travel.py`, the same `time(i,j,at)` interface):
   `t = t0·c(dow,h) · (1 + Σ_k I_k)`, `h = 08 + (offset+at)/60`. Incidents are **geometric**
   (centre coordinate + radius) → they survive an instance slice. Parity: `c≡1` + 0 incidents ⇒
   exactly FreeFlow (tested). `c(dow,h)` is an urban profile calibrated against the shape of the
   TomTom Augsburg curve (peaks 08/17), tagged `simulated-on-real` (`config/congestion.py`).
2. **Events** (`env/events.py`): traffic (an incident), breakdown (−a vehicle, its stops → the pool),
   urgent (a pharmacy with a narrow window); the smooth diurnal is NOT a trigger. A seeded stream.
3. **Drop-in without rewriting the env**: `DynamicVRPEnv(VRPEnv)` overrides `_load` → travel
   survives the internal `reset()` (rollout/greedy reset the env). The base env is untouched.
4. **evaluate_solution(travel=…)**: congestion-aware time (backward-compatible, free-flow by default).
5. **replan/compare.py**: re-plan the residual (depot + unserved + urgent, windows shifted into the
   event's base) with each method; latency (warmup + median, end-to-end) + quality (the single scorer).

## The "after on reaction speed" table (2 seeds × 6 events, OR-Tools deadline 2s)

`residual re-plan, mean over events · results/dynamic.json (outside git)`

| method   | latency (median) | cost,€ | on-time,% | unserved | × slower than RL |
|----------|------------------|--------|-----------|----------|------------------|
| greedy   | 7.8 ms           | 446.1  | 100.0     | 0.00     | 0×               |
| RL       | 20.3 ms          | 450.6  | 100.0     | 0.00     | 1×               |
| OR-Tools | 2001 ms          | 487.3  | 100.0     | 0.60     | **98×**          |

## Conclusion (the table plus an honest nuance — we do NOT overclaim)

- **THE MAIN GATE is taken and robust: RL reacts ×98 faster than OR-Tools (20ms vs 2s), sub-100ms
  in absolute terms.** This is a forward pass without search — the target property of dynamics (dec-0001 §5).
- **Quality is event-dependent, NOT an RL win.** OR-Tools remains the strongest optimiser on
  LARGE residuals (n=56: OR 548€ < RL 673€ — RL is worse). The aggregate OR-Tools loss in the table
  comes from **static-snapshot pessimism**, NOT from the deadline (verified: 2s→8s moves OR 548→542€,
  already converged): OR-Tools cannot see time-dependency → the snapshot freezes congestion at the
  event's peak hour (ratio ×1.46 mean, ×2.86 max, with no incident decay or diurnal decline) → it
  sometimes drops a stop that is reachable under the true time (+200€), inflating the mean cost.
  RL ≈ greedy.
- **RL reacts THROUGH feasibility, not through congestion features**: `build_graph` feeds free-flow
  edges and the policy was trained free-flow (Phase 6) → under congestion RL is **out of distribution
  (OOD)**. It does not "drive around jams", it merely respects the time-dependent mask. Hence RL is
  worse on large residuals. Congestion features + fine-tuning on residuals → Phase 8.

## Simplifications (ponytail, explicit)

- **class(e) is collapsed into one urban profile** — the snapshot's OD matrix has no per-segment class
  (`with_graph=False`); the TomTom Traffic Index is itself city-level. Upgrade: per-edge on the graph.
- **re-plan = re-optimisation of the remaining stops from the depot** with a fresh T_max/vehicle from
  the event time (the standard periodic re-optimisation of dynamic VRP; the env does not support
  mid-route continuation). Not a "continuation from the middle of a leg".
- **the OR-Tools static snapshot = congestion at the moment of the re-plan** (realistic for a static
  solver: the dispatcher takes the current traffic) but pessimistic (it cannot see the decline) → see
  the nuance above.
- **the partial state is tied to the timeline of the executed greedy plan** (`served_by`), not to
  chance and not to a re-simulation after every re-plan.

## Alternatives rejected

- **rewriting VRPEnv for parallel vehicles in wall-clock** — a large rework of the core for the sake of
  mid-route continuation; periodic re-optimisation gives the same latency/quality measurement cheaper.
- **a snapshot on average/forecast congestion** — it would give OR-Tools a better static chance, but
  that is already forecasting (Phase 8 backlog), not an honest "here and now" static baseline.
- **claiming "RL beats OR-Tools on quality"** — false (event-dependent, OR is stronger at large
  n; the aggregate is skewed by snapshot pessimism). We publish latency as the headline and quality as is.

**Reproducibility (prohibition #4):** **both latency AND quality of OR-Tools are wall-clock-dependent**:
GLS churns neighbourhoods until the deadline → faster/idler hardware fits more iterations → a DIFFERENT
route → different cost/on-time/unserved (verified by an adversarial review: the same seed+config, 5 repeats
→ reward −589.14 vs −587.45). Only RL/greedy quality is deterministic from seed+config (+ a fixed
checkpoint hash). All numbers are median/mean on fixed hardware; the provenance (hash of `policy_best.pt` +
torch/numpy/ortools versions + platform) lives in `results/dynamic.json` (prohibition #4 for artefact
traceability). Tests: parity with FreeFlow, env_checker under congestion, events (traffic/breakdown/urgent),
a feasible re-plan, THE MAIN GATE (RL latency orders of magnitude below OR).
