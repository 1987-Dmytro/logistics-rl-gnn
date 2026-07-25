---
type: decision
id: dec-2026-07-22-local-search-polish
date: 2026-07-22
status: accepted
tags: [decision, local-search, polish, portfolio, dynamic, cvrptw, phase6b]
---

# 0009 — Local-search polish (Phase 6b · Step 3.5)

**Context:** [[0008-phase6b-inference-search]] closed the dynamic floor (portfolio ≥ greedy) and
modestly cut statics (sample-K −2.45% on top of multistart). Step 3.5: **NO training** — a classic
neural constructor + polishing the decoded routes with local search. The hypothesis: polish pulls
quality up to OR-Tools. Checkpoint congestion-best (sha `24c8cfb0607235f8`, the same as in 0007/0008).

## What was built

1. **`env/scoring.py::check_feasible`** — hard feasibility (cap/TW/T_max/fleet) as a **mirror** of the
   `evaluate_solution` time-walk (without a second cost logic). Strictly as `env._feasible` —
   **including the per-customer return check** `t_after + travel(j,0) ≤ T_max` (see "Correctness").
2. **`replan/local_search.py::polish`** — 2-opt + Or-opt(1–3) intra, relocate + swap inter.
   Cost/feasibility come ONLY from `evaluate_solution`/`check_feasible`. **A full eval of every
   candidate** (a complete re-evaluation → correct under time-dependent congestion: a reversal/move shifts
   every downstream time, delta-cost is wrong). First-improvement, looping until convergence or `budget_ms`.
   **Invariant: the result ≤ the input**. The operators work over existing slots (vehicles_used never grows).
3. **Integration**: `PortfolioPlanner` polishes the top-M within a shared budget (the original candidates stay
   in the pool → the ≤ greedy guarantee holds); the static eval polishes every start to convergence. `run_polish.py`.

## Correctness (adversarial review, Ultracode)

A 5-lens workflow (never-worse · feasibility-looser-than-env · operators · determinism ·
edge cases) → verify. **1 confirmed bug**: `check_feasible` checked the T_max return only on the
final edge, while the env checks after EVERY customer. Under **asymmetric** (non-metric) OSM travel
(one-way streets, geometry) polish could accept a cheap 2-opt neighbour that is env-infeasible (from an
intermediate node there is no time to return) and **return it**. Fixed (per-customer return = env) + 2
regression tests (the oracle + the money path). Operators/determinism/copy-safety came out clean. pytest
**99 passed**.

## Result (static to convergence, budget 30000ms; dynamic in-budget 400ms)

**The polish decomposition** (full-62, seeds 0–9, free-flow):

| start | raw € | polished € | Δ polish |
|-------|-------|-----------|----------|
| greedy | 825.4 | **652.2** | −21.0% |
| RL-multistart | 785.3 | **658.9** | −16.1% |
| sample-K | 789.8 | **650.9** | −17.6% |

**The static gap**: the polished portfolio **631.6€** → vs Step 3's 766.1 **−17.6%**, vs OR-Tools 611.1 **+3.4%**
(it was +25.4% in 0008). **Dynamic** (0004, 5×6): RL portfolio+polish **827.3€ vs greedy 851.5 (−2.8%)**,
**the guarantee holds 0/25 violations**, latency **689ms <1s** (OR 2001ms, ×3).

## Conclusion — honest (leading with the main point)

- **Polish ERASES the advantage of the neural constructor.** greedy→652.2, RL→658.9, sample-K→650.9 — **every
  start converges into a ~650–659 window**, the RL edge is **+1.0%** (after polish RL is even slightly WORSE
  than greedy). Classic local search is the dominant lever; the RL start yields no win. This is the asymmetry
  of [[0008-phase6b-inference-search]] one level deeper (there the pre-existing multistart carried it, not
  sample-K; here it is polish, not the policy).
- **+3.4% to OR-Tools is an achievement OF POLISH, not of RL.** The same result comes from a greedy start; it
  does NOT prove the GNN+RL thesis. In absolute terms, though, it is the largest shift of the phase: the gap
  to the optimiser went +25%→+3.4%.
- **The dynamic −2.8% is a confound** (a polished portfolio vs **unpolished** greedy). The honest framing:
  a deployment comparison (portfolio+polish vs cheap greedy) + a by-construction guarantee 0/25 + latency
  <1s. I do NOT attribute the −2.8% to the policy. 689ms is the median of a local machine (the tail under
  load may exceed 1s).
- **Convergence was verified**: a probe on the heavy seed1 (greedy→634.7 in ~16s, flat at 20/40/80s);
  the 8000ms run was budget-bound (portfolio 643.2) → re-run at 30000ms (631.6, converged).

## Next

Statics are nearly at OR-Tools (+3.4%) — the ceiling of construction+polish is close. The open question for the
project's thesis: **is there a niche where GNN+RL is not levelled by polish** (e.g. very large residuals /
a hard realtime budget where polish cannot converge in time). Otherwise the honest outcome of Phase 6b: latency
(×3–134) is the real contribution of RL; on quality the classic methods (multistart, local search) dominate.

## Tests

check_feasible (env parity + the cap/TW/T_max/fleet boundaries + agreement + **asymmetry** + the money path),
the operators (customer conservation + depot ends), polish (never worse than the input across 6 seeds, feasibility,
determinism at convergence, the budget, parity with a brute-force optimum), portfolio+polish ≤ greedy guarantee.
Provenance/sha — `results/polish_summary.json` (outside git, prohibition #1).
Links: [[0004-dynamics]] · [[0006-pomo-static]] · [[0008-phase6b-inference-search]].
