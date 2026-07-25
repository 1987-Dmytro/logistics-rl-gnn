---
type: decision
id: dec-2026-07-22-inference-search
date: 2026-07-22
status: accepted
tags: [decision, inference, search, pomo, portfolio, dynamic, cvrptw, phase6b]
---

# 0008 — Inference search (Phase 6b · Step 3)

**Context:** [[0007-phase6b-congestion-training]] closed training under congestion, but with an honest
ceiling — **under dynamics RL does NOT overtake greedy** (0004 re-plan: RL +1.6% WORSE than greedy,
event-dependent), and the static win transfers weakly to a residual. Step 3: **NO training,
decode only** — inference search on the congestion-best checkpoint (sha `24c8cfb0607235f8`, the same as in
0007). The goal: cut the static gap and **close the dynamic floor** (make RL ≥ greedy).

## What was built

1. **A batched decode** (`decoder.logits_batch`, `policy.sample_k`): `sample_k(K, temperature)` —
   K stochastic rollouts, **one encode + a batched decode over K** (the network runs vectorised, one
   forward per step). Its own `torch.Generator` → determinism per seed (the global RNG is untouched);
   `temperature>0`. Forced starts are NOT imposed (POMO multistart-greedy is a separate candidate).
2. **`take_best`** — the best candidate by the SINGLE `evaluate_solution` (under the same travel).
3. **`PortfolioPlanner`** (`replan/portfolio.py`): candidates = { sample-K ∪ RL-multistart-greedy ∪
   the greedy heuristic } → the best + end-to-end latency. **Guarantee BY CONSTRUCTION: the result ≤
   greedy** — the greedy candidate is byte-identical to the `greedy` method in the table (the same
   instance+travel+fleet+scorer → `min(...) ≤ greedy` identically).
4. **Wiring into the 0004 harness** (`compare_replan(rl_planner=)`, `run_dynamic.run`) + `run_search.py`
   (3 measurements + provenance → `results/search_summary.json`).

## The key design

- **The latency of inference search is env-bound, NOT neural-bound**: the batched decode keeps the network
  ~flat in K, only K× `env.step` per step grows → latency is ~linear in K (K=256 breaks 1s). Hence
  dynamics run at **K≤32** and statics (offline, no gate) at K=128.
- **The guarantee rests on the byte-identity of the greedy candidate** — not on "RL being smart".
- **sample_k is pure temperature sampling** (multistart-greedy is pre-existing, a separate candidate).

## Result (congestion-best, free-flow static + congestion dynamic)

**The K table** (sample-K take-best, full-62 free-flow, **3 seeds** — a latency/quality sweep):

| K | best,€ | vs greedy | lat,ms |
|---|--------|-----------|--------|
| 16 | 830.0 | −0.1% | 114 |
| 128 | 783.4 | −5.7% | 811 |
| 256 | 782.3 | −5.8% | 1588 |

**Static** (full-62, seeds 0–9, free-flow, K=128; the congestion features are neutral — mult≡1, node_cong=0):

| method | € | vs greedy | vs OR 611 |
|--------|---|-----------|-----------|
| greedy | 825.4 | — | +35.1% |
| RL multistart-only (pre-existing) | 785.3 | −4.9% | +28.5% |
| sample-K take-best (new, standalone) | 789.8 | −4.3% | +29.2% |
| **PortfolioPlanner** | **766.1** | **−7.2%** | **+25.4%** |

**Dynamic** (the 0004 harness, 5×6, K=32): RL portfolio **843.9€ vs greedy 851.5 = −0.9%** (in 0007 it was
**+1.6% WORSE**); **the guarantee holds 0/25 violations, the worst Δ +0.00€**; latency **430ms <1s** (OR-Tools
2001ms, ×5). unserved rl 2.0 = greedy 2.0 (OR 2.48).

## Conclusion — honestly (an asymmetry)

- **The dynamic floor is CLOSED (real and new — the main win of Step 3):** the portfolio is ≥ greedy at EVERY
  one of the 25 events by construction; it was +1.6% worse → it is now −0.9% (it takes RL where RL wins and
  greedy otherwise). Latency is intact (430ms <1s). This is the direct lever against the 0004 gap "RL worse
  than greedy on a residual".
- **Statics — modest, carried by the pre-existing multistart:** the discriminator (multistart-only 785.3 vs
  portfolio 766.1) shows that **the new lever, sample-K, added −2.45% on top of multistart** (not zero — the
  "useless" hypothesis is refuted) and that the portfolio beats the previous deployment best.pt (770.4). But
  the heavy lifting is done by the **pre-existing** multistart-greedy, not by inference search. A marginal
  plus, not a breakthrough.
- **OR-Tools is below RL on cost only because of the 2s deadline** (which chokes OR: unserved 2.48 vs 2.0) —
  the headline stays on **the guarantee + latency**, not on "RL beats OR".

## Next

The dynamic floor is closed → an RL rollout (the portfolio) is safe with respect to greedy. If an RL win over
greedy ON QUALITY (not merely ≥) on large residuals is required — Path B (residual fine-tuning,
[[0007-phase6b-congestion-training]]).
Otherwise Phase 6b closes: congestion training (modest) + inference search (the floor closed, statics cut modestly).

## Tests

parity of the batched vs single decode, determinism of sample_k per seed, `temperature>0`, the
portfolio ≤ greedy guarantee on every instance, take_best skipping None candidates, latency logged,
the planner not mutating the input instance. pytest **77 passed**. Provenance/sha — `results/search_summary.json`
(outside git, prohibition #1; the platform is local — RL/greedy quality is deterministic from seed+config+weights).
Links: [[0004-dynamics]] · [[0006-pomo-static]] · [[0007-phase6b-congestion-training]].
