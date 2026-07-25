---
type: decision
id: dec-2026-07-22-ablation-latency-niche
date: 2026-07-22
status: accepted
tags: [decision, ablation, latency, polish, rl, cvrptw, phase6b]
---

# 0010 — Ablation: the latency niche of RL (Phase 6b)

**Context:** [[0009-phase6b-local-search-polish]] closed with an open question for the thesis: polish
levelled the starts only **at convergence** (~16s at n=62). Is there a regime where the RL START decides —
where polish cannot level things out under a hard realtime budget (or on a large residual)? The ablation
tests that head on. **NO training**, the same congestion-best checkpoint (sha `24c8cfb0607235f8`,
as in 0008/0009).

## What was built

`run_dynamic.iter_events` — the 0004 harness stream extracted (the same 5×6, identical provenance; `run()`
now sits on top). `scripts/run_ablation.py` — at EVERY event 4 systems under a hard **end-to-end**
budget budget_ms ∈ {50,100,200,500} (decode+polish; an overrun = a deadline snapshot, polish is anytime):
`rl_raw` (the policy's greedy decode), `greedy_raw` (the heuristic — **the start control**), `greedy_polish`,
`rl_polish`. One scorer `evaluate_solution`; paired Δ over events (median/wins), as in 0007;
a budget decomposition (construct_ms/polish_ms) + a win-size profile (n_pending). 5 guards: the budget really
binds (unbounded polish > budget), one scorer, determinism at convergence, the event join.
pytest **104 passed**, ruff clean.

## Result (0004, 5 seeds × 6 events, 400 records; the cost is budget-bound → wall-clock-dependent)

| budget | rl_raw | greedy_raw | greedy_polish | rl_polish | rl_pol − gr_pol |
|--------|--------|-----------|---------------|-----------|-----------------|
| 50ms  | 865.5 (18ms) | **851.5 (7ms)** | 839.7 | 851.2 | **Δ̃+10.3€ w5/l18** |
| 100ms | 865.5 | 851.5 | 836.5 | 843.3 | Δ̃+4.4€ w9/l14 |
| 200ms | 865.5 | 851.5 | 830.6 | 832.1 | Δ̃+0.0€ w12/l11 |
| 500ms | 865.5 | 851.5 | 823.1 | 821.4 | Δ̃+0.0€ w12/l11 |

**The start (before polish):** `rl_raw` 865.5 vs `greedy_raw` 851.5 — the RL start is **Δ̃+9.6€ (mean +14.0€)
and wins only 6/25**. The polish contribution grows with the budget (rl medΔ −10.5→−44.8€, greedy −11.9→−30.3€).

## Conclusion — honest (leading with the main point)

- **There is NO niche.** The hypothesis "a better RL start pays off where polish cannot converge" collapses at
  its premise: **the reachable RL start (a single greedy decode) is NOT better than greedy — it is WORSE**.
  865.5 vs 851.5, winning 6/25. There is nothing to polish into an advantage over greedy.
- **Under a hard budget RL loses on BOTH axes.** (1) the start is worse; (2) the decode takes **18ms vs
  greedy's 7ms** → at 50ms only ~32ms of polish remain against greedy's ~43ms — the RL decode steals its own
  polishing budget. The result @50ms: `rl_polish` **loses** to `greedy_polish` (Δ̃+10.3€, l18/25) —
  this is an ANTI-niche, not a niche. The RL latency win existed only vs OR-Tools (2001ms); against greedy —
  greedy is both faster and has the better start.
- **As the budget grows polish levels things out** (200/500ms: Δ̃+0.0€, 12/11 = a coin flip) — confirming
  [[0009-phase6b-local-search-polish]] already under a sub-second budget, not only at the 16s convergence.
- **There is no large-residual niche either:** `rl_polish` wins do NOT concentrate on large instances
  (the median n_pending of wins is 32–41 ≈ the overall 43; max n=60). At noticeably larger residuals — the same coin flip.
- **The caveat = a load-bearing scope (given BEFORE the synthesis).** The verdict concerns ONE decode: the
  realtime regime affords exactly one decode, not the POMO multistart of 0009 (K× decode). Multistart RL (785.3)
  as a START is stronger than greedy, but it is unreachable under a hard budget. The ablation takes **50–500ms**,
  0009 takes convergence at **~16s**; the monotone trend (wins 5→9→12→12, Δ̃ +10.3→+0.0€) ACROSS that gap
  also closes the "but maybe RL wins at an intermediate ~2s".

**Phase 6b outcome (precisely):** RL brings no quality advantage **OVER THE CLASSICS** — at convergence
multistart RL only DREW with greedy ([[0009-phase6b-local-search-polish]]), and under a hard budget a
single decode LOSES (here). This is "the classics dominate on quality", NOT "RL is bad". The real
contribution of RL is an instant answer **vs OR-Tools** (not vs greedy: greedy is both faster and better as a
start). Next — either Path B (residual fine-tuning: make the RL START genuinely better than greedy on the
re-plan distribution), or accept the outcome and do not reopen it.

## Tests

The budget holds (latency ≤ budget+ε, the deadline is really reached), one scorer, determinism
at convergence, `measure_event` emits every system/budget + raw_cost is consistent with the start.
Provenance/sha — `results/ablation_summary.json` (outside git, prohibition #1).
Links: [[0004-dynamics]] · [[0008-phase6b-inference-search]] · [[0009-phase6b-local-search-polish]].
