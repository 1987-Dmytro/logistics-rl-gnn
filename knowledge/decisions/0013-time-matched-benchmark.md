---
type: decision
id: dec-2026-07-23-time-matched-benchmark
date: 2026-07-23
status: accepted
tags: [decision, benchmark, ortools, anytime, time-matched, cvrptw, phase8, verdict]
---

# 0013 — Time-matched: OR-Tools anytime vs the system (task #15, the final benchmark)

**The last honest test of the deployed system: give OR-Tools THE SAME wall-clock and measure quality on
IDENTICAL instances.** The question: is the system's "latency edge" honest in STATICS, or was the system
merely compared against an under-budgeted OR-Tools. `scripts/run_timematch.py` (NO training): full-62,
seeds 0–9, the same `generate_instance` + the single `evaluate_solution` (prohibition #3), budgets
{0.7,2,5,30}s → an anytime curve. The artefact `results/timematch.json` (outside git, prohibition #1).

## The conflation quarantine (caught by the advisor in Phase 8, repeated here)

The brief asked to "compare with the system's 631.6€ @ 689ms" — but those are **two different settings**:
the system reaches 631.6€ with a polish budget of 30000ms/candidate ×≤3 (`system_metrics.json`) → its
static wall-clock is **≥30s**, while 689ms is the **dynamic** re-plan latency on a residual instance
(where the cost is 827€, unserved 2.0, `polish_summary.json`). We do NOT place the system's point at 689ms:
it sits at its REAL static x (≥30s), y=631.6€; 689ms/827€ goes into a separate field as dynamics. Not one
point "631.6€ @ 689ms".

## Paired discipline (as in [[0010-phase6b-ablation-latency-niche]]), not unpaired σ

The data are PAIRED (the same seeds 0–9 for OR-Tools and the system) → the across-seed σ~44€ is the
common-mode difficulty of the instance and **cancels**. I lead with median-Δ + wins on the shared seeds (the
system's per-seed values were added in `system_metrics.json:per_seed_cost_eur`, a deterministic re-run of
eval_system, parity 631.62 ✓) rather than with an across-seed σ as a noise floor (that would violate the
spirit of prohibition #3 in an outward-facing number).

| OR-Tools budget | cost, € (±std) | wins/10 vs the system | median Δ/seed |
|-----------------|---------------:|----------------------:|-------------:|
| 0.7s | 629.7 ± 44 | 6/10 | −4.9€ |
| 2.0s | 626.1 ± 42 | 7/10 | −10.5€ |
| 5.0s | 625.0 ± 42 | 7/10 | −11.2€ |
| 30s  | 610.5 ± 39 | **8/10** | **−18.6€** |

(The 30s point = 610.5€ — parity with 611.1€ [[0002-baselines]] within the wall-clock jitter of GLS
best-so-far, tol 2€ as in eval_system; the curve is monotone.)

## Conclusion — honest

- **The system has NO static advantage, neither in quality nor in latency.** OR-Tools reaches parity with
  the system's static quality (631.6€) already at **<1s** (6/10, median −4.9€), and by 30s it clearly
  beats it on **8/10** seeds (median **−18.6€/seed**) — while the system spends ≥30s of polish to reach
  631.6€. Time-matched, in statics the classics dominate, as along the whole Phase 6b chain.
- **The only durable edge of GNN+RL is DYNAMICS:** on a residual re-plan event the system reacts
  in 689ms vs OR-Tools' 2001ms ([[0009-phase6b-local-search-polish]] dynamic 5×6), the neural start ~15ms
  ([[0010-phase6b-ablation-latency-niche]], but quality-inferior). Not quality vs greedy, not statics.
- **This agrees with the closed thesis** ([[0012-pathB-residual-verdict]]): on quality RL does not beat
  the classics. Time-matched sharpened it: the system has no static latency edge either — OR-Tools is faster
  to the same price. The before/after case stays honest (the system is −23.5% vs greedy), but NOT
  "faster/better than OR-Tools".

## Disposition

The "Time-matched" section of `docs/final_metrics.md` is emitted by `final_metrics.py` (one owner of the file
— otherwise it gets overwritten; the paired columns come from two durable json files). Tests: the 30s parity,
monotonicity, determinism of the core, `paired_stats`, the paired median (pytest 125 ✓). This is the final
benchmark — no further runs are planned.

## Provenance

`results/timematch.json` (the curve, per-seed OR) + `system_metrics.json:per_seed_cost_eur` (the system,
parity 631.62) — both outside git (prohibition #1). OR-Tools 9.15.6755, seeds 0–9 full-62 free-flow, one
scorer. The code is commit `0d739dd`. Links: [[0002-baselines]] · [[0009-phase6b-local-search-polish]] ·
[[0010-phase6b-ablation-latency-niche]] · [[0012-pathB-residual-verdict]].
