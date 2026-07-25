---
type: decision
id: dec-2026-07-21-phase6-training
date: 2026-07-21
status: accepted
tags: [decision, rl, reinforce, cvrptw, gnn, collapse, phase5, phase6]
---

# 0003 — RL training (REINFORCE) + the "after" 804€ (Phase 5+6)

**Context:** after the "before" ([[0002-baselines]]) — train a GNN+attention policy and obtain an
honest "after" on IDENTICAL instances/seeds (prohibition #3). The network: GATEncoder (GATv2 ×3,
d=128, encode-once) + a Kool-style AttentionDecoder + VRPPolicy (autoregressive rollout, feasibility
from `env.action_mask`). Training: REINFORCE + a rollout baseline (Kool).

## The "after" table (full-62, seeds 0–9, greedy decode)

`K=8 Q=80 T_max=240min · snapshot augsburg_20260720 · == the Phase 4 "before"`

| method      | reward,€ | gap vs OR-Tools | vs greedy     |
|-------------|----------|-----------------|---------------|
| OR-Tools    | −611.14  | —               | −26%          |
| RL "after"  | −804.1   | +31.6%          | −2.6%         |
| greedy      | −825.38  | +35.0%          | —             |

**The RL "after" −804€ BEATS greedy −825€ (−2.6%)** — the first valid win of RL over the heuristic.
Training stayed healthy for 100 epochs (`|g|` 13–20, `H`→0.24 confident, the freeze guard never
fired). Weights `results/policy_best.pt` (outside git). The config is
`config/train.py:TrainConfig` (100 epochs, 50 steps, batch 32, lr=1e-3, grad-clip 1.0, n∈[40,62]).

## Decision

1. **REINFORCE + a rollout baseline** (Kool): the baseline is a frozen copy of the policy (greedy on
   THE SAME instances, paired). The baseline update uses a paired t-test (`ttest_rel`, one-sided p/2 <
   0.05) on val seeds (disjoint from train). Adam lr=1e-3, grad-clip=1.0 (a Phase 5 finding).
2. **The loss sign is +Σlogπ**, `mean((cost−b)·Σlogπ)`: ∇L=E[(cost−b)∇Σlogπ]=∇E[cost] → descent ↓cost.
   The spec gave `−Σlogπ` (which would raise cost) — rejected, verified empirically (cost fell).
3. **The advantage is NORMALISED** (mean0/std1, detach) — see the collapse below.
4. **The decoder has NO `C·tanh` clip** on the logits (raw `q·k/√d`) — see the collapse below.

## The critical training collapse (root cause = the Phase 5 lr collapse)

Full run #1 froze at epoch 1: `|g|→0`, the "after" was an all-depot 12400€. Diagnostics a/b
(instrumenting one step): the code is healthy at init (loss.requires_grad=True, Σlogπ.grad_fn≠None,
model.training=True) → not (a)/(b) but **(c) a saturation collapse**. Three coupled causes:

- **the tanh clip saturates** — `C·tanh(logits)` at |logits|≫1 gives grad≈0 → the network cannot learn.
- **an unnormalised advantage** — the all-depot baseline = n·200 (an enormous scale) drove the
  softmax into saturation within 1 epoch, before any useful signal appeared.
- **a baseline deadlock** — under all-depot greedy(current)==greedy(baseline) → `ttest_rel`
  returns `p=nan`, the baseline is never updated and the deadlock sustains itself.

**The key link:** the same tanh clip was the root of the Phase 5 "lr=1e-2 → uniform collapse". Back
then it was blamed on lr; in fact it was the clip saturating. One root cause, finally removed here.

**Fix:** (1) remove the tanh clip; (2) normalise the advantage; (3) a runtime freeze guard —
`|g|<1e-6` for 3 epochs in a row → abort the run (not 100 wasted); (4) smoke guards (grad>0,
val moves, the baseline p is not nan, entropy ∈[0.05, log k]) — the collapse used to hide until epoch
100; (5) an entropy bonus `+β·H` in reserve (`entropy_beta=0`, enabled if the softmax over-sharpens
without the clip). Run #2 (fixed) was healthy and produced the "after" of 804€.

## Alternatives rejected

- **Keep the tanh clip and treat lr** — not the root cause (Phase 5 showed that), saturation returns.
- **Leave the advantage unnormalised and lower lr** — it masks the scale; the collapse is slower but the same.
- **Publish the "after" from run #1 (all-depot)** — void (prohibitions #3/#4).

**Reproducibility:** seed+config in `TrainConfig`; the "after" is a greedy decode on `eval_seeds`
0–9 (== the "before"). The guards ensure a silent collapse never reaches a published number again.
