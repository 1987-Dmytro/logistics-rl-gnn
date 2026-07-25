"""Path B — residual-curriculum training (Phase 6b, pre-registration 0011).

Smoke:  python scripts/train_residual.py --smoke   (mechanism: the residual axis moves, full does
        NOT fall, |g|>0, both axes in the log — the number is illustrative, not a gate)
Full: python scripts/train_residual.py            (server per the runbook train-on-server,
        TRAIN=scripts/train_residual.py; ≤36h, patience=15; best-by-val-residual →
        results/policy_pomo_residual.pt). The GATE (0011) is computed SEPARATELY after the run:
        python scripts/run_ablation.py --ckpt results/policy_pomo_residual.pt  → read
        analysis.start_rl_vs_greedy: median_delta_eur<0 AND a_wins>12.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from logistics_rl_gnn.config.pomo import POMOConfig
from logistics_rl_gnn.models.policy import VRPPolicy
from logistics_rl_gnn.train.residual_curriculum import ResidualPOMOTrainer


def _make_log(cfg: POMOConfig):
    """Epoch log line — BOTH axes: val-RES (selection/gate proxy) + val-FULL (anti-forgetting)."""

    def _log(rec: dict) -> None:
        print(
            f"ep {rec['epoch']:3d} | train {rec['train_cost']:7.1f} | "
            f"val-RES {rec['val_res_cost']:7.1f} (g{rec['val_res_gap_greedy']:+.1%}) | "
            f"val-FULL {rec['val_full_cost']:7.1f} (g{rec['gap_greedy']:+.1%}) | "
            f"H {rec['entropy']:.3f} |g| {rec['grad_norm']:.2f} std {rec['start_std']:.1f} | "
            f"es {rec['since_improve']}/{cfg.patience} fr {rec['inst_hash'][:6]}"
        )

    return _log


def _smoke_cfg() -> POMOConfig:
    return POMOConfig.for_residual(
        epochs=6, steps_per_epoch=8, batch=4, max_starts=5, patience=4,
        n_range=(15, 20), val_range=(1_000_000, 1_000_004),
        test_range=(2_000_000, 2_000_004), res_val_range=(4_000_000, 4_000_006),
        ckpt=None,
    )  # fmt: skip


def _report_smoke(hist: list[dict]) -> None:
    """Both axes: residual moves down, full does NOT degrade, the mechanism is alive (|g|>0)."""
    res = [h["val_res_cost"] for h in hist]
    full = [h["val_full_cost"] for h in hist]
    gnorm = [h["grad_norm"] for h in hist]
    print("\n=== SMOKE curves (both axes) ===")
    print("  ep    val-RES    val-FULL   |g|")
    for h in hist:
        print(f"  {h['epoch']:2d}  {h['val_res_cost']:8.1f}  {h['val_full_cost']:8.1f}  "
              f"{h['grad_norm']:.2f}")
    res_moved = min(res) < res[0] - 1e-6
    full_ok = min(full) <= full[0] * 1.15  # 15% tolerance (a short warm-started smoke)
    live = max(gnorm) > 0.0 and all(np.isfinite(res)) and all(np.isfinite(full))
    print(f"\n  residual↓ (min<ep0): {res_moved} ({res[0]:.1f}→{min(res):.1f}) | "
          f"full did not degrade: {full_ok} ({full[0]:.1f}→{min(full):.1f}) | "
          f"mechanism alive (|g|>0, finite): {live}")
    print("  [smoke: illustrative numbers; the real gate is a full run + run_ablation]")
    assert live, "mechanism dead: |g|≈0 or NaN on both axes"
    assert full_ok, "the full axis degraded >15% — anti-forgetting is not holding (check the mix)"


def main() -> None:
    ap = argparse.ArgumentParser(description="Path B residual-curriculum (Phase 6b, 0011)")
    ap.add_argument("--smoke", action="store_true", help="small run (mechanism demo)")
    ap.add_argument("--epochs", type=int, help="override the number of epochs")
    ap.add_argument("--warm-start", type=str, default=None, help="override the warm start")
    args = ap.parse_args()

    cfg = _smoke_cfg() if args.smoke else POMOConfig.for_residual()
    if args.epochs:
        cfg.epochs = args.epochs
    if args.warm_start:
        cfg.warm_start = args.warm_start

    torch.manual_seed(cfg.seed)
    policy = VRPPolicy()
    if cfg.warm_start and Path(cfg.warm_start).exists():
        policy.load_state_dict(torch.load(cfg.warm_start, weights_only=True))
        print(f"warm-start: {cfg.warm_start}")
    elif cfg.warm_start:
        print(f"[warm start {cfg.warm_start} not found — training from scratch (fine for smoke)]")

    trainer = ResidualPOMOTrainer(policy, cfg)
    base = trainer._validate()  # the warm-start bar BEFORE training (both axes)
    print(
        f"{'SMOKE' if args.smoke else 'FULL'} residual-curriculum | epochs≤{cfg.epochs} "
        f"patience={cfg.patience} batch={cfg.batch} starts={cfg.max_starts} "
        f"steps/ep={cfg.steps_per_epoch} res_frac={cfg.residual_frac} "
        f"lr={cfg.lr:.0e} β={cfg.entropy_beta}\n"
        f"WARM-START BAR: val-RES {base['val_res_cost']:.1f} "
        f"(g{base['val_res_gap_greedy']:+.1%}) | val-FULL {base['val_full_cost']:.1f} "
        f"(g{base['gap_greedy']:+.1%}) | res-heur {trainer.val_res_heur:.1f}"
    )

    hist = trainer.fit(log_fn=_make_log(cfg))
    sel = min(hist, key=lambda h: h["val_res_cost"])  # selection epoch (best-by-val-residual)
    if len(hist) < cfg.epochs:
        print(f"[early-stop at epoch {hist[-1]['epoch']} — {cfg.patience} epochs without gain]")
    print(
        f"\nSelection best-by-val-RESIDUAL: epoch {sel['epoch']} | "
        f"val-RES {sel['val_res_cost']:.1f} (g{sel['val_res_gap_greedy']:+.1%}) | "
        f"val-FULL {sel['val_full_cost']:.1f} (g{sel['gap_greedy']:+.1%})"
    )
    if args.smoke:
        _report_smoke(hist)
    else:
        print(f"\nWeights → {cfg.ckpt} (best-by-val-residual).")
        print("GATE 0011 (after the run): "
              f"python scripts/run_ablation.py --ckpt {cfg.ckpt} → "
              "start_rl_vs_greedy: median_delta_eur<0 AND a_wins>12 → decision 0012.")


if __name__ == "__main__":
    main()
