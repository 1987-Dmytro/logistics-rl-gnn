"""POMO training config (Phase 6b Step 1·refit). Multi-start + shared baseline on statics,
anti-overfit protocol: seed split train/val/test, entropy bonus, early-stop best-by-val.

lr=1e-3 + grad-clip=1.0 — a Phase 5/6 finding (clip normalises ‖grad‖ → the step stays ~lr
regardless of the centred advantage scale; 1e-2 produced uniform collapse). max_starts = N
distinct first nodes (POMO N starts), trimmed to the number of pharmacies feasible at step 0.
epochs is an upper bound (epochs_max); a real run is cut short by early-stop on val (patience).
test is held out and NEVER used for model selection. val_n_range is a lever: if val saturates at
the heuristic (weak selection signal), raise val/test to the deployment size n≈62 (see 0006)
while train stays on n_range.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class POMOConfig:
    epochs: int = 300  # epochs_max — upper bound; early-stop (patience) cuts it short
    steps_per_epoch: int = 30  # batches per epoch
    batch: int = 16  # instances per batch (max_starts trajectories for each)
    max_starts: int = 16  # POMO N_starts: N distinct first nodes (≤ pharmacy count)
    lr: float = 1e-3
    grad_clip: float = 1.0
    entropy_beta: float = 0.01  # +β·H bonus (loss−=β·H) against softmax over-sharpening
    patience: int = 15  # epochs without val improvement → early-stop (best-by-val kept)
    n_range: tuple[int, int] = (40, 62)  # pharmacies per train episode
    train_range: tuple[int, int] = (0, 1_000_000)  # train seeds A (subset instances)
    val_range: tuple[int, int] = (1_000_000, 1_000_064)  # val ~64 (selection/early-stop), A∩=∅
    test_range: tuple[int, int] = (2_000_000, 2_000_064)  # test ~64 (final, NOT for selection), ∩=∅
    val_n_range: tuple[int, int] | None = None  # lever: val/test on another n (None → as n_range)
    seed: int = 0
    ckpt: str | None = "results/policy_pomo_refit.pt"  # refit → its own file (770.4€ untouched)
    eval_seeds: tuple[int, ...] = tuple(range(10))  # full-62 seeds for "after" (= Phase 4 "before")
    # --- Step 2: training under congestion (dynamics during training) ---
    congestion: bool = False  # True → train/val/test on congestion instances (else free-flow)
    warm_start: str | None = None  # path to weights for a warm start (Step 2: static best.pt)
    cong_dow: int = 1  # DELIVERY_WEEKDAY (Tuesday) — congestion profile of the delivery day
    cong_offset_max_min: float = 480.0  # tour start ∈ [08:00, 16:00] (diurnal variety)
    cong_incidents: tuple[int, int] = (1, 3)  # incidents per t0 (≥1 → coverage, see advisor)
    cong_inc_dur_min: tuple[float, float] = (180.0, 360.0)  # long-lived → signal lasts an episode
    # --- Path B: residual-curriculum (0011-prereg) ---
    residual_frac: float = 0.5  # share of residual episodes in a batch (rest are full congestion)
    res_frac_range: tuple[float, float] = (0.2, 0.8)  # progress of the greedy prefix (0011)
    res_train_base: int = 3_000_000  # residual-train seed base (generate_instance; disjoint 0–9)
    res_val_range: tuple[int, int] = (4_000_000, 4_000_048)  # residual-val pool (≥48, 0011)

    def val_seeds(self) -> range:
        return range(*self.val_range)

    def test_seeds(self) -> range:
        return range(*self.test_range)

    def res_val_seeds(self) -> range:
        return range(*self.res_val_range)

    @classmethod
    def for_congestion(cls, **kw) -> POMOConfig:
        """Step 2: warm start from static 770.4€, fine-tune under congestion. lr below 1e-3 (refit
        wobbled at 1e-3 on the SAME distribution → a warm start into a NEW one risks the basin)."""
        base = dict(
            congestion=True,
            warm_start="results/policy_pomo_best.pt",  # 770.4€ static (arch-compatible)
            lr=3e-4,
            entropy_beta=0.03,  # a bit more exploration for fine-tuning into a new distribution
            ckpt="results/policy_pomo_congestion.pt",  # separate (best.pt/refit untouched)
        )
        return cls(**{**base, **kw})

    @classmethod
    def for_residual(cls, **kw) -> POMOConfig:
        """Path B (0011): warm start from congestion-best, mixing 50% full + 50% residual episodes.
        congestion=True → the full half is congestion statics (the warm-start distribution,
        anti-forgetting); the residual half is built separately by the generator. lr/β as in
        for_congestion (fine-tuning into an adjacent distribution)."""
        base = dict(
            congestion=True,
            warm_start="results/policy_pomo_congestion.pt",
            lr=3e-4,
            entropy_beta=0.03,
            ckpt="results/policy_pomo_residual.pt",  # separate (congestion/best/refit untouched)
        )
        return cls(**{**base, **kw})

    @classmethod
    def smoke(cls) -> POMOConfig:
        # small and fast: tiny instances, few starts, patience<epochs (early-stop + cost↓ visible)
        return cls(
            epochs=5,
            steps_per_epoch=8,
            batch=4,
            max_starts=5,
            patience=3,
            n_range=(15, 20),
            val_range=(1_000_000, 1_000_004),
            test_range=(2_000_000, 2_000_004),
            ckpt=None,
            seed=0,
        )
