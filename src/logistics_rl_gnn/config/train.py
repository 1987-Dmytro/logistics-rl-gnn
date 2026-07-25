"""RL training config (Phase 6). lr=1e-3 + grad-clip — Phase 5 finding (1e-2 → uniform collapse)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainConfig:
    epochs: int = 100
    steps_per_epoch: int = 50  # batches per epoch
    batch: int = 32  # instances per batch
    lr: float = 1e-3
    grad_clip: float = 1.0
    n_range: tuple[int, int] = (40, 62)  # pharmacies per episode
    train_range: tuple[int, int] = (0, 100_000)  # range of train seeds A
    val_range: tuple[int, int] = (1_000_000, 1_000_048)  # val seeds B (disjoint from A)
    baseline_p: float = 0.05  # paired t-test threshold for updating the rollout baseline
    entropy_beta: float = 0.0  # +β·H bonus in reserve (on if softmax over-sharpens w/o tanh-clip)
    seed: int = 0
    ckpt: str | None = "results/policy_best.pt"  # best by val (outside git: results/ + *.pt)
    eval_seeds: tuple[int, ...] = tuple(range(10))  # full-62 seeds for "after" (= Phase 4 "before")

    def val_seeds(self) -> range:
        return range(*self.val_range)

    @classmethod
    def smoke(cls) -> TrainConfig:
        # small and fast: tiny instances, ≥5 epochs (catches the collapse that hid until 100)
        return cls(
            epochs=5,
            steps_per_epoch=20,
            batch=8,
            n_range=(15, 20),
            val_range=(1_000_000, 1_000_006),
            ckpt=None,
            seed=0,
        )
