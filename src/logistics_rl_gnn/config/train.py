"""Конфиг обучения RL (Phase 6). lr=1e-3 + grad-clip — находка Phase 5 (1e-2 → uniform-коллапс)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainConfig:
    epochs: int = 100
    steps_per_epoch: int = 50  # батчей на эпоху
    batch: int = 32  # инстансов в батче
    lr: float = 1e-3
    grad_clip: float = 1.0
    n_range: tuple[int, int] = (40, 62)  # аптек в эпизоде
    train_range: tuple[int, int] = (0, 100_000)  # диапазон train-сидов A
    val_range: tuple[int, int] = (1_000_000, 1_000_048)  # val-сиды B (непересекающийся с A)
    baseline_p: float = 0.05  # порог paired t-test для обновления rollout-baseline
    seed: int = 0
    ckpt: str | None = "results/policy_best.pt"  # лучшая по val (вне git: results/ + *.pt)
    eval_seeds: tuple[int, ...] = tuple(range(10))  # full-62 сиды для «Стало» (== «Было» Phase 4)

    def val_seeds(self) -> range:
        return range(*self.val_range)

    @classmethod
    def smoke(cls) -> TrainConfig:
        # мелко и быстро (<30с): малые инстансы, 2 эпохи × 20 шагов = 40 апдейтов
        return cls(
            epochs=2,
            steps_per_epoch=20,
            batch=8,
            n_range=(15, 20),
            val_range=(1_000_000, 1_000_006),
            ckpt=None,
            seed=0,
        )
