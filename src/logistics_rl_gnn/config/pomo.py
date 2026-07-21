"""Конфиг POMO-обучения (Phase 6b Шаг 1). Multi-start + shared baseline на статике.

lr=1e-3 + grad-clip=1.0 — находка Phase 5/6 (clip нормирует ‖grad‖ → шаг ~lr независимо от
масштаба центрированного advantage; 1e-2 давал uniform-коллапс). max_starts = N разных первых
узлов (POMO N стартов), режется до числа допустимых-на-шаге-0 аптек.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class POMOConfig:
    epochs: int = 100
    steps_per_epoch: int = 30  # батчей на эпоху
    batch: int = 16  # инстансов в батче (на каждый — max_starts траекторий)
    max_starts: int = 16  # POMO: N разных первых узлов (≤ числа аптек)
    lr: float = 1e-3
    grad_clip: float = 1.0
    n_range: tuple[int, int] = (40, 62)  # аптек в эпизоде
    train_range: tuple[int, int] = (0, 100_000)  # train-сиды A
    val_range: tuple[int, int] = (1_000_000, 1_000_016)  # val-сиды B (непересекающийся с A)
    entropy_beta: float = 0.0  # бонус +β·H в резерве (против переострения softmax)
    seed: int = 0
    ckpt: str | None = "results/policy_pomo_best.pt"  # лучшая по val (вне git: results/ + *.pt)
    eval_seeds: tuple[int, ...] = tuple(range(10))  # full-62 сиды для «Стало» (== «Было» Phase 4)

    def val_seeds(self) -> range:
        return range(*self.val_range)

    @classmethod
    def smoke(cls) -> POMOConfig:
        # мелко и быстро: малые инстансы, мало стартов, ≥4 эпохи (видно cost↓ + разброс стартов)
        return cls(
            epochs=5,
            steps_per_epoch=8,
            batch=4,
            max_starts=5,
            n_range=(15, 20),
            val_range=(1_000_000, 1_000_004),
            ckpt=None,
            seed=0,
        )
