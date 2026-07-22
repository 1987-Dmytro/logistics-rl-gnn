"""Конфиг POMO-обучения (Phase 6b Шаг 1·refit). Multi-start + shared baseline на статике,
анти-оверфит-протокол: seed-сплит train/val/test, entropy-бонус, early-stop best-by-val.

lr=1e-3 + grad-clip=1.0 — находка Phase 5/6 (clip нормирует ‖grad‖ → шаг ~lr независимо от
масштаба центрированного advantage; 1e-2 давал uniform-коллапс). max_starts = N разных первых
узлов (POMO N стартов), режется до числа допустимых-на-шаге-0 аптек. epochs = верхний предел
(epochs_max); реально прогон режет early-stop по val (patience). test — held-out, НИКОГДА не для
отбора модели. val_n_range — лычаг: если val насыщается у эвристики (сигнал отбора слаб) — поднять
val/test до deployment-размера n≈62 (см. 0006), train остаётся на n_range.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class POMOConfig:
    epochs: int = 300  # epochs_max — верхний предел; early-stop (patience) обрежет раньше
    steps_per_epoch: int = 30  # батчей на эпоху
    batch: int = 16  # инстансов в батче (на каждый — max_starts траекторий)
    max_starts: int = 16  # POMO N_starts: N разных первых узлов (≤ числа аптек)
    lr: float = 1e-3
    grad_clip: float = 1.0
    entropy_beta: float = 0.01  # бонус +β·H (loss−=β·H) против переострения softmax (анти-оверфит)
    patience: int = 15  # эпох без улучшения val → early-stop (best-by-val сохранён)
    n_range: tuple[int, int] = (40, 62)  # аптек в train-эпизоде
    train_range: tuple[int, int] = (0, 1_000_000)  # train-сиды A (subset-инстансы)
    val_range: tuple[int, int] = (1_000_000, 1_000_064)  # val ~64 (отбор/early-stop), A∩=∅
    test_range: tuple[int, int] = (2_000_000, 2_000_064)  # test ~64 (финал, НЕ для отбора), ∩=∅
    val_n_range: tuple[int, int] | None = None  # лычаг: val/test на другом n (None → как n_range)
    seed: int = 0
    ckpt: str | None = "results/policy_pomo_refit.pt"  # refit → отдельный файл (не трогаем 770.4€)
    eval_seeds: tuple[int, ...] = tuple(range(10))  # full-62 сиды для «Стало» (== «Было» Phase 4)

    def val_seeds(self) -> range:
        return range(*self.val_range)

    def test_seeds(self) -> range:
        return range(*self.test_range)

    @classmethod
    def smoke(cls) -> POMOConfig:
        # мелко и быстро: малые инстансы, мало стартов, patience<epochs (видно early-stop + cost↓)
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
