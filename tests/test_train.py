"""Phase 6 training smoke tests (NOT a full run). Guards against a hidden collapse
(Phase 6 diag: the freeze hid until epoch 100): grad_norm>0 every epoch, val changes
(not all-depot), the t-test is alive (p not nan), entropy neither crashed nor uniform, cost falls.
Requires a snapshot + torch/torch-geometric/scipy.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
pytest.importorskip("scipy")

from logistics_rl_gnn.config import instance as im  # noqa: E402
from logistics_rl_gnn.config.train import TrainConfig  # noqa: E402
from logistics_rl_gnn.models.policy import VRPPolicy  # noqa: E402
from logistics_rl_gnn.train.reinforce import Trainer  # noqa: E402

_NEED_SNAP = pytest.mark.skipif(im._latest_snapshot_dir() is None, reason="no snapshot")


def _smoke_cfg(**kw):
    base = dict(
        epochs=5,  # ≥5: the collapse showed from epoch 1 → a short smoke catches it
        steps_per_epoch=15,
        batch=6,
        n_range=(12, 16),
        val_range=(1_000_000, 1_000_004),
        ckpt=None,
        seed=0,
    )
    return TrainConfig(**{**base, **kw})


@_NEED_SNAP
def test_smoke_learns_no_collapse():
    torch.manual_seed(0)
    trainer = Trainer(VRPPolicy(), _smoke_cfg())
    before = float(
        np.mean([-trainer.policy.rollout(e, "greedy")[2]["reward"] for e in trainer.val_envs])
    )
    hist = trainer.fit()
    vals = [h["val_cost"] for h in hist]
    ents = [h["entropy"] for h in hist]
    gnorms = [h["grad_norm"] for h in hist]
    ps = [h["p"] for h in hist]

    assert vals[-1] < 0.6 * before, f"training did not reduce cost: {before:.0f} → {vals[-1]:.0f}"
    # FREEZE GUARD: |g|→0 (saturation) — the collapse that hid until epoch 100
    assert all(g > 1e-6 for g in gnorms), f"grad_norm hit zero (saturation collapse): {gnorms}"
    # ALL-DEPOT GUARD: val stuck at one value (n·200)
    assert len({round(v, 1) for v in vals}) > 1, f"val is frozen (all-depot?): {vals}"
    # BASELINE GUARD: the t-test is not nan (no greedy(current)==greedy(baseline) deadlock)
    assert all(not np.isnan(p) for p in ps), f"baseline deadlock (p=nan): {ps}"
    # ENTROPY GUARD: without tanh-clip softmax could over-sharpen → not crashed to 0; not uniform
    assert all(0.05 < e < math.log(20) for e in ents), f"entropy outside [0.05, log20]: {ents}"
    for h in hist:  # no NaN in the logs
        assert np.isfinite([h["train_cost"], h["val_cost"], h["grad_norm"], h["entropy"]]).all()


@_NEED_SNAP
def test_runtime_freeze_guard_aborts():
    # runtime guard: if grad_norm≈0 (saturation) for N epochs in a row → abort, not 100 wasted
    class FrozenTrainer(Trainer):
        def train_batch(self, seeds):  # imitate a frozen gradient
            return {"cost": 9999.0, "entropy": 2.9, "grad_norm": 0.0}

    torch.manual_seed(0)
    tr = FrozenTrainer(VRPPolicy(), _smoke_cfg(epochs=10, steps_per_epoch=1))
    with pytest.raises(RuntimeError, match="frozen"):
        tr.fit()


@_NEED_SNAP
def test_determinism_by_seed():
    def run():
        torch.manual_seed(0)
        return Trainer(VRPPolicy(), _smoke_cfg(epochs=3, steps_per_epoch=6, batch=4)).fit()

    h1, h2 = run(), run()
    assert [round(r["train_cost"], 4) for r in h1] == [round(r["train_cost"], 4) for r in h2]
    assert [round(r["val_cost"], 4) for r in h1] == [round(r["val_cost"], 4) for r in h2]
