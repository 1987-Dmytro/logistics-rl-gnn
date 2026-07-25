"""Phase 9 — falsifiability: are these the right weights and does the model actually add anything.

tamper: no weights file → SystemExit with a clear message (NOT a silent fallback to greedy —
otherwise the demo keeps printing 'system numbers' computed without the model).
sha parity: the checkpoint sha256 == the provenance of the durable summary the demo reads.
weight-swap: a random init of THE SAME architecture → the model's candidates are markedly dearer
than the trained ones (if the pipeline ignored the weights, the metric would not move).

Only weight-swap is heavy (polish off → seconds); everything else runs without a solver.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "scripts"))

demo = pytest.importorskip("demo")  # skips on a runner without torch/opening_hours

_CKPT = _ROOT / "results" / "policy_pomo_congestion.pt"
_SM = _ROOT / "results" / "system_metrics.json"
_SNAP = _ROOT / "data" / "snapshots" / "augsburg_20260720"
_NEED = _CKPT.exists() and _SM.exists() and (_SNAP / "meta.json").exists()
_SWAP_MIN_GAP = 0.15  # a random init must be ≥15% worse than the trained one (seed 0: +265%)


def _fake_sm(ckpt: Path) -> dict:
    return {
        "config": {"ckpt": str(ckpt), "budget_ms": 1.0, "k_samples": 1, "temperature": 1.0,
                   "rl_starts": 1},
        "provenance": {"checkpoint": {"path": str(ckpt), "sha256_16": "0" * 16}},
        "per_seed_cost_eur": [0.0],
    }


def test_tamper_missing_weights_exits_loudly(tmp_path, monkeypatch):
    """No weights → SystemExit (not a silent fallback); the message names `--no-model`."""
    gone = tmp_path / "gone.pt"
    sm = tmp_path / "system_metrics.json"
    sm.write_text(json.dumps(_fake_sm(gone)), encoding="utf-8")
    monkeypatch.setattr(demo, "_SM", sm)
    with pytest.raises(SystemExit) as e:  # fails BEFORE the heavy pipeline — no snapshot needed
        demo.run_demo(seed=0, event_kind="traffic", out_dir=str(tmp_path), open_maps=False)
    msg = str(e.value)
    assert "NO WEIGHTS" in msg and "--no-model" in msg
    assert not list(tmp_path.glob("*.html")), "no silent fallback: no artefacts are written"


def test_provenance_mismatch_exits(tmp_path):
    """Different weights at the same path → a loud stop (the demo numbers would be incomparable)."""
    ckpt = tmp_path / "weights.pt"
    ckpt.write_bytes(b"NOT the trained weights")
    sm = {"provenance": {"checkpoint": {"sha256_16": "24c8cfb0607235f8"}}}
    with pytest.raises(SystemExit, match="PROVENANCE MISMATCH"):
        demo.check_model_provenance(ckpt, sm)


def test_provenance_needs_recorded_sha(tmp_path):
    """A summary without provenance also stops: nothing to verify the weights against."""
    ckpt = tmp_path / "weights.pt"
    ckpt.write_bytes(b"x")
    with pytest.raises(SystemExit, match="nothing to verify"):
        demo.check_model_provenance(ckpt, {"provenance": {}})


@pytest.mark.skipif(not _NEED, reason="ckpt/system_metrics outside git (#1)")
def test_sha_parity_with_summary():
    """The real checkpoint sha256 == the durable summary provenance; date and decision are found."""
    sm = json.loads(_SM.read_text())
    prov = demo.check_model_provenance(_ROOT / sm["config"]["ckpt"], sm)  # config path from root
    assert prov["sha256"].startswith(sm["provenance"]["checkpoint"]["sha256_16"])
    assert prov["date"], "the training date was not found in the training summary"
    assert prov["decision"] and (_ROOT / prov["decision"]).exists(), "decision file not referenced"


@pytest.mark.skipif(not _NEED, reason="ckpt/snapshot outside git (#1)")
def test_weight_swap_degrades_model_candidates():
    """A random init of the same architecture → the mean cost of RL candidates is clearly worse.

    polish is off (budget_ms=0) and K is small → seconds. We compare ONLY the model's candidates
    (greedy does not depend on the weights and would dilute the effect)."""
    import eval_system as es
    import run_dynamic as rd
    import torch

    from logistics_rl_gnn.config import instance as im
    from logistics_rl_gnn.models.policy import VRPPolicy

    inst = im.generate_instance(snapshot_dir=_SNAP, seed=0)
    torch.manual_seed(0)
    trained = rd._load_policy(_CKPT)
    torch.manual_seed(0)
    random_init = VRPPolicy()
    random_init.eval()

    kw = {"budget_ms": 0.0, "k_samples": 8, "temp": 1.0, "rl_starts": 4}
    good, bad = {}, {}
    es.system_routes(trained, inst, report=good, **kw)
    es.system_routes(random_init, inst, report=bad, **kw)

    assert good["rl_mean"] is not None and bad["rl_mean"] is not None
    gap = bad["rl_mean"] / good["rl_mean"] - 1.0
    assert gap > _SWAP_MIN_GAP, (
        f"swapping the weights barely changed the candidates (+{gap:.1%} ≤ {_SWAP_MIN_GAP:.0%}) — "
        f"does the pipeline ignore the model? trained {good['rl_mean']:.1f}€, random "
        f"{bad['rl_mean']:.1f}€"
    )


@pytest.mark.skipif(not _NEED, reason="ckpt/snapshot outside git (#1)")
def test_no_model_portfolio_has_no_rl_candidates():
    """`--no-model` = a portfolio without RL candidates: only greedy in the table, no rl_mean."""
    import eval_system as es

    from logistics_rl_gnn.config import instance as im

    rep: dict = {}
    es.system_routes(None, im.generate_instance(snapshot_dir=_SNAP, seed=0), budget_ms=0.0,
                     k_samples=8, temp=1.0, rl_starts=4, report=rep)
    assert [r["source"] for r in rep["rows"]] == ["greedy"]
    assert rep["rl_mean"] is None and rep["used_model"] is False
    assert rep["chosen"] == "greedy+polish"  # nobody else can win — the model did not take part


@pytest.mark.skipif(not _NEED, reason="ckpt/snapshot outside git (#1)")
def test_model_contribution_is_a_real_ablation():
    """Model contribution = min(all candidates+polish) vs polished greedy on ONE instance/budget.

    Invariant by construction: with the model it is NEVER WORSE than without (RL candidates only
    add to the pool). And the 'without the model' side of a run WITH the model == a run where the
    model was never loaded (same polish budget per candidate) — else the line compares apples."""
    import eval_system as es
    import run_dynamic as rd
    import torch

    from logistics_rl_gnn.config import instance as im

    # a small instance + a generous budget → polish CONVERGES (rather than being cut by wall-clock),
    # otherwise the equality of the two sides would depend on machine load and the test would flake
    inst = im.generate_instance(snapshot_dir=_SNAP, seed=0, include_stops=set(range(1, 13)))
    kw = {"budget_ms": 20000.0, "k_samples": 8, "temp": 1.0, "rl_starts": 4}
    torch.manual_seed(0)
    with_model, without_model = {}, {}
    es.system_routes(rd._load_policy(_CKPT), inst, report=with_model, **kw)
    es.system_routes(None, inst, report=without_model, **kw)

    assert with_model["cost_model"] <= with_model["cost_nomodel"] + 1e-9  # the model never hurts
    assert with_model["cost_nomodel"] == pytest.approx(without_model["cost_model"], abs=1e-6)
