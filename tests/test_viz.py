"""Phase 8 — viz/metric guards: final_metrics parity with decision 0002/0008/0009, idempotence,
determinism, no mutation of the durable json. The image scripts are seeded (torch.manual_seed(0))
→ determinism is checked on their DETERMINISTIC core (parse/build), not on PNG bytes (timestamps).

Durable results/*.json are OUTSIDE git (#1) → the parity test is skipped when they are absent.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "scripts"))

import final_metrics as fm  # noqa: E402

# viz_training pulls matplotlib (group [viz]) at module level → imported lazily in the test under
# importorskip; otherwise a bare runner (without [viz]) fails collection with ImportError.

_JSON = ("baselines.json", "system_metrics.json", "polish_summary.json")
_NEED = [_ROOT / "results" / f for f in _JSON]
_HAVE = all(p.exists() for p in _NEED)
_skip = pytest.mark.skipif(not _HAVE, reason="durable results/*.json outside git (#1)")

# decision anchors (parity) — numbers from 0002 (greedy/OR), 0009 (system), 0008 (pre-polish)
_ANCHOR_0002_GREEDY = 825.38
_ANCHOR_0002_ORTOOLS = 611.14
_ANCHOR_0009_SYSTEM = 631.62
_ANCHOR_0008_PORTFOLIO = 766.14


@_skip
def test_final_metrics_parity_with_decisions():
    """final_metrics agrees with the numbers of decision 0002/0009 (else the artefacts drifted)."""
    m = fm.build()
    r = m["rows"]
    assert abs(r["greedy"]["cost_eur"] - _ANCHOR_0002_GREEDY) < 0.5
    assert abs(r["ortools"]["cost_eur"] - _ANCHOR_0002_ORTOOLS) < 0.5
    assert abs(r["system"]["cost_eur"] - _ANCHOR_0009_SYSTEM) < 0.5  # parity to 0009 631.6
    # the 0008 chain (portfolio before polish) in the provenance
    assert abs(m["provenance"]["step3_portfolio_eur"] - _ANCHOR_0008_PORTFOLIO) < 0.5


@_skip
def test_final_metrics_deltas_direction():
    """Signs/order of the deltas: system −23.5% vs greedy, +3.4% vs OR, reaction ×>1 vs OR-Tools."""
    d = fm.build()["deltas"]
    assert -0.30 < d["cost_vs_greedy"] < -0.18  # ≈ −23.5%
    assert 0.0 < d["cost_vs_ortools"] < 0.08  # ≈ +3.4%
    assert d["distance_vs_greedy"] < 0.02  # distance ~flat/slightly lower (not the main lever)
    assert d["time_vs_greedy"] < 0.0  # time is lower
    assert d["reaction_speedup_vs_ortools"] > 1.0  # faster than OR-Tools


@_skip
def test_final_metrics_idempotent_and_pure():
    """build() is idempotent + does NOT mutate the durable json (read-only)."""
    before = hashlib.sha256((_ROOT / "results" / "baselines.json").read_bytes()).hexdigest()
    a = fm.build()
    b = fm.build()
    after = hashlib.sha256((_ROOT / "results" / "baselines.json").read_bytes()).hexdigest()
    assert a == b, "build() is not idempotent"
    assert before == after, "build() mutated baselines.json"


def test_eval_system_anchor_matches_0009():
    """The parity anchor in eval_system == durable 0009 631.6€ (self-consistency, no run)."""
    pytest.importorskip("torch")  # eval_system imports the policy at module level (group [model])
    import eval_system as es

    assert abs(es._DURABLE_COST_0009 - _ANCHOR_0009_SYSTEM) < 0.01


def test_viz_training_parse_deterministic():
    """parse_log is deterministic; on a known log it yields non-empty (epoch,cost) if present."""
    pytest.importorskip("matplotlib")  # viz_training pulls matplotlib (group [viz])
    import viz_training as vt

    log = _ROOT / "results" / "pomo_20260721_1914.log"
    if not log.exists():
        pytest.skip("the training log is outside git")
    a = vt.parse_log(str(log))
    b = vt.parse_log(str(log))
    assert a == b and len(a[0]) > 0  # (epochs, costs, hs) reproducible and non-empty


def test_final_metrics_md_table_shape():
    """_fmt_md yields an MD table with 3 systems + the key metrics (when durable json exist)."""
    if not _HAVE:
        pytest.skip("durable json outside git")
    md = fm._fmt_md(fm.build())
    assert "| Metric |" in md and "Costs, €" in md
    assert "Re-plan latency" in md and "Distance" in md
