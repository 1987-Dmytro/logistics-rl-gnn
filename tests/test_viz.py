"""Phase 8 — стражи viz/метрик: парити final_metrics с decision 0002/0008/0009, идемпотентность,
детерминизм, не-мутация durable json. Скрипты-картинки seeded (torch.manual_seed(0), фикс. seed) →
детерминизм проверяем на их ДЕТЕРМИНИР. ядре (parse/build), не на байтах PNG (там таймстамп).

Durable results/*.json — ВНЕ git (запрет №1) → парити-тест skipif при отсутствии.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "scripts"))

import final_metrics as fm  # noqa: E402

# viz_training тянет matplotlib (группа [viz]) на уровне модуля → импортим лениво в тесте под
# importorskip; иначе на голом раннере (без [viz]) collection упадёт ImportError.

_JSON = ("baselines.json", "system_metrics.json", "polish_summary.json")
_NEED = [_ROOT / "results" / f for f in _JSON]
_HAVE = all(p.exists() for p in _NEED)
_skip = pytest.mark.skipif(not _HAVE, reason="durable results/*.json вне git (запрет №1)")

# decision-якоря (парити) — числа из 0002 (greedy/OR), 0009 (система), 0008 (portfolio до polish)
_ANCHOR_0002_GREEDY = 825.38
_ANCHOR_0002_ORTOOLS = 611.14
_ANCHOR_0009_SYSTEM = 631.62
_ANCHOR_0008_PORTFOLIO = 766.14


@_skip
def test_final_metrics_parity_with_decisions():
    """final_metrics сходится с числами decision 0002/0009 (иначе артефакты разъехались)."""
    m = fm.build()
    r = m["rows"]
    assert abs(r["greedy"]["cost_eur"] - _ANCHOR_0002_GREEDY) < 0.5
    assert abs(r["ortools"]["cost_eur"] - _ANCHOR_0002_ORTOOLS) < 0.5
    assert abs(r["system"]["cost_eur"] - _ANCHOR_0009_SYSTEM) < 0.5  # парити к 0009 631.6
    # 0008-цепочка (portfolio до polish) в провенансе
    assert abs(m["provenance"]["step3_portfolio_eur"] - _ANCHOR_0008_PORTFOLIO) < 0.5


@_skip
def test_final_metrics_deltas_direction():
    """Знаки/порядок дельт: система −23.5% к greedy, +3.4% к OR, реакция ×>1 vs OR-Tools."""
    d = fm.build()["deltas"]
    assert -0.30 < d["cost_vs_greedy"] < -0.18  # ≈ −23.5%
    assert 0.0 < d["cost_vs_ortools"] < 0.08  # ≈ +3.4%
    assert d["distance_vs_greedy"] < 0.02  # пробег ~flat/чуть ниже (честно: не главный рычаг)
    assert d["time_vs_greedy"] < 0.0  # время ниже
    assert d["reaction_speedup_vs_ortools"] > 1.0  # быстрее OR-Tools


@_skip
def test_final_metrics_idempotent_and_pure():
    """build() идемпотентен + НЕ мутирует durable json (только читает)."""
    before = hashlib.sha256((_ROOT / "results" / "baselines.json").read_bytes()).hexdigest()
    a = fm.build()
    b = fm.build()
    after = hashlib.sha256((_ROOT / "results" / "baselines.json").read_bytes()).hexdigest()
    assert a == b, "build() не идемпотентен"
    assert before == after, "build() мутировал baselines.json"


def test_eval_system_anchor_matches_0009():
    """Якорь парити в eval_system == durable 0009 631.6€ (self-consistency, без прогона)."""
    import eval_system as es

    assert abs(es._DURABLE_COST_0009 - _ANCHOR_0009_SYSTEM) < 0.01


def test_viz_training_parse_deterministic():
    """parse_log детерминирован; на известном логе даёт непустые (epoch,cost) — если лог есть."""
    pytest.importorskip("matplotlib")  # viz_training тянет matplotlib (группа [viz])
    import viz_training as vt

    log = _ROOT / "results" / "pomo_20260721_1914.log"
    if not log.exists():
        pytest.skip("тренировочный лог вне git")
    a = vt.parse_log(str(log))
    b = vt.parse_log(str(log))
    assert a == b and len(a[0]) > 0  # (epochs, costs, hs) воспроизводимо и непусто


def test_final_metrics_md_table_shape():
    """_fmt_md даёт MD-таблицу с 3 системами + ключевыми метриками (если durable есть)."""
    if not _HAVE:
        pytest.skip("durable json вне git")
    md = fm._fmt_md(fm.build())
    assert "| Метрика |" in md and "Издержки, €" in md
    assert "Латентность re-plan" in md and "Пробег" in md
