"""Phase 9 — фальсифицируемость: те ли это веса и правда ли модель что-то даёт.

tamper: нет файла весов → SystemExit с внятным сообщением (НЕ тихий фолбэк на greedy — иначе
демо продолжит печатать «числа системы», считая их без модели).
sha-парити: sha256 чекпойнта == провенанс durable-сводки, из которой демо берёт числа.
weight-swap: random-init ТОЙ ЖЕ архитектуры → кандидаты модели заметно дороже обученных (если бы
пайплайн игнорировал веса, метрика не шелохнулась бы).

Тяжёлый только weight-swap (polish выключен → секунды); остальное — без солвера.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "scripts"))

demo = pytest.importorskip("demo")  # skips на раннере без torch/opening_hours

_CKPT = _ROOT / "results" / "policy_pomo_congestion.pt"
_SM = _ROOT / "results" / "system_metrics.json"
_SNAP = _ROOT / "data" / "snapshots" / "augsburg_20260720"
_NEED = _CKPT.exists() and _SM.exists() and (_SNAP / "meta.json").exists()
_SWAP_MIN_GAP = 0.15  # random-init обязан быть хуже обученного ≥15% (замер seed 0: +265%)


def _fake_sm(ckpt: Path) -> dict:
    return {
        "config": {"ckpt": str(ckpt), "budget_ms": 1.0, "k_samples": 1, "temperature": 1.0,
                   "rl_starts": 1},
        "provenance": {"checkpoint": {"path": str(ckpt), "sha256_16": "0" * 16}},
        "per_seed_cost_eur": [0.0],
    }


def test_tamper_missing_weights_exits_loudly(tmp_path, monkeypatch):
    """Нет весов → SystemExit (не тихий фолбэк), сообщение называет честный выход `--no-model`."""
    gone = tmp_path / "gone.pt"
    sm = tmp_path / "system_metrics.json"
    sm.write_text(json.dumps(_fake_sm(gone)), encoding="utf-8")
    monkeypatch.setattr(demo, "_SM", sm)
    with pytest.raises(SystemExit) as e:  # падаем ДО тяжёлого пайплайна — снапшот не нужен
        demo.run_demo(seed=0, event_kind="traffic", out_dir=str(tmp_path), open_maps=False)
    msg = str(e.value)
    assert "НЕТ ВЕСОВ" in msg and "--no-model" in msg
    assert not list(tmp_path.glob("*.html")), "тихого фолбэка нет: артефакты не пишутся"


def test_provenance_mismatch_exits(tmp_path):
    """Другие веса под тем же путём → громкий останов (числа демо стали бы несопоставимы)."""
    ckpt = tmp_path / "weights.pt"
    ckpt.write_bytes(b"NOT the trained weights")
    sm = {"provenance": {"checkpoint": {"sha256_16": "24c8cfb0607235f8"}}}
    with pytest.raises(SystemExit, match="ПРОВЕНАНС-МИСМАТЧ"):
        demo.check_model_provenance(ckpt, sm)


def test_provenance_needs_recorded_sha(tmp_path):
    """Сводка без провенанса — тоже останов: проверить веса нечем, молча верить нельзя."""
    ckpt = tmp_path / "weights.pt"
    ckpt.write_bytes(b"x")
    with pytest.raises(SystemExit, match="нечем проверить"):
        demo.check_model_provenance(ckpt, {"provenance": {}})


@pytest.mark.skipif(not _NEED, reason="ckpt/system_metrics вне git (запрет №1)")
def test_sha_parity_with_summary():
    """sha256 реального чекпойнта == провенанс durable-сводки; дата обучения и решение найдены."""
    sm = json.loads(_SM.read_text())
    prov = demo.check_model_provenance(_ROOT / sm["config"]["ckpt"], sm)  # путь в конфиге от корня
    assert prov["sha256"].startswith(sm["provenance"]["checkpoint"]["sha256_16"])
    assert prov["date"], "дата обучения не найдена в training-summary"
    assert prov["decision"] and (_ROOT / prov["decision"]).exists(), "решение не сослалось на файл"


@pytest.mark.skipif(not _NEED, reason="ckpt/snapshot вне git (запрет №1)")
def test_weight_swap_degrades_model_candidates():
    """Random-init той же архитектуры → средний cost RL-кандидатов заметно хуже обученных.

    polish выключен (budget_ms=0) и K маленькое → секунды. Сравниваем ТОЛЬКО кандидатов модели
    (greedy от весов не зависит и разбавил бы эффект)."""
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
        f"подмена весов почти не изменила кандидатов (+{gap:.1%} ≤ {_SWAP_MIN_GAP:.0%}) — "
        f"пайплайн не зависит от модели? обучен {good['rl_mean']:.1f}€, random "
        f"{bad['rl_mean']:.1f}€"
    )


@pytest.mark.skipif(not _NEED, reason="ckpt/snapshot вне git (запрет №1)")
def test_no_model_portfolio_has_no_rl_candidates():
    """`--no-model` = портфель без RL-кандидатов: в таблице только greedy, rl_mean отсутствует."""
    import eval_system as es

    from logistics_rl_gnn.config import instance as im

    rep: dict = {}
    es.system_routes(None, im.generate_instance(snapshot_dir=_SNAP, seed=0), budget_ms=0.0,
                     k_samples=8, temp=1.0, rl_starts=4, report=rep)
    assert [r["source"] for r in rep["rows"]] == ["greedy"]
    assert rep["rl_mean"] is None and rep["used_model"] is False
    assert rep["chosen"] == "greedy+polish"  # победить больше некому — модель не участвовала


@pytest.mark.skipif(not _NEED, reason="ckpt/snapshot вне git (запрет №1)")
def test_model_contribution_is_a_real_ablation():
    """Вклад модели = min(все кандидаты+polish) vs polished greedy на ОДНОМ инстансе и бюджете.

    Инвариант по построению: с моделью НЕ ХУЖЕ, чем без (RL-кандидаты лишь добавляются в пул).
    И «без модели» из прогона С моделью == прогон, где модель вообще не грузилась (тот же polish-
    бюджет на кандидата) — иначе строка «вклад» сравнивала бы разные вещи."""
    import eval_system as es
    import run_dynamic as rd
    import torch

    from logistics_rl_gnn.config import instance as im

    # маленький инстанс + щедрый бюджет → polish СХОДИТСЯ (а не обрывается по wall-clock),
    # иначе равенство сторон зависело бы от загрузки машины и тест флейкал бы
    inst = im.generate_instance(snapshot_dir=_SNAP, seed=0, include_stops=set(range(1, 13)))
    kw = {"budget_ms": 20000.0, "k_samples": 8, "temp": 1.0, "rl_starts": 4}
    torch.manual_seed(0)
    with_model, without_model = {}, {}
    es.system_routes(rd._load_policy(_CKPT), inst, report=with_model, **kw)
    es.system_routes(None, inst, report=without_model, **kw)

    assert with_model["cost_model"] <= with_model["cost_nomodel"] + 1e-9  # модель не ухудшает
    assert with_model["cost_nomodel"] == pytest.approx(without_model["cost_model"], abs=1e-6)
