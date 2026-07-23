"""Phase 8 — страж demo.py: end-to-end на seed 0. Статик-cost ДЕТЕРМИНИРОВАН → точный парити с
system_metrics per_seed[0]; динамика (residual/latency) wall-clock → фиксируем ТОЛЬКО структурно
(файлы есть, served+pending == все клиенты по 0004-механике, поля в диапазоне) — не точные числа.

Тяжёлый (полный system_routes + re-plan + OR-Tools ~30с) → skipif при отсутствии ckpt/снапшота/
durable (запрет №1). demo тянет torch/opening_hours → importorskip гейтит модуль на голом раннере.
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


@pytest.mark.skipif(not _NEED, reason="ckpt/snapshot/system_metrics вне git (запрет №1)")
def test_demo_end_to_end_seed0(tmp_path):
    """demo отрабатывает end-to-end (seed 0, traffic): файлы есть, статик-cost парити per_seed[0],
    served+pending == все клиенты (0004-механика). Динамику НЕ фиксируем точно (wall-clock)."""
    from logistics_rl_gnn.config import instance as im

    s = demo.run_demo(seed=0, event_kind="traffic", out_dir=str(tmp_path), open_maps=False)

    # статик детерминирован → ТОЧНЫЙ парити с durable per_seed[0]
    sm = json.loads(_SM.read_text())
    assert abs(s["static_cost"] - sm["per_seed_cost_eur"][0]) < 0.5

    # выходные файлы существуют и непусты (plan_before/route_sheet/plan_after)
    assert len(s["files"]) == 3
    for f in s["files"]:
        assert Path(f).exists() and Path(f).stat().st_size > 0

    # 0004-механика: обслужено + остаток == все клиенты (traffic: нет urgent-re-delivery)
    inst = im.generate_instance(snapshot_dir=_SNAP, seed=0)
    assert s["n_served"] + s["n_pending"] == len(inst.demand) - 1

    # cost_before ДЕТЕРМИНИРОВАН (greedy+served_by+_continue_old_plan+scorer, без polish/wall-clock)
    # → пиним durable-значение (ловит рассинхром нумерации/своп/знак Δ, чего range-ассерты не ловят)
    assert abs(s["cost_before"] - 578.7) < 2.0
    # cost_after — wall-clock-чувствителен (polish 400мс) → только санити (конечный, положительный)
    assert 0.0 < s["cost_after"] < 5000.0
    assert s["n_moved"] <= s["n_pending"]


def test_continue_old_plan_numbering():
    """policy-free страж рассинхрона нумерации (Finding-1) за ~1с: _continue_old_plan нумерует по
    ПЕРЕДАННОМУ residual-idx (== residual_instance) и исключает served (urgent-re-delivery)."""
    class _St:  # минимальный DynamicState (нужен лишь .served)
        served = {2}  # стоп 2 обслужен, но включён в idx как urgent-re-delivery

    idx = [0, 1, 2, 3, 4]  # residual-нумерация: pending {1,3,4} ∪ urgent {2}
    exec_routes = [[0, 1, 2, 3, 0], [0, 4, 0]]  # маш.1 → 1,2,3; маш.2 → 4
    veh_of = {1: 1, 2: 1, 3: 1, 4: 2}
    old = demo._continue_old_plan(exec_routes, _St(), idx=idx, drop_vehicle=None, veh_of=veh_of)
    visited = {p for r in old for p in r if p != 0}
    assert visited == {1, 3, 4}  # позиции = индексы в idx (буг дал бы {1,2,3} на своей idx)
    assert 2 not in visited  # served urgent-re-delivery в старый план НЕ входит → unserved
    # breakdown: машина 2 выбыла → её стоп (idx-поз 4) осиротел
    bd = demo._continue_old_plan(exec_routes, _St(), idx=idx, drop_vehicle=2, veh_of=veh_of)
    assert {p for r in bd for p in r if p != 0} == {1, 3}
