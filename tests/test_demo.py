"""Phase 8 — страж demo.py: end-to-end на seed 0. Статик-cost ДЕТЕРМИНИРОВАН → точный парити с
system_metrics per_seed[0]; динамика (residual/latency) wall-clock → фиксируем ТОЛЬКО структурно
(файлы есть, served+pending == все клиенты по 0004-механике, поля в диапазоне) — не точные числа.

Тяжёлый (полный system_routes + re-plan + OR-Tools ~30с) → skipif при отсутствии ckpt/снапшота/
durable (запрет №1). demo тянет torch/opening_hours → importorskip гейтит модуль на голом раннере.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

_FILES = {"1_morning_plan.html", "route_sheet.md", "2_incident_no_replan.html",
          "3_incident_replan.html", "compare.html"}


def _banner_price(path) -> float:
    """Цена из шапки карты (data-demo-price) — машиночитаемый якорь «числа шапок == demo-выводу»."""
    m = re.search(r'data-demo-price="([0-9.]+)"', Path(path).read_text(encoding="utf-8"))
    assert m, f"нет data-demo-price в {path}"
    return float(m.group(1))

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

    # 5 выходных артефактов с самоописательными именами — существуют и непусты
    by_name = {Path(f).name: f for f in s["files"]}
    assert set(by_name) == _FILES
    for f in s["files"]:
        assert Path(f).exists() and Path(f).stat().st_size > 0

    # числа шапок == demo-выводу (парсим data-demo-price из каждой карты)
    assert abs(_banner_price(by_name["1_morning_plan.html"]) - s["morning_cost"]) < 1e-3
    assert abs(_banner_price(by_name["2_incident_no_replan.html"]) - s["cost_before"]) < 1e-3
    assert abs(_banner_price(by_name["3_incident_replan.html"]) - s["cost_after"]) < 1e-3

    # compare.html — кадр A/B/C: ссылается на обе карты И показывает стоимость OR-Tools
    # (запрет №3: честный бейзлайн не спрятан в презентационном артефакте)
    comp = Path(by_name["compare.html"]).read_text(encoding="utf-8")
    assert "2_incident_no_replan.html" in comp and "3_incident_replan.html" in comp
    assert f"{s['or_cost']:.1f} €" in comp

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
    """policy-free страж рассинхрона нумерации за ~1с: _continue_old_plan нумерует по ПЕРЕДАННОМУ
    residual-idx (== residual_instance) и исключает served (urgent-re-delivery). idx НЕ-тождествен,
    с дырой → pos[b]≠b: раг-index-буг (append b вместо pos[b]) выдал бы ДРУГОЙ набор."""
    class _St:  # минимальный DynamicState (нужен лишь .served)
        served = {2, 3}  # 2 — urgent-re-delivery (в idx, но обслужен); 3 — обычный served (вне idx)

    idx = [0, 1, 2, 4]  # residual-нумерация: pending {1,4} ∪ urgent {2}; стоп 3 обслужен → НЕ в idx
    exec_routes = [[0, 1, 2, 3, 0], [0, 4, 0]]  # маш.1 → 1,2,3; маш.2 → 4
    veh_of = {1: 1, 2: 1, 3: 1, 4: 2}
    old = demo._continue_old_plan(exec_routes, _St(), idx=idx, drop_vehicle=None, veh_of=veh_of)
    visited = {p for r in old for p in r if p != 0}
    # позиции в idx: стоп 1→pos1, стоп 4→pos3. Раг-index-буг дал бы {1,4} (сами стопы) — тест ловит.
    assert visited == {1, 3}
    assert 2 not in visited  # served urgent-re-delivery (в idx) в старый план НЕ входит → unserved
    # breakdown: машина 2 выбыла → её стоп (4, idx-поз 3) осиротел; остаётся лишь стоп 1 (поз 1)
    bd = demo._continue_old_plan(exec_routes, _St(), idx=idx, drop_vehicle=2, veh_of=veh_of)
    assert {p for r in bd for p in r if p != 0} == {1}


@pytest.mark.skipif(not (_SNAP / "graph.graphml").exists(),
                    reason="graph.graphml вне git (запрет №1)")
def test_road_polyline_on_graph_nodes():
    """Дорожная геометрия: полилиния хопа реальными улицами (nx.shortest_path) — реальный путь
    (>2 вершин, не прямая-fallback) И каждая вершина ЛЕЖИТ на узле графа (x/y узла)."""
    import osmnx as ox
    import pandas as pd

    g = ox.load_graphml(_SNAP / "graph.graphml")
    coords = {(round(g.nodes[p]["y"], 7), round(g.nodes[p]["x"], 7)) for p in g.nodes}
    ids = pd.read_parquet(_SNAP / "nodes.parquet")["node_id"].astype(int).tolist()
    line = demo._road_latlon(g, ids[0], ids[1], {})  # депо→аптека связны → реальный путь
    assert len(line) > 2, "ожидали реальный путь по улицам, а не прямую-fallback"
    for lat, lon in line:
        assert (round(lat, 7), round(lon, 7)) in coords  # вершина == узел графа
