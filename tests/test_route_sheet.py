"""Phase 8 — стражи route sheet: walk-семантика (монотонность/окна/загрузка/cost==scorer) на
дешёвом синтет-инстансе + парити cost с system_metrics per_seed (durable, skip-guard).

Структурные тесты БЕЗ снапшота/солвера (голый раннер) — синтет Instance в памяти. Парити читает
durable results/*.json (вне git, запрет №1) → skipif. route_sheet тянет torch/opening_hours →
importorskip гейтит модуль на раннере без [dev,env].
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "scripts"))

rs = pytest.importorskip("route_sheet")  # skips на голом раннере (torch/opening_hours отсутствуют)

from logistics_rl_gnn.config.instance import Instance  # noqa: E402
from logistics_rl_gnn.env.scoring import CostConfig, evaluate_solution  # noqa: E402


def _toy_instance() -> Instance:
    """Депо + 3 аптеки, все стопы достижимы в окнах (dt>0 → времена строго растут)."""
    t = np.array([  # сек
        [0, 600, 900, 1200],
        [600, 0, 300, 900],
        [900, 300, 0, 300],
        [1200, 900, 300, 0],
    ], dtype=float)
    win = np.array([[0, 36000], [0, 36000], [3600, 36000], [0, 36000]], dtype=float)  # сек
    return Instance(
        node_ids=[0, 1, 2, 3], snapshot_stops=[0, 1, 2, 3],
        kinds=["depot", "pharmacy", "pharmacy", "pharmacy"],
        time_matrix=t, dist_matrix=t * 10.0, coords=np.zeros((4, 2)),
        windows=win, demand=np.array([0, 3, 5, 7]),
        service=np.array([0, 240, 240, 240], dtype=float),
        tw_source=["DEPOT", "REAL", "REAL", "ASSUMED"], excluded_stops=[],
        start_datetime=datetime(2024, 1, 2, 8, 0), horizon_s=36000, meta={},
    )


def test_walk_times_monotone():
    """Прибытия строго возрастают вдоль маршрута (t накапливается, dt>0)."""
    stops, _ = rs.walk_route([0, 1, 2, 3, 0], _toy_instance())
    arr = [s["arr_min"] for s in stops]
    assert arr == sorted(arr) and len(set(arr)) == len(arr)


def test_walk_exact_per_stop_values():
    """Точные per-stop значения пиннятся: arr_min и КУМУЛЯТИВНАЯ load_after (регрессия отображаемых
    колонок, что монотонность/Σ не ловят). Ручной walk: dt=10/5/5′, e2=60′→ждём; загрузка 3/5/7."""
    stops, _ = rs.walk_route([0, 1, 2, 3, 0], _toy_instance())
    assert [round(s["arr_min"]) for s in stops] == [10, 19, 69]  # прибытия, не отправления
    assert [s["load_after"] for s in stops] == [3, 8, 15]  # кумулятив, не per-stop спрос


def test_late_arrival_breaks_on_time():
    """Тесное REAL-l (arrival > l) → on_time_pct < 100 — страж честной шапки (не хардкод «100%»)."""
    inst = _toy_instance()
    inst.windows[2] = [0.0, 600.0]  # стоп 2: l = 10′ (сек), прибытие 19′ → опоздание
    sheet = rs.build_sheet([[0, 1, 2, 3, 0]], inst)
    assert sheet["totals"]["on_time_pct"] < 100.0


def test_arrival_within_real_windows():
    """arrival ≤ l_i для ВСЕХ REAL-окон (жёсткое TW, не через штраф)."""
    sheet = rs.build_sheet([[0, 1, 2, 3, 0]], _toy_instance())
    for v in sheet["vehicles"]:
        for s in v["stops"]:
            if s["tw_source"] == "REAL":
                assert s["arr_min"] <= s["l_min"] + 1e-6


def test_load_sums_to_demand():
    """Σ загрузки по машинам == суммарный спрос (все стопы обслужены)."""
    inst = _toy_instance()
    sheet = rs.build_sheet([[0, 1, 2, 3, 0]], inst)
    assert sheet["totals"]["boxes"] == int(inst.demand.sum())


def test_load_sums_to_demand_split():
    """То же при разбиении на 2 машины (сумма инвариантна к разбиению)."""
    inst = _toy_instance()
    sheet = rs.build_sheet([[0, 1, 0], [0, 2, 3, 0]], inst)
    assert sheet["totals"]["boxes"] == int(inst.demand.sum())
    assert sheet["totals"]["vehicles_used"] == 2


def test_walk_cost_matches_scorer():
    """Cost листа (из walk-итогов) == evaluate_solution (единый scorer) — парити walk/scorer."""
    inst = _toy_instance()
    routes = [[0, 1, 2, 3, 0]]
    sheet = rs.build_sheet(routes, inst)
    q = evaluate_solution(routes, inst, CostConfig())
    assert abs(sheet["cost_eur"] - (-q["reward"])) < 1e-6
    # duty = езда + ожидание + сервис == time_min scorer'а
    T = sheet["totals"]
    assert abs((T["drive"] + T["wait"] + T["service"]) - T["time_min"]) < 1e-6
    assert abs(T["time_min"] - q["time_min"]) < 1e-6


_RS = _ROOT / "results" / "route_sheet.json"
_SM = _ROOT / "results" / "system_metrics.json"


@pytest.mark.skipif(not (_RS.exists() and _SM.exists()), reason="durable results/*.json вне git")
def test_route_sheet_cost_parity_with_system_metrics():
    """Cost листа == system_metrics per_seed[seed] (лист = РОВНО тот план — запреты №3/№4)."""
    rsd = json.loads(_RS.read_text())
    sm = json.loads(_SM.read_text())
    seed = rsd["seed"]
    assert abs(rsd["cost_eur"] - sm["per_seed_cost_eur"][seed]) < 0.5
    assert abs(rsd["cost_eur"] - rsd["anchor_per_seed"]) < 1e-6
