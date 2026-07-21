"""Phase 7 dynamic-тесты. Паритет с FreeFlow + env_checker + логика событий — всегда;
re-plan/латентность (главный гейт) — на снапшоте (skip без него) + torch/ortools.
"""

from __future__ import annotations

import math
from datetime import datetime

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from logistics_rl_gnn.config import congestion as cg
from logistics_rl_gnn.config import instance as im
from logistics_rl_gnn.env.events import (
    BreakdownEvent,
    DiurnalTick,
    DynamicState,
    UrgentEvent,
    congestion_for,
    make_dynamic_env,
    residual_instance,
    served_by,
)
from logistics_rl_gnn.env.travel import FreeFlowTravel, Incident


def _tiny_instance(seed=0):
    time_m = np.array([[0, 600, 1200], [600, 0, 900], [1200, 900, 0]], dtype=float)
    dist_m = np.array([[0, 5000, 10000], [5000, 0, 7500], [10000, 7500, 0]], dtype=float)
    return im.Instance(
        node_ids=[0, 1, 2],
        snapshot_stops=[0, 1, 2],
        kinds=["depot", "pharmacy", "pharmacy"],
        time_matrix=time_m,
        dist_matrix=dist_m,
        coords=np.array([[10.0, 48.0], [10.1, 48.1], [10.2, 48.0]]),
        windows=np.array([[0, 36000], [0, 36000], [0, 36000]], dtype=float),
        demand=np.array([0.0, 10.0, 20.0]),
        service=np.array([0.0, 240.0, 240.0]),
        tw_source=["DEPOT", "REAL", "REAL"],
        excluded_stops=[],
        start_datetime=datetime(2024, 1, 2, 8),
        horizon_s=36000,
        meta={"seed": seed},
    )


# ---------- паритет + env_checker (без снапшота) ----------


def test_congestion_parity_with_freeflow():
    # c≡1 (flat_c) И нет активных инцидентов ⇒ ровно FreeFlow на любом at_minute
    inst = _tiny_instance()
    t0 = np.asarray(inst.time_matrix) / 60.0
    ff = FreeFlowTravel(t0)
    ct = congestion_for(inst, dow=1, flat_c=True)
    for i in (0, 1, 2):
        for j in (0, 1, 2):
            for at in (0.0, 90.0, 300.0):
                assert ct.time(i, j, at) == pytest.approx(ff.time(i, j))


def test_congestion_drop_in_survives_reset():
    # env НЕ переписан: CongestionTravel живёт в env.travel после reset (rollout/greedy сбрасывают)
    inst = _tiny_instance()
    ct = congestion_for(inst, dow=1)
    env = make_dynamic_env(inst, travel=ct, fleet_size=1, t_max_min=1000.0)
    env.reset(seed=0)
    assert env.travel is ct  # не откатился к FreeFlow


def test_env_checker_congestion():
    inst = _tiny_instance()
    env = make_dynamic_env(inst, travel=congestion_for(inst, dow=1), fleet_size=1, t_max_min=1000.0)
    check_env(env, skip_render_check=True)


def test_traffic_incident_raises_in_zone_only():
    inst = _tiny_instance()
    inc = Incident(tuple(inst.coords[1]), cg.INCIDENT_RADIUS_KM, 1.0, 40.0, 40.0)  # зона узла 1
    ct_no = congestion_for(inst, dow=1)  # только диурнал
    ct = congestion_for(inst, dow=1, incidents=[inc])  # диурнал + инцидент
    # в один момент (at=50, в окне инцидента): в зоне (ребро 0→1) инцидент поднимает время,
    # вне зоны (ребро 0→2) — совпадает с чистым диурналом (инцидент геометрически локален)
    assert ct.time(0, 1, 50.0) > ct_no.time(0, 1, 50.0)
    assert ct.time(0, 2, 50.0) == pytest.approx(ct_no.time(0, 2, 50.0))
    # вне окна инцидента (at=200) даже в зоне — снова только диурнал
    assert ct.time(0, 1, 200.0) == pytest.approx(ct_no.time(0, 1, 200.0))


def test_edge_closure_removes_edge():
    inst = _tiny_instance()
    inc = Incident(tuple(inst.coords[1]), cg.INCIDENT_RADIUS_KM, math.inf, 40.0, 40.0)  # закрытие
    ct = congestion_for(inst, dow=1, incidents=[inc])
    assert math.isinf(ct.time(0, 1, 50.0))  # закрыто в окне
    assert math.isfinite(ct.time(0, 1, 200.0))  # снова открыто вне окна


def test_breakdown_reduces_fleet():
    st = DynamicState(_tiny_instance(), dow=1)
    assert st.fleet(8) == 8
    assert BreakdownEvent(30.0).apply(st) is True  # триггер
    assert st.broken == 1 and st.fleet(8) == 7


def test_urgent_adds_node_with_tight_window():
    inst = _tiny_instance()
    st = DynamicState(inst, dow=1)
    st.served = {1, 2}  # обе обслужены
    ev = UrgentEvent(30.0, {"idx": 1, "demand": 7, "delta_s": 1800.0})
    assert ev.apply(st) is True
    res = residual_instance(st)
    # срочный узел 1 включён (несмотря на served), окно узкое [0, 1800]сек
    assert inst.snapshot_stops[1] in res.snapshot_stops
    k = res.snapshot_stops.index(inst.snapshot_stops[1])
    assert res.windows[k, 0] == pytest.approx(0.0)
    assert res.windows[k, 1] == pytest.approx(1800.0)
    assert res.demand[k] == 7


def test_diurnal_tick_does_not_trigger():
    st = DynamicState(_tiny_instance(), dow=1)
    assert DiurnalTick(120.0).apply(st) is False  # плавный c(h) — НЕ re-plan


# ---------- re-plan / латентность (снапшот + torch/ortools) ----------

_NEED_SNAP = pytest.mark.skipif(im._latest_snapshot_dir() is None, reason="нет снапшота")


def _snapshot_residual(now_min=40.0, incident=True, broken=0):
    """Реальный residual в момент now_min: частичное состояние из greedy-плана + опц. инцидент."""
    from logistics_rl_gnn.baselines.greedy import greedy_routes

    inst = im.generate_instance(seed=0)
    dow = im.DELIVERY_WEEKDAY
    exec_travel = congestion_for(inst, dow=dow)
    exec_routes = greedy_routes(env=make_dynamic_env(inst, travel=exec_travel))
    st = DynamicState(inst, dow)
    st.now_min = now_min
    st.served = served_by(exec_routes, inst, exec_travel, now_min)
    st.broken = broken
    if incident:
        st.incidents = [Incident(tuple(inst.coords[1]), cg.INCIDENT_RADIUS_KM, 1.2, now_min, 45.0)]
    res = residual_instance(st)
    travel = congestion_for(res, dow=dow, offset_min=now_min, incidents=st.incidents)
    return res, travel, st, inst


@_NEED_SNAP
def test_replan_feasible_under_events():
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    pytest.importorskip("ortools")
    import torch

    from logistics_rl_gnn.models.policy import VRPPolicy
    from logistics_rl_gnn.replan.compare import compare_replan

    torch.manual_seed(0)
    res, travel, st, _ = _snapshot_residual(now_min=40.0, incident=True, broken=1)
    assert len(res.demand) - 1 > 5, "residual должен быть содержательным на раннем событии"
    out = compare_replan(res, travel, VRPPolicy(), fleet_size=st.fleet(8), deadline_s=1)
    # env-маскинг (cap/TW/T_max под congestion) → RL и greedy без опозданий; возврат ≤ T_max
    for m in ("rl", "greedy"):
        assert out[m]["on_time_pct"] == pytest.approx(100.0), f"{m}: опоздания под событием"
        assert out[m]["unserved"] >= 0


@_NEED_SNAP
def test_main_gate_rl_latency_orders_below_ortools():
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    pytest.importorskip("ortools")
    import torch

    from logistics_rl_gnn.models.policy import VRPPolicy
    from logistics_rl_gnn.replan.compare import compare_replan

    torch.manual_seed(0)
    res, travel, st, _ = _snapshot_residual(now_min=40.0, incident=True, broken=0)
    out = compare_replan(res, travel, VRPPolicy(), fleet_size=st.fleet(8), deadline_s=1)
    rl_ms, ort_ms = out["rl"]["latency_ms"], out["ortools"]["latency_ms"]
    # ГЛАВНЫЙ ГЕЙТ: RL реагирует на порядки быстрее OR-Tools + абсолютный потолок (не тавтология)
    assert ort_ms / rl_ms > 20.0, f"RL не на порядки быстрее: RL={rl_ms:.1f}мс OR={ort_ms:.1f}мс"
    assert rl_ms < 100.0, f"RL-латентность {rl_ms:.1f}мс — реакция должна быть суб-100мс"
