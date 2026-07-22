"""Phase 6b Шаг 3.5 — local-search polish декодированных маршрутов.

Стражи (порядок = порядок сборки): оракул check_feasible (parity с env-семантикой + границы
cap/TW/T_max + agreement с evaluate_solution) → потом операторы polish (не ухудшает, feasibility
сохранена, детерминизм, бюджет, parity-оптимум).
"""

from __future__ import annotations

import itertools
from datetime import datetime

import numpy as np
import pytest

from logistics_rl_gnn.config import instance as im
from logistics_rl_gnn.env.scoring import CostConfig, check_feasible, evaluate_solution


def _mk(time_min, win_min, demand, svc_min, *, n):
    """Контролируемый инстанс: времена/окна/сервис в МИНУТАХ (внутри → сек, как реальный)."""
    tm = np.asarray(time_min, dtype=float) * 60.0
    dm = np.asarray(time_min, dtype=float) * 100.0  # дистанция произвольна (не влияет на feas)
    return im.Instance(
        node_ids=list(range(n)),
        snapshot_stops=list(range(n)),
        kinds=["depot"] + ["pharmacy"] * (n - 1),
        time_matrix=tm,
        dist_matrix=dm,
        coords=np.column_stack([np.linspace(10, 10.2, n), np.linspace(48, 48.2, n)]),
        windows=np.asarray(win_min, dtype=float) * 60.0,
        demand=np.asarray(demand, dtype=float),
        service=np.asarray(svc_min, dtype=float) * 60.0,
        tw_source=["DEPOT"] + ["REAL"] * (n - 1),
        excluded_stops=[],
        start_datetime=datetime(2024, 1, 2, 8),
        horizon_s=100000,
        meta={"seed": 0},
    )


def _boundary_inst():
    """3 узла, симметрично 10мин/ребро; c1,c2 demand 40, service 5мин; окна [0,100]."""
    t = [[0, 10, 10], [10, 0, 10], [10, 10, 0]]
    return _mk(t, [[0, 1000], [0, 100], [0, 100]], [0, 40, 40], [0, 5, 5], n=3)


# route [0,1,2,0]: arrive1=10 (+svc5→15), arrive2=25 (+svc5→30), return=40мин; load=80.


def test_check_feasible_boundary_cap():
    inst = _boundary_inst()
    r = [[0, 1, 2, 0]]
    assert check_feasible(r, inst, t_max_min=240, vehicle_cap=80, fleet_size=8)  # load 80 == Q
    assert not check_feasible(r, inst, t_max_min=240, vehicle_cap=79, fleet_size=8)  # 80 > 79


def test_check_feasible_boundary_tmax():
    inst = _boundary_inst()
    r = [[0, 1, 2, 0]]
    assert check_feasible(r, inst, t_max_min=40, vehicle_cap=80, fleet_size=8)  # возврат ровно 40
    assert not check_feasible(r, inst, t_max_min=39.9, vehicle_cap=80, fleet_size=8)


def test_check_feasible_boundary_tw():
    # окно c1 = [0, 9]: прибытие 10 > 9 → TW нарушено (env-жёстко, не через штраф)
    t = [[0, 10, 10], [10, 0, 10], [10, 10, 0]]
    inst = _mk(t, [[0, 1000], [0, 9], [0, 100]], [0, 40, 40], [0, 5, 5], n=3)
    assert not check_feasible([[0, 1, 2, 0]], inst, t_max_min=240, vehicle_cap=80, fleet_size=8)


def test_check_feasible_fleet_cap():
    inst = _boundary_inst()
    two = [[0, 1, 0], [0, 2, 0]]  # 2 непустых маршрута
    assert check_feasible(two, inst, t_max_min=240, vehicle_cap=80, fleet_size=2)
    assert not check_feasible(two, inst, t_max_min=240, vehicle_cap=80, fleet_size=1)


def test_check_feasible_per_customer_return_asymmetric():
    """Асимметричный возврат (не-метрич., как OSM): return≤T_max проверяется ПОСЛЕ КАЖДОГО клиента.

    Регрессия на находку review: [0,1,2,0] завершается за 20мин, НО прямой возврат из узла 1
    стоит 100 → env не построил бы (10+100>50). check_feasible обязан отвергнуть (строго как env).
    """
    t = [[0, 10, 10], [100, 0, 5], [10, 5, 0]]  # t(1,0)=100 — дорогой прямой возврат из 1
    inst = _mk(t, [[0, 1000]] * 3, [0, 10, 10], [0, 0, 0], n=3)
    assert not check_feasible([[0, 1, 2, 0]], inst, t_max_min=50, vehicle_cap=80, fleet_size=8)
    assert check_feasible([[0, 1, 2, 0]], inst, t_max_min=120, vehicle_cap=80, fleet_size=8)


def test_check_feasible_agrees_with_env_on_real_greedy():
    """Реальный greedy-план из env феасибл по check_feasible (тот же источник семантики)."""
    from logistics_rl_gnn.baselines.greedy import greedy_routes
    from logistics_rl_gnn.env.vrp_env import VRPEnv

    env = VRPEnv()
    routes = greedy_routes(env=env, seed=0)
    assert check_feasible(
        routes, env._inst, t_max_min=im.T_MAX_MIN, vehicle_cap=im.VEHICLE_CAP, fleet_size=env.K
    )


def test_check_feasible_agreement_with_evaluate_tw():
    """Agreement: при ok cap/T_max/fleet, TW-вердикт check_feasible == (on_time 100% в evaluate)."""
    t = [[0, 10, 10], [10, 0, 10], [10, 10, 0]]
    ok = _mk(t, [[0, 1000], [0, 100], [0, 100]], [0, 10, 10], [0, 5, 5], n=3)
    late = _mk(t, [[0, 1000], [0, 9], [0, 100]], [0, 10, 10], [0, 5, 5], n=3)
    r = [[0, 1, 2, 0]]
    cfg = CostConfig()
    assert check_feasible(r, ok, t_max_min=240, vehicle_cap=80, fleet_size=8)
    assert evaluate_solution(r, ok, cfg)["on_time_pct"] == pytest.approx(100.0)
    assert not check_feasible(r, late, t_max_min=240, vehicle_cap=80, fleet_size=8)
    assert evaluate_solution(r, late, cfg)["on_time_pct"] < 100.0  # оба видят опоздание


# ---------- операторы polish ----------

import time  # noqa: E402

from logistics_rl_gnn.replan.local_search import (  # noqa: E402
    _or_opt,
    _relocate,
    _swap,
    _two_opt,
    polish,
)


def _custs(routes):
    """Мультимножество клиентов (не-депо) во всех маршрутах — для проверки консервации."""
    return sorted(n for r in routes for n in r if n != 0)


@pytest.mark.parametrize("op", [_two_opt, _or_opt, _relocate, _swap])
def test_operator_conserves_customers_and_structure(op):
    """Оператор: кандидаты сохраняют мультимножество клиентов и depot-концы (нет потери/дубля)."""
    routes = [[0, 1, 2, 3, 0], [0, 4, 5, 0]]
    base = _custs(routes)
    n = 0
    for cand in op(routes):
        assert _custs(cand) == base, f"{op.__name__}: клиенты изменились"
        for r in cand:
            assert r[0] == 0 and r[-1] == 0, f"{op.__name__}: depot-конец сломан"
        n += 1
    assert n > 0, f"{op.__name__}: не сгенерировал кандидатов"


def _real_greedy(seed, travel_none=True):
    from logistics_rl_gnn.baselines.greedy import greedy_routes
    from logistics_rl_gnn.env.vrp_env import VRPEnv

    env = VRPEnv()
    routes = greedy_routes(env=env, seed=seed)
    return routes, env._inst


@pytest.mark.parametrize("seed", range(6))
def test_polish_never_worse_and_feasible(seed):
    """Инвариант: polish ≤ вход И феасибл, на многих реальных сидах (full-62 free-flow)."""
    routes, inst = _real_greedy(seed)
    cfg = CostConfig()
    base = -evaluate_solution(routes, inst, cfg)["reward"]
    out, cost = polish(routes, inst, None, budget_ms=300.0, fleet_size=im.FLEET_SIZE)
    assert cost <= base + 1e-6, f"seed {seed}: polish ухудшил ({cost} > {base})"
    assert check_feasible(
        out, inst, t_max_min=im.T_MAX_MIN, vehicle_cap=im.VEHICLE_CAP, fleet_size=im.FLEET_SIZE
    ), f"seed {seed}: polish вернул инфеасибл"
    assert _custs(out) == _custs(routes), f"seed {seed}: pot/polish потерял клиентов"


def test_polish_never_returns_env_infeasible_asymmetric():
    """Money-path (репро verifier'а): дешёвый 2-opt-сосед env-инфеасибл — polish не берёт.

    4 узла, T_max=50: вход [0,1,3,2,0] феасибл (возврат из 1 = 48≤50); реверс → [0,3,1,2,0] дешевле
    (20<21), НО возврат из 1 = 52>50 (env-инфеасибл). polish обязан вернуть феасибл (строгий check).
    """
    t = [[0, 6, 5, 5], [42, 0, 5, 5], [5, 5, 0, 5], [5, 5, 5, 0]]
    inst = _mk(t, [[0, 1000]] * 4, [0, 10, 10, 10], [0, 0, 0, 0], n=4)
    inp = [[0, 1, 3, 2, 0]]
    assert check_feasible(inp, inst, t_max_min=50, vehicle_cap=80, fleet_size=8)
    out, _ = polish(inp, inst, None, budget_ms=2000.0, t_max_min=50, fleet_size=8)
    ok = check_feasible(out, inst, t_max_min=50, vehicle_cap=80, fleet_size=8)
    assert ok, "polish вернул env-инфеасибл"


def test_polish_deterministic_at_convergence():
    """Детерминизм при сходимости (нет set/dict-итерации). МАЛЫЙ инстанс → сходимость за мс, бюджет
    не связывает даже под нагрузкой (на n=62 бюджет мог бы обрезать → wall-clock-недетерминизм)."""
    rng = np.random.default_rng(3)
    n = 9
    coords = rng.uniform(0, 8, size=(n, 2))
    d = np.hypot(coords[:, None, 0] - coords[None, :, 0], coords[:, None, 1] - coords[None, :, 1])
    inst = _mk(d.tolist(), [[0, 100000]] * n, [0.0] + [10.0] * (n - 1), [0] + [2] * (n - 1), n=n)
    routes = [[0, 1, 2, 3, 4, 0], [0, 5, 6, 7, 8, 0]]
    a = polish(routes, inst, None, budget_ms=5000.0, t_max_min=100000, fleet_size=4)
    b = polish(routes, inst, None, budget_ms=5000.0, t_max_min=100000, fleet_size=4)
    assert a[0] == b[0] and a[1] == pytest.approx(b[1])
    raw = -evaluate_solution(routes, inst, CostConfig())["reward"]
    assert a[1] <= raw + 1e-9  # реально сошёлся (не хуже входа)


def test_polish_respects_budget():
    """Малый бюджет → возврат в пределах бюджета + слак (проверка дедлайна перед каждым eval)."""
    routes, inst = _real_greedy(0)
    t0 = time.perf_counter()
    polish(routes, inst, None, budget_ms=40.0, fleet_size=im.FLEET_SIZE)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert elapsed_ms < 40.0 + 200.0, f"бюджет 40мс, ушло {elapsed_ms:.0f}мс"


def test_polish_leaves_optimum_untouched():
    """Parity: на brute-force-оптимуме tiny polish НЕ ломает оптимальное (возвращает его)."""
    # 3 клиента, 1 машина; асимметричные времена → единственный дешёвый порядок
    t = [[0, 5, 9, 20], [5, 0, 6, 15], [9, 6, 0, 7], [20, 15, 7, 0]]
    inst = _mk(t, [[0, 1000]] * 4, [0, 10, 10, 10], [0, 2, 2, 2], n=4)
    cfg = CostConfig()

    best_r, best_c = None, float("inf")
    for perm in itertools.permutations([1, 2, 3]):
        r = [[0, *perm, 0]]
        if not check_feasible(r, inst, t_max_min=240, vehicle_cap=80, fleet_size=1):
            continue
        c = -evaluate_solution(r, inst, cfg)["reward"]
        if c < best_c:
            best_r, best_c = r, c
    out, cost = polish(best_r, inst, None, budget_ms=3000.0, fleet_size=1)
    assert cost == pytest.approx(best_c), "polish изменил стоимость оптимума"
    assert _custs(out) == _custs(best_r)
