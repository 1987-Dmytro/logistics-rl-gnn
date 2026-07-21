"""Тест VRPEnv (Phase 3 core). env_checker + tiny-reward идут всегда; property/детерминизм
на реальном снапшоте — skip без него.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from logistics_rl_gnn.config import instance as im
from logistics_rl_gnn.env.vrp_env import VRPEnv


def _tiny_instance(seed=0):
    # депо(0) + 2 аптеки(1,2); секунды/метры (env переведёт в мин/км)
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
        service=np.array([0.0, 240.0, 240.0]),  # 4 мин
        tw_source=["DEPOT", "REAL", "REAL"],
        excluded_stops=[],
        start_datetime=datetime(2024, 1, 2, 8),
        horizon_s=36000,
        meta={},
    )


def _tiny_env(**kw):
    return VRPEnv(instance_fn=_tiny_instance, fleet_size=1, t_max_min=1000.0, **kw)


def test_env_checker():
    check_env(_tiny_env(), skip_render_check=True)


def test_tiny_reward():
    env = _tiny_env()  # c_f=50, c_d=0.5, c_t=20 (дефолт cfg)
    env.reset(seed=0)
    _, r1, term1, _, _ = env.step(1)
    assert not term1 and r1 == 0.0  # разрежённый: до терминала 0
    _, r2, term2, trunc2, info = env.step(2)
    assert term2 and not trunc2
    # ручной расчёт (мин/км), маршрут 0→1→2→0:
    #   время = 10 + 4 + 15 + 4 + 20 = 53 мин; пробег = 5 + 7.5 + 10 = 22.5 км; машин=1
    expected = -(50.0 * 1 + 0.5 * 22.5 + 20.0 * 53.0 / 60.0)
    assert r2 == pytest.approx(expected)
    assert info["vehicles_used"] == 1
    assert info["total_km"] == pytest.approx(22.5)
    assert info["total_time_min"] == pytest.approx(53.0)
    assert info["n_unserved"] == 0
    assert info["routes"] == [[0, 1, 2, 0]]


def test_dense_matches_sparse_total():
    # dense раздаёт variable по шагам, sparse — терминалом; суммарный reward эпизода
    # ОДИНАКОВ и равен единому evaluate_solution (одна формула, разное credit-assignment)
    from logistics_rl_gnn.env.scoring import CostConfig, evaluate_solution

    def rollout(dense):
        env = _tiny_env(dense_reward=dense)
        env.reset(seed=0)
        total, info = 0.0, {}
        for a in (1, 2):
            _, r, term, trunc, info = env.step(a)
            total += r
            if term or trunc:
                break
        return total, info["routes"]

    tot_sparse, routes = rollout(False)
    tot_dense, _ = rollout(True)
    expected = evaluate_solution(routes, _tiny_instance(), CostConfig())["reward"]
    assert tot_sparse == pytest.approx(expected)
    assert tot_dense == pytest.approx(expected)


def test_invalid_action_terminates():
    env = _tiny_env()
    env.reset(seed=0)
    env.step(1)  # аптека 1 посещена
    _, r, term, _, info = env.step(1)  # снова 1 — не по маске
    assert term and r == pytest.approx(-im.PENALTY_INVALID)
    assert info["invalid_action"] == 1


# ---------- на реальном снапшоте (skip без снапшота) ----------


def _snap_ok():
    return im._latest_snapshot_dir() is not None


@pytest.mark.skipif(not _snap_ok(), reason="нет снапшота: python scripts/build_snapshot.py")
def test_env_checker_snapshot():
    check_env(VRPEnv(), skip_render_check=True)


@pytest.mark.skipif(not _snap_ok(), reason="нет снапшота")
def test_property_invariants():
    k, q = im.FLEET_SIZE, im.VEHICLE_CAP
    for seed in range(5):
        env = VRPEnv()
        obs, _ = env.reset(seed=seed)
        term = trunc = False
        info = {}
        while not (term or trunc):
            obs, _, term, trunc, info = env.step(env.sample_valid_action(obs))
        routes = info["routes"]
        served = [n for rt in routes for n in rt if n != 0]
        for rt in routes:  # старт/финиш в депо
            assert rt[0] == 0 and rt[-1] == 0
        assert len(served) == len(set(served)), "аптека посещена >1 раза"
        assert info["vehicles_used"] <= k
        assert info["late_min"] == 0.0  # masking → без опозданий (TW соблюдены)
        inst = env._instance_fn(seed)
        for rt in routes:  # cap не нарушен на маршруте
            assert sum(inst.demand[n] for n in rt if n != 0) <= q + 1e-9
        assert sum(inst.demand[n] for n in served) <= k * q + 1e-9  # спрос ≤ K*Q


@pytest.mark.skipif(not _snap_ok(), reason="нет снапшота")
def test_determinism():
    def rollout(seed):
        env = VRPEnv()
        obs, _ = env.reset(seed=seed)
        acts, rews = [], []
        term = trunc = False
        while not (term or trunc):
            a = env.sample_valid_action(obs)
            obs, r, term, trunc, _ = env.step(a)
            acts.append(a)
            rews.append(r)
        return acts, rews

    a1, r1 = rollout(123)
    a2, r2 = rollout(123)
    assert a1 == a2 and r1 == pytest.approx(r2)
