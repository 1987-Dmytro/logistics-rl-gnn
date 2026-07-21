"""Phase 6b Шаг 0 — наблюдаемость congestion (пламбинг признаков, БЕЗ смены обучения).

Политика ВИДИТ время/пробки; под free-flow — паритет (регрессия). Проверяем:
  - obs несёт congestion-поля нужных форм (node_congestion + time_context);
  - node_congestion ловит зону инцидента (и конечен при закрытии);
  - forward no-NaN на CongestionTravel С ЗАКРЫТИЕМ (inf→сентинел — единственный новый risky-путь);
  - прямой ПАРИТЕТ build_graph под free-flow (edge_attr бит-в-бит, node_congestion≡0);
  - детерминизм forward по seed под congestion.
Главный гейт-регрессия (overfit-tiny сходится к ~78.9 под новым пламбингом) — в
test_model.test_overfit_tiny_cost_drops (гоняет free-flow сквозь этот код). skip без torch.
"""

from __future__ import annotations

import numpy as np
import pytest
from test_env import _tiny_env, _tiny_instance  # реюз Phase 3 tiny

from logistics_rl_gnn.env.travel import CongestionTravel, Incident, time_context


def _incident_env(*, closure: bool):
    """tiny-env с CongestionTravel: инцидент в зоне узла 1 (закрытие или магнитуда)."""
    from logistics_rl_gnn.env.events import make_dynamic_env

    inst = _tiny_instance()
    mag = np.inf if closure else 1.2
    inc = Incident(tuple(inst.coords[1]), 1.2, mag, t_start_min=0.0, duration_min=120.0)
    ct = CongestionTravel(inst.time_matrix / 60.0, inst.coords, dow=1, incidents=[inc])
    return make_dynamic_env(inst, travel=ct, fleet_size=1, t_max_min=1000.0), ct


# ---------- obs-контракт (без torch) ----------


def test_obs_has_congestion_fields_and_shapes():
    env = _tiny_env()
    obs, _ = env.reset(seed=0)
    assert obs["node_features"].shape == (env.k, 9)  # +node_congestion (столбец 8)
    assert obs["time_context"].shape == (4,)
    assert np.allclose(obs["node_features"][:, 8], 0.0)  # free-flow → у узлов нет congestion
    assert np.isfinite(obs["time_context"]).all()
    assert np.all(np.abs(obs["time_context"]) <= 1.0)  # sin/cos ∈ [-1,1]


def test_time_context_pure_function():
    tc = time_context(0.0, 1)
    assert tc.shape == (4,)
    assert np.isfinite(tc).all()
    # разные времена → разный контекст (сигнал живой)
    assert not np.allclose(tc, time_context(180.0, 1))


def test_node_congestion_flags_incident_zone():
    inst = _tiny_instance()
    coords = inst.coords
    inc = Incident(tuple(coords[1]), 1.2, magnitude=1.0, t_start_min=0.0, duration_min=60.0)
    ct = CongestionTravel(inst.time_matrix / 60.0, coords, dow=1, incidents=[inc])
    nc = ct.node_congestion(coords, at_minute=10.0)
    assert nc[1] > 0.0  # узел 1 в зоне инцидента
    assert nc[0] == 0.0 and nc[2] == 0.0  # депо и узел 2 вне радиуса 1.2км
    # закрытие → конечный сентинел (не inf), но всё ещё сигнал
    closed = CongestionTravel(
        inst.time_matrix / 60.0,
        coords,
        dow=1,
        incidents=[Incident(tuple(coords[1]), 1.2, np.inf, 0.0, 60.0)],
    )
    nc_c = closed.node_congestion(coords, 10.0)
    assert np.isfinite(nc_c).all() and nc_c[1] > 0.0


def test_free_flow_node_congestion_zero():
    env = _tiny_env()
    env.reset(seed=0)
    assert np.allclose(env.travel.node_congestion(env.coords, env.cur_time), 0.0)


# ---------- модель (skip без torch/torch-geometric) ----------

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from logistics_rl_gnn.models.policy import VRPPolicy, build_graph  # noqa: E402


def _policy(seed=0):
    torch.manual_seed(seed)
    return VRPPolicy()


def test_build_graph_parity_freeflow():
    """Паритет free-flow: канал 0 == прежней норме (бит-в-бит), канал 1 ≡1, node_cong≡0."""
    env = _tiny_env()
    env.reset(seed=0)
    node_feat, ei, ea = build_graph(env, torch.device("cpu"))
    assert node_feat.shape == (env.k, 8)
    assert ea.shape == (env.k * env.k, 2)  # 2 канала: travel-норма + congestion-множитель
    tm = torch.as_tensor(env.time_m, dtype=torch.float32)
    ea0_old = tm.reshape(-1) / (tm.max() + 1e-8)  # прежняя формула (канал 0)
    assert torch.equal(ea[:, 0], ea0_old)  # канал 0 под free-flow не сдвинулся
    assert torch.all(ea[:, 1] == 1.0)  # канал 1 (множитель) ≡1 под free-flow (нейтрален)
    assert torch.all(node_feat[:, 7] == 0.0)  # node_congestion (столбец 7) нейтрален


def test_edge_channels_carry_congestion():
    """2 edge-канала (Phase 6b Шаг 1, закрывает нюанс 0005): канал 0 (travel-норма) стирает
    равномерный диурнал (топология, max-норма), канал 1 (множитель travel/ff) ЕГО показывает.
    Локальный инцидент сдвигает и канал 0 (переживает max-норму)."""
    inst = _tiny_instance()
    env_ff = _tiny_env()
    env_ff.reset(seed=0)
    _, _, ea_ff = build_graph(env_ff, torch.device("cpu"))
    # чистый диурнал (без инцидентов, час 8 → c=1.30)
    ct = CongestionTravel(inst.time_matrix / 60.0, inst.coords, dow=1)
    from logistics_rl_gnn.env.events import make_dynamic_env

    dv = make_dynamic_env(inst, travel=ct, fleet_size=1, t_max_min=1000.0)
    dv.reset(seed=0)
    _, _, ea_diur = build_graph(dv, torch.device("cpu"))
    assert torch.equal(ea_diur[:, 0], ea_ff[:, 0])  # канал 0: диурнал сокращается (топология)
    off = ea_diur[:, 1][ea_diur[:, 1] != 1.0]  # недиагональные множители (diag=1)
    assert off.numel() > 0 and torch.allclose(off, torch.full_like(off, 1.30), atol=1e-4)  # =c(8)
    # локальный инцидент → канал 0 СДВИГАЕТСЯ (переживает max-норму)
    env_inc, _ = _incident_env(closure=False)
    env_inc.reset(seed=0)
    _, _, ea_inc = build_graph(env_inc, torch.device("cpu"))
    assert not torch.equal(ea_inc[:, 0], ea_ff[:, 0])


def test_forward_no_nan_under_closure():
    """Гейт корректности: encode/forward конечны, когда закрытие даёт inf в travel-матрице."""
    policy = _policy(0)
    env, _ = _incident_env(closure=True)
    obs, _ = env.reset(seed=0)
    node_embs, graph_emb, _ = enc = policy.encode(env)
    _, _, ea = build_graph(env, torch.device("cpu"))
    assert torch.isfinite(ea).all()  # оба edge-канала конечны под закрытием (inf→cap/сентинел)
    assert torch.isfinite(node_embs).all()  # inf→сентинел в build_graph отработал
    assert torch.isfinite(graph_emb).all()
    dist = policy.action_dist(env, obs, enc)
    assert not torch.isnan(dist.probs).any()
    assert torch.allclose(dist.probs.sum(), torch.tensor(1.0), atol=1e-5)
    assert obs["node_features"][1, 8] > 0.0  # узел 1 в зоне закрытия виден в obs


def test_forward_deterministic_with_congestion():
    def fwd():
        policy = _policy(42)
        env, _ = _incident_env(closure=False)
        obs, _ = env.reset(seed=0)
        return policy.action_dist(env, obs, policy.encode(env)).probs.detach()

    p1, p2 = fwd(), fwd()
    assert torch.allclose(p1, p2)  # seed весов → воспроизводимый forward под congestion
    assert not torch.isnan(p1).any()
