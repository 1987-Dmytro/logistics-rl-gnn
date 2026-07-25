"""Phase 6b Step 0 — congestion observability (feature plumbing, WITHOUT changing training).

The policy SEES time/traffic; under free-flow — parity (regression). We check:
  - obs carries the congestion fields with the right shapes (node_congestion + time_context);
  - node_congestion catches an incident zone (and stays finite on a closure);
  - forward has no NaN on CongestionTravel WITH A CLOSURE (inf→sentinel is the only risky path);
  - direct PARITY of build_graph under free-flow (edge_attr bit for bit, node_congestion≡0);
  - determinism of forward by seed under congestion.
The main regression gate (overfit-tiny converges to ~78.9 under the new plumbing) lives in
test_model.test_overfit_tiny_cost_drops (it runs free-flow through this code). Skipped w/o torch.
"""

from __future__ import annotations

import numpy as np
import pytest
from test_env import _tiny_env, _tiny_instance  # reuse the Phase 3 tiny

from logistics_rl_gnn.env.travel import CongestionTravel, Incident, time_context


def _incident_env(*, closure: bool):
    """tiny env with CongestionTravel: an incident in the zone of node 1 (closure or magnitude)."""
    from logistics_rl_gnn.env.events import make_dynamic_env

    inst = _tiny_instance()
    mag = np.inf if closure else 1.2
    inc = Incident(tuple(inst.coords[1]), 1.2, mag, t_start_min=0.0, duration_min=120.0)
    ct = CongestionTravel(inst.time_matrix / 60.0, inst.coords, dow=1, incidents=[inc])
    return make_dynamic_env(inst, travel=ct, fleet_size=1, t_max_min=1000.0), ct


# ---------- the obs contract (no torch) ----------


def test_obs_has_congestion_fields_and_shapes():
    env = _tiny_env()
    obs, _ = env.reset(seed=0)
    assert obs["node_features"].shape == (env.k, 9)  # +node_congestion (column 8)
    assert obs["time_context"].shape == (4,)
    assert np.allclose(obs["node_features"][:, 8], 0.0)  # free-flow → no congestion at nodes
    assert np.isfinite(obs["time_context"]).all()
    assert np.all(np.abs(obs["time_context"]) <= 1.0)  # sin/cos ∈ [-1,1]


def test_time_context_pure_function():
    tc = time_context(0.0, 1)
    assert tc.shape == (4,)
    assert np.isfinite(tc).all()
    # different times → different context (the signal is alive)
    assert not np.allclose(tc, time_context(180.0, 1))


def test_node_congestion_flags_incident_zone():
    inst = _tiny_instance()
    coords = inst.coords
    inc = Incident(tuple(coords[1]), 1.2, magnitude=1.0, t_start_min=0.0, duration_min=60.0)
    ct = CongestionTravel(inst.time_matrix / 60.0, coords, dow=1, incidents=[inc])
    nc = ct.node_congestion(coords, at_minute=10.0)
    assert nc[1] > 0.0  # node 1 is inside the incident zone
    assert nc[0] == 0.0 and nc[2] == 0.0  # the depot and node 2 are outside the 1.2 km radius
    # closure → a finite sentinel (not inf), yet still a signal
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


def test_congestion_matrix_parity_vs_time():
    """Vectorised matrix(at) == element-wise time(i,j,at): 2 incidents (finite+closure), offset,
    nodes on the zone boundary and shifted in latitude (guards per-node midpoint lat in _km)."""
    center = (10.900, 48.370)
    coords = np.array(
        [
            center,  # 0: the zone centre (dist 0)
            [10.900, 48.38076],  # 1: ~1.19 km (inside the 1.2 radius)
            [10.900, 48.38103],  # 2: ~1.22 km (outside)
            [10.915, 48.362],  # 3: shifted in lon+lat (the midpoint lat differs from clat)
            [10.880, 48.378],  # 4: the zone of the second incident
        ],
        dtype=float,
    )
    k = len(coords)
    rng = np.random.default_rng(0)
    t0 = rng.uniform(1.0, 30.0, size=(k, k))
    np.fill_diagonal(t0, 0.0)
    for mag in (1.0, np.inf):  # a finite contribution + a closure
        inc1 = Incident(center, 1.2, mag, t_start_min=10.0, duration_min=90.0)
        inc2 = Incident((10.880, 48.378), 1.0, 0.7, t_start_min=0.0, duration_min=200.0)
        ct = CongestionTravel(t0, coords, dow=1, offset_min=20.0, incidents=[inc1, inc2])
        for at in (0.0, 25.0, 80.0):  # 80 → abs 100 = the inc1 window edge (decay=0 / closure=inf)
            M = ct.matrix(at)
            for i in range(k):
                for j in range(k):
                    ref = ct.time(i, j, at)
                    if np.isinf(ref):
                        assert np.isinf(M[i, j]), (i, j, at, mag)
                    else:
                        assert M[i, j] == pytest.approx(ref, rel=1e-9, abs=1e-9), (i, j, at, mag)


# ---------- the model (skipped without torch/torch-geometric) ----------

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from logistics_rl_gnn.models.policy import VRPPolicy, build_graph  # noqa: E402


def _policy(seed=0):
    torch.manual_seed(seed)
    return VRPPolicy()


def test_build_graph_parity_freeflow():
    """Free-flow parity: channel 0 == the old norm (bit for bit), channel 1 ≡1, node_cong≡0."""
    env = _tiny_env()
    env.reset(seed=0)
    node_feat, ei, ea = build_graph(env, torch.device("cpu"))
    assert node_feat.shape == (env.k, 8)
    assert ea.shape == (env.k * env.k, 2)  # 2 channels: travel norm + congestion multiplier
    tm = torch.as_tensor(env.time_m, dtype=torch.float32)
    ea0_old = tm.reshape(-1) / (tm.max() + 1e-8)  # the old formula (channel 0)
    assert torch.equal(ea[:, 0], ea0_old)  # channel 0 under free-flow did not shift
    assert torch.all(ea[:, 1] == 1.0)  # channel 1 (the multiplier) ≡1 under free-flow (neutral)
    assert torch.all(node_feat[:, 7] == 0.0)  # node_congestion (column 7) is neutral


def test_edge_channels_carry_congestion():
    """2 edge channels (Phase 6b Step 1, closes the 0005 subtlety): channel 0 (travel norm) erases
    a uniform diurnal (topology, max-norm), channel 1 (the travel/ff multiplier) SHOWS it.
    A local incident shifts channel 0 as well (it survives the max-norm)."""
    inst = _tiny_instance()
    env_ff = _tiny_env()
    env_ff.reset(seed=0)
    _, _, ea_ff = build_graph(env_ff, torch.device("cpu"))
    # a pure diurnal (no incidents, hour 8 → c=1.30)
    ct = CongestionTravel(inst.time_matrix / 60.0, inst.coords, dow=1)
    from logistics_rl_gnn.env.events import make_dynamic_env

    dv = make_dynamic_env(inst, travel=ct, fleet_size=1, t_max_min=1000.0)
    dv.reset(seed=0)
    _, _, ea_diur = build_graph(dv, torch.device("cpu"))
    assert torch.equal(ea_diur[:, 0], ea_ff[:, 0])  # channel 0: the diurnal cancels (topology)
    off = ea_diur[:, 1][ea_diur[:, 1] != 1.0]  # off-diagonal multipliers (diag=1)
    assert off.numel() > 0 and torch.allclose(off, torch.full_like(off, 1.30), atol=1e-4)  # =c(8)
    # a local incident → channel 0 SHIFTS (it survives the max-norm)
    env_inc, _ = _incident_env(closure=False)
    env_inc.reset(seed=0)
    _, _, ea_inc = build_graph(env_inc, torch.device("cpu"))
    assert not torch.equal(ea_inc[:, 0], ea_ff[:, 0])


def test_forward_no_nan_under_closure():
    """Correctness gate: encode/forward stay finite when a closure puts inf in the travel matrix."""
    policy = _policy(0)
    env, _ = _incident_env(closure=True)
    obs, _ = env.reset(seed=0)
    node_embs, graph_emb, _ = enc = policy.encode(env)
    _, _, ea = build_graph(env, torch.device("cpu"))
    assert torch.isfinite(ea).all()  # both edge channels finite under a closure (inf→cap/sentinel)
    assert torch.isfinite(node_embs).all()  # inf→sentinel inside build_graph worked
    assert torch.isfinite(graph_emb).all()
    dist = policy.action_dist(env, obs, enc)
    assert not torch.isnan(dist.probs).any()
    assert torch.allclose(dist.probs.sum(), torch.tensor(1.0), atol=1e-5)
    assert obs["node_features"][1, 8] > 0.0  # node 1 in the closure zone is visible in obs


def test_forward_deterministic_with_congestion():
    def fwd():
        policy = _policy(42)
        env, _ = _incident_env(closure=False)
        obs, _ = env.reset(seed=0)
        return policy.action_dist(env, obs, policy.encode(env)).probs.detach()

    p1, p2 = fwd(), fwd()
    assert torch.allclose(p1, p2)  # weight seed → reproducible forward under congestion
    assert not torch.isnan(p1).any()
