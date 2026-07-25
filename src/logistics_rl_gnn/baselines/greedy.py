"""Greedy nearest-feasible — the honest baseline (prohibition #3).

Formalises the demonstrator: run VRPEnv with the greedy policy "nearest feasible stop by travel
time" → cap/TW/T_max masking comes FROM the environment (one source of feasibility, no
divergence). No feasible stop → close the route (action 0) and the env starts the next vehicle.
Deterministic: the instance is seeded, argmin ties break on the index.
"""

from __future__ import annotations

from logistics_rl_gnn.env.vrp_env import VRPEnv


def greedy_routes(seed: int = 0, *, env: VRPEnv | None = None, **env_kw) -> list[list[int]]:
    """Nearest-feasible routes on instance `seed`. Returns routes (env format)."""
    env = env or VRPEnv(**env_kw)
    obs, _ = env.reset(seed=seed)
    info: dict = {}
    done = False
    while not done:
        mask = obs["action_mask"]
        cands = [j for j in range(1, env.k) if mask[j]]  # feasible pharmacies (0 = depot)
        # nearest by travel time, tie → lower index; no feasible stop → 0 (depot/next vehicle)
        a = min(cands, key=lambda j: (env.time_m[env.pos, j], j)) if cands else 0
        obs, _, term, trunc, info = env.step(a)
        done = term or trunc
    return info["routes"]
