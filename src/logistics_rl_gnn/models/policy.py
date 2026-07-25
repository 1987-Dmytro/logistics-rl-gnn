"""VRPPolicy — wires encoder+decoder to VRPEnv (Phase 5). NO train loop (that is Phase 6):
forward + rollout only. Feasibility comes from env.action_mask (single source of truth).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from logistics_rl_gnn.env.scoring import evaluate_solution
from logistics_rl_gnn.env.travel import time_context
from logistics_rl_gnn.models.decoder import AttentionDecoder
from logistics_rl_gnn.models.encoder import GATEncoder

_MULT_CAP = 10.0  # cap on the congestion multiplier (a closure → "≈blocked", not inf)


def build_graph(env, device):
    """Static normalised instance graph from env data (single source).

    Nodes (8): [x, y, demand/Q, e/H, l/H, service/T_max, is_depot, node_congestion].
    Edges (edge_attr [E,2]): [0] travel_time(i,j,cur_time) of the ACTIVE model, per-instance
    max-normalised (topology; under free-flow == the previous value); [1] congestion_multiplier =
    travel/free_flow = c·(1+ΣI) — ≡1 under free-flow, yet it makes the diurnal/incident visible
    PER EDGE (closes the 0005 subtlety: max-norm erased a uniform diurnal in channel 0).
    -> (node_feat [k,8], edge_index [2,E], edge_attr [E,2]) on device.
    """
    k = env.k
    coord = torch.as_tensor(env.coords, dtype=torch.float32)  # [k, 2] (lon, lat)
    cmin, cmax = coord.amin(0), coord.amax(0)
    coord = (coord - cmin) / (cmax - cmin + 1e-8)  # per-instance min-max → [0,1]
    horizon = env._inst.horizon_s / 60.0  # minutes
    win = torch.as_tensor(env.win, dtype=torch.float32)  # [k, 2] minutes
    nc = torch.as_tensor(  # congestion level at the node (0 under free-flow → parity)
        env.travel.node_congestion(env.coords, env.cur_time), dtype=torch.float32
    )
    node_feat = torch.stack(
        [
            coord[:, 0],
            coord[:, 1],
            torch.as_tensor(env.demand, dtype=torch.float32) / env.Q,
            win[:, 0] / horizon,
            win[:, 1] / horizon,
            torch.as_tensor(env.service_min, dtype=torch.float32) / env.t_max_min,
            torch.tensor([1.0] + [0.0] * (k - 1)),  # is_depot (node 0 = depot)
            nc,
        ],
        dim=1,
    )
    # congestion-aware travel matrix (snapshot at cur_time); under FreeFlow == env.time_m (parity).
    # travel.matrix() is vectorised (k² time() calls choked POMO retrain at k=62); parity tested.
    ff = torch.as_tensor(env.time_m, dtype=torch.float32)  # free-flow t0 [k,k] minutes
    tm = torch.as_tensor(env.travel.matrix(env.cur_time), dtype=torch.float32)
    # channel 1: multiplier travel/ff (≡1 under free-flow). diag ff=0 → 1; closure (inf) → cap.
    mult = torch.where(ff > 0, tm / ff.clamp_min(1e-8), torch.ones_like(ff))
    mult = torch.where(torch.isinf(mult), torch.full_like(mult, _MULT_CAP), mult)
    mult = mult.clamp_max(_MULT_CAP)
    if torch.isinf(tm).any():  # channel 0: closure → a large finite "very slow" (no NaN)
        tm = torch.where(torch.isinf(tm), tm[torch.isfinite(tm)].max() * 3.0, tm)
    idx = torch.arange(k)
    edge_index = torch.stack([idx.repeat_interleave(k), idx.repeat(k)])  # complete graph [2, k*k]
    ea0 = tm.reshape(-1, 1) / (tm.max() + 1e-8)  # travel_time normalised (topology)
    edge_attr = torch.cat([ea0, mult.reshape(-1, 1)], dim=1)  # [k*k, 2]
    return node_feat.to(device), edge_index.to(device), edge_attr.to(device)


class VRPPolicy(nn.Module):
    def __init__(self, d_model: int = 128, heads: int = 8, n_layers: int = 3):
        super().__init__()
        self.encoder = GATEncoder(in_dim=8, d_model=d_model, heads=heads, n_layers=n_layers)
        self.decoder = AttentionDecoder(d_model=d_model, heads=heads)

    @property
    def device(self):
        return next(self.parameters()).device

    def encode(self, env):
        """One encoder pass per instance → (node_embs, graph_emb, decoder precomp)."""
        node_feat, ei, ea = build_graph(env, self.device)
        node_embs, graph_emb = self.encoder(node_feat, ei, ea)
        return node_embs, graph_emb, self.decoder.precompute(node_embs)

    def _context(self, env, enc) -> torch.Tensor:
        """Context vector of π(·|s): [graph_emb, emb(pos), dyn(2), tctx(4)].
        Shared by the single (train) and batched (sample_k) paths — one layout, no duplicates."""
        node_embs, graph_emb, _ = enc
        horizon = env._inst.horizon_s / 60.0
        dyn = torch.tensor(
            [env.rem_cap / env.Q, env.cur_time / horizon], dtype=torch.float32, device=self.device
        )
        tctx = torch.as_tensor(  # time-context (congestion phase): under free-flow a constant input
            time_context(env.abs_minute, env.dow), dtype=torch.float32, device=self.device
        )
        return torch.cat([graph_emb, node_embs[env.pos], dyn, tctx])

    def action_dist(self, env, obs, enc) -> torch.distributions.Categorical:
        """Distribution π(a|s) in the current env state. enc = (node_embs, graph_emb, precomp)."""
        mask = torch.as_tensor(obs["action_mask"], dtype=torch.float32, device=self.device)
        return self.decoder.dist(self._context(env, enc), enc[2], mask)

    def sample_k(self, envs, enc, *, temperature: float = 1.0, seed: int = 0) -> list:
        """K stochastic rollouts from a SHARED enc, decoding BATCHED over K (one forward per step).

        envs — K fresh copies of ONE (fixed) instance → shared static graph → shared enc; they are
        reset here (divergence comes from sampling only). No forced starts — pure temperature
        sampling from step 0 (POMO multistart-greedy is a separate portfolio candidate).
        Only the env copies are mutated; seed → determinism (own generator, global RNG untouched).
        -> list[routes] of length K.
        """
        assert temperature > 0, "temperature > 0 (greedy is a separate strategy)"
        dev = self.device
        gen = torch.Generator(device=dev).manual_seed(int(seed))
        obs = [e.reset(seed=0)[0] for e in envs]  # fixed instance (make_dynamic_env ignores seed)
        routes: list = [None] * len(envs)
        active = list(range(len(envs)))
        precomp = enc[2]
        with torch.no_grad():
            while active:
                ctx = torch.stack([self._context(envs[i], enc) for i in active])  # [A, ctx]
                mask = torch.stack(
                    [
                        torch.as_tensor(obs[i]["action_mask"], dtype=torch.float32, device=dev)
                        for i in active
                    ]
                )  # [A, N+1]
                logits = self.decoder.logits_batch(ctx, precomp, mask)
                probs = torch.softmax(logits / temperature, -1)
                acts = torch.multinomial(probs, 1, generator=gen).squeeze(1)  # [A]
                still = []
                for slot, a in zip(active, acts.tolist(), strict=False):
                    obs[slot], _, term, trunc, info = envs[slot].step(int(a))
                    if term or trunc:
                        routes[slot] = info["routes"]
                    else:
                        still.append(slot)
                active = still
        return routes

    def rollout(self, env, mode: str = "sample", seed=None, return_entropy: bool = False):
        """Autoregressive episode. -> (routes, sum_logπ [grad], metrics[, mean_entropy]).

        return_entropy=True appends the step-mean entropy of π as a 4th element — a TENSOR with
        gradient (collapse guard + optional entropy bonus in the loss).
        """
        obs, info = env.reset(seed=seed)
        enc = self.encode(env)
        logps, ents = [], []
        done = False
        while not done:
            dist = self.action_dist(env, obs, enc)
            a = dist.probs.argmax() if mode == "greedy" else dist.sample()
            logps.append(dist.log_prob(a))
            if return_entropy:  # manual entropy (masked p=0 → 0·log ≈ 0, no NaN from −inf logits)
                p = dist.probs
                ents.append(-(p * (p + 1e-12).log()).sum())
            obs, _, term, trunc, info = env.step(int(a.item()))
            done = term or trunc
        sum_logp = torch.stack(logps).sum() if logps else torch.zeros((), device=self.device)
        metrics = evaluate_solution(info["routes"], env._inst, env._cost_cfg)  # env cfg
        if return_entropy:
            mean_ent = torch.stack(ents).mean() if ents else torch.zeros((), device=self.device)
            return info["routes"], sum_logp, metrics, mean_ent
        return info["routes"], sum_logp, metrics
