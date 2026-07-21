"""VRPPolicy — связка энкодер+декодер с VRPEnv (Phase 5). БЕЗ train-loop (это Phase 6):
только forward + rollout. Феасибилити — из env.action_mask (single source of truth).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from logistics_rl_gnn.env.scoring import evaluate_solution
from logistics_rl_gnn.env.travel import time_context
from logistics_rl_gnn.models.decoder import AttentionDecoder
from logistics_rl_gnn.models.encoder import GATEncoder


def build_graph(env, device):
    """Статический нормализованный граф инстанса из env-данных (single source).

    Узлы (8): [x, y, demand/Q, e/H, l/H, service/T_max, is_depot, node_congestion].
    Рёбра: полный граф; edge_attr = travel_time(i,j, cur_time) АКТИВНОЙ модели (норм.) — под
    free-flow == прежнее free-flow-время (паритет). -> (node_feat [k,8], edge_index [2,E],
    edge_attr [E,1]) на device.
    """
    k = env.k
    coord = torch.as_tensor(env.coords, dtype=torch.float32)  # [k, 2] (lon, lat)
    cmin, cmax = coord.amin(0), coord.amax(0)
    coord = (coord - cmin) / (cmax - cmin + 1e-8)  # per-instance min-max → [0,1]
    horizon = env._inst.horizon_s / 60.0  # мин
    win = torch.as_tensor(env.win, dtype=torch.float32)  # [k, 2] мин
    nc = torch.as_tensor(  # уровень congestion у узла (0 под free-flow → паритет)
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
            torch.tensor([1.0] + [0.0] * (k - 1)),  # is_depot (узел 0 = депо)
            nc,
        ],
        dim=1,
    )
    # congestion-aware travel-матрица (снимок в cur_time); под FreeFlow == env.time_m (паритет).
    # ponytail: k² вызовов travel.time на encode; при POMO-ретрейне (k=62) — векторизовать матрицей.
    tm = torch.tensor(
        [[env.travel.time(i, j, env.cur_time) for j in range(k)] for i in range(k)],
        dtype=torch.float32,
    )
    if torch.isinf(tm).any():  # закрытие → крупный конечный «очень медленно» (без NaN в норме)
        tm = torch.where(torch.isinf(tm), tm[torch.isfinite(tm)].max() * 3.0, tm)
    idx = torch.arange(k)
    edge_index = torch.stack([idx.repeat_interleave(k), idx.repeat(k)])  # полный граф [2, k*k]
    edge_attr = tm.reshape(-1, 1) / (tm.max() + 1e-8)  # travel_time норм. [k*k, 1]
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
        """Один прогон энкодера на инстанс → (node_embs, graph_emb, precomp декодера)."""
        node_feat, ei, ea = build_graph(env, self.device)
        node_embs, graph_emb = self.encoder(node_feat, ei, ea)
        return node_embs, graph_emb, self.decoder.precompute(node_embs)

    def action_dist(self, env, obs, enc) -> torch.distributions.Categorical:
        """Распределение π(a|s) в текущем состоянии env. enc = (node_embs, graph_emb, precomp)."""
        node_embs, graph_emb, precomp = enc
        horizon = env._inst.horizon_s / 60.0
        mask = torch.as_tensor(obs["action_mask"], dtype=torch.float32, device=self.device)
        dyn = torch.tensor(
            [env.rem_cap / env.Q, env.cur_time / horizon], dtype=torch.float32, device=self.device
        )
        tctx = torch.as_tensor(  # time-context (congestion-фаза): под free-flow — постоянный вход
            time_context(env.abs_minute, env.dow), dtype=torch.float32, device=self.device
        )
        context = torch.cat([graph_emb, node_embs[env.pos], dyn, tctx])
        return self.decoder.dist(context, precomp, mask)

    def rollout(self, env, mode: str = "sample", seed=None, return_entropy: bool = False):
        """Автогрегрессивный эпизод. -> (routes, sum_logπ [grad], metrics[, mean_entropy]).

        return_entropy=True добавляет 4-м элементом среднюю по шагам энтропию π — ТЕНЗОР с
        градиентом (страж коллапса + опц. entropy-бонус в loss).
        """
        obs, info = env.reset(seed=seed)
        enc = self.encode(env)
        logps, ents = [], []
        done = False
        while not done:
            dist = self.action_dist(env, obs, enc)
            a = dist.probs.argmax() if mode == "greedy" else dist.sample()
            logps.append(dist.log_prob(a))
            if return_entropy:  # ручная энтропия (маскед p=0 → 0·log ≈ 0, без NaN от −inf логитов)
                p = dist.probs
                ents.append(-(p * (p + 1e-12).log()).sum())
            obs, _, term, trunc, info = env.step(int(a.item()))
            done = term or trunc
        sum_logp = torch.stack(logps).sum() if logps else torch.zeros((), device=self.device)
        metrics = evaluate_solution(info["routes"], env._inst, env._cost_cfg)  # cfg среды
        if return_entropy:
            mean_ent = torch.stack(ents).mean() if ents else torch.zeros((), device=self.device)
            return info["routes"], sum_logp, metrics, mean_ent
        return info["routes"], sum_logp, metrics
