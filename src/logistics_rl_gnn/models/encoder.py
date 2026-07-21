"""GAT-энкодер статического графа инстанса (Phase 5). Kool-style: кодируем ОДИН раз на
инстанс, декодер потом ходит по узлам автогрегрессивно.

Вход — нормализованные признаки узла [x, y, demand/Q, e/horizon, l/horizon, service/T_max,
is_depot, node_congestion] (8) и рёбра полного графа с edge_attr (2): [0] travel_time АКТИВНОЙ
модели (норм.), [1] congestion_multiplier travel/free_flow (≡1 под free-flow). Выход —
эмбеддинги узлов [N+1, d_model] + graph_emb (mean-pool). Device-agnostic (обычный nn.Module).
"""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class GATEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int = 8,
        d_model: int = 128,
        heads: int = 8,
        n_layers: int = 3,
        edge_dim: int = 2,
    ):
        super().__init__()
        assert d_model % heads == 0, "d_model должен делиться на heads"
        self.in_proj = nn.Linear(in_dim, d_model)
        # concat=True + out=d_model//heads → выход ровно d_model; self-loops не добавляем
        # (полный граф уже содержит диагональ (i,i)). edge_dim=2: travel-норма + congestion-множ.
        self.convs = nn.ModuleList(
            GATv2Conv(
                d_model,
                d_model // heads,
                heads=heads,
                concat=True,
                edge_dim=edge_dim,
                add_self_loops=False,
                residual=False,
            )
            for _ in range(n_layers)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(d_model) for _ in range(n_layers))

    def forward(self, x, edge_index, edge_attr):
        """x [N+1,in_dim], edge_index [2,E], edge_attr [E,1] → (node_embs [N+1,d], graph_emb [d])"""
        h = self.in_proj(x)
        for conv, norm in zip(self.convs, self.norms, strict=True):
            h = norm(h + F.elu(conv(h, edge_index, edge_attr)))  # residual + LN
        return h, h.mean(dim=0)
