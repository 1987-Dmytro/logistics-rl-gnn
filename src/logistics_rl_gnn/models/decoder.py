"""Attention-декодер (автогрегрессивный, Kool-style, Phase 5).

На каждом шаге: context = [graph_emb, emb(current), rem_cap/Q, cur_time/horizon, time_context(4)]
→ query (time_context = sin/cos часа + sin/cos дня недели, Phase 6b Шаг 0);
многоголовый glimpse по эмбеддингам узлов (маскед) уточняет query; финальная
compatibility = q·k/√d (БЕЗ C·tanh: насыщение зануляло градиент, Phase 6 diag) → маска
инфеасибл в −inf → Categorical(logits). K/V проецируются ОДИН раз на инстанс (`precompute`).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.distributions import Categorical


class AttentionDecoder(nn.Module):
    def __init__(self, d_model: int = 128, heads: int = 8, ctx_extra: int = 6):
        super().__init__()
        assert d_model % heads == 0
        self.d, self.h, self.hd = d_model, heads, d_model // heads
        # ctx_extra=6: rem_cap/Q, cur_time/horizon + time_context(4). Меняешь состав → правь здесь.
        self.Wq = nn.Linear(2 * d_model + ctx_extra, d_model)  # [graph_emb, emb(cur), 6 скаляров]
        self.Wk_g = nn.Linear(d_model, d_model, bias=False)  # glimpse keys
        self.Wv_g = nn.Linear(d_model, d_model, bias=False)  # glimpse values
        self.Wout = nn.Linear(d_model, d_model, bias=False)  # glimpse output-проекция
        self.Wk_c = nn.Linear(d_model, d_model, bias=False)  # compatibility keys

    def precompute(self, node_embs) -> dict:
        """Проекции узлов (кэш на инстанс). node_embs [N+1, d]."""
        return {"Kg": self.Wk_g(node_embs), "Vg": self.Wv_g(node_embs), "Kc": self.Wk_c(node_embs)}

    def logits(self, context, precomp: dict, mask) -> torch.Tensor:
        """context [ctx_dim], mask [N+1] (1=feasible) → logits [N+1] (инфеасибл = −inf)."""
        n = precomp["Kc"].shape[0]
        q = self.Wq(context)  # [d]
        # --- многоголовый glimpse (маскед) ---
        qh = q.view(self.h, self.hd)  # [H, hd]
        Kg = precomp["Kg"].view(n, self.h, self.hd).permute(1, 0, 2)  # [H, N+1, hd]
        Vg = precomp["Vg"].view(n, self.h, self.hd).permute(1, 0, 2)
        scores = torch.einsum("hd,hnd->hn", qh, Kg) / math.sqrt(self.hd)  # [H, N+1]
        scores = scores.masked_fill(mask.unsqueeze(0) == 0, float("-inf"))
        glimpse = torch.einsum("hn,hnd->hd", torch.softmax(scores, -1), Vg).reshape(self.d)
        qc = self.Wout(glimpse)  # [d]
        # compatibility: сырой scaled dot-product (БЕЗ C·tanh — насыщение tanh зануляло градиент
        # при росте логитов; у softmax градиент так не исчезает, Phase 6 diag). Маска → −inf.
        logits = torch.matmul(precomp["Kc"], qc) / math.sqrt(self.d)  # [N+1]
        return logits.masked_fill(mask == 0, float("-inf"))

    def dist(self, context, precomp: dict, mask) -> Categorical:
        return Categorical(logits=self.logits(context, precomp, mask))
