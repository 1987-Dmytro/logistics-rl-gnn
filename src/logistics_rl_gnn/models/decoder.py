"""Attention decoder (autoregressive, Kool-style, Phase 5).

At every step: context = [graph_emb, emb(current), rem_cap/Q, cur_time/horizon, time_context(4)]
→ query (time_context = sin/cos of the hour + sin/cos of the weekday, Phase 6b Step 0);
a multi-head glimpse over the node embeddings (masked) refines the query; the final
compatibility = q·k/√d (WITHOUT C·tanh: saturation zeroed the gradient, Phase 6 diag) → infeasible
masked to −inf → Categorical(logits). K/V are projected ONCE per instance (`precompute`).
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
        # ctx_extra=6: rem_cap/Q, cur_time/horizon + time_context(4). Change the set → change this.
        self.Wq = nn.Linear(2 * d_model + ctx_extra, d_model)  # [graph_emb, emb(cur), 6 scalars]
        self.Wk_g = nn.Linear(d_model, d_model, bias=False)  # glimpse keys
        self.Wv_g = nn.Linear(d_model, d_model, bias=False)  # glimpse values
        self.Wout = nn.Linear(d_model, d_model, bias=False)  # glimpse output projection
        self.Wk_c = nn.Linear(d_model, d_model, bias=False)  # compatibility keys

    def precompute(self, node_embs) -> dict:
        """Node projections (cached per instance). node_embs [N+1, d]."""
        return {"Kg": self.Wk_g(node_embs), "Vg": self.Wv_g(node_embs), "Kc": self.Wk_c(node_embs)}

    def logits(self, context, precomp: dict, mask) -> torch.Tensor:
        """context [ctx_dim], mask [N+1] (1=feasible) → logits [N+1] (infeasible = −inf)."""
        n = precomp["Kc"].shape[0]
        q = self.Wq(context)  # [d]
        # --- multi-head glimpse (masked) ---
        qh = q.view(self.h, self.hd)  # [H, hd]
        Kg = precomp["Kg"].view(n, self.h, self.hd).permute(1, 0, 2)  # [H, N+1, hd]
        Vg = precomp["Vg"].view(n, self.h, self.hd).permute(1, 0, 2)
        scores = torch.einsum("hd,hnd->hn", qh, Kg) / math.sqrt(self.hd)  # [H, N+1]
        scores = scores.masked_fill(mask.unsqueeze(0) == 0, float("-inf"))
        glimpse = torch.einsum("hn,hnd->hd", torch.softmax(scores, -1), Vg).reshape(self.d)
        qc = self.Wout(glimpse)  # [d]
        # compatibility: raw scaled dot-product (WITHOUT C·tanh — tanh saturation zeroed the
        # gradient as logits grew; softmax does not lose it that way, Phase 6 diag). Mask → −inf.
        logits = torch.matmul(precomp["Kc"], qc) / math.sqrt(self.d)  # [N+1]
        return logits.masked_fill(mask == 0, float("-inf"))

    def logits_batch(self, context, precomp: dict, mask) -> torch.Tensor:
        """Batched `logits` over B states of ONE instance (shared precomp, Phase 6b Step 3).

        context [B, ctx_dim], mask [B, N+1] (1=feasible) → logits [B, N+1]. Same maths as the
        single-state `logits`, vectorised over B (one forward for all K rollouts of the search).
        The single-state path (train) is untouched — a parity test checks the batch vs `logits`.
        """
        b, n = context.shape[0], precomp["Kc"].shape[0]
        q = self.Wq(context)  # [B, d]
        qh = q.view(b, self.h, self.hd)  # [B, H, hd]
        Kg = precomp["Kg"].view(n, self.h, self.hd).permute(1, 0, 2)  # [H, N+1, hd] (shared)
        Vg = precomp["Vg"].view(n, self.h, self.hd).permute(1, 0, 2)
        scores = torch.einsum("bhd,hnd->bhn", qh, Kg) / math.sqrt(self.hd)  # [B, H, N+1]
        scores = scores.masked_fill(mask.unsqueeze(1) == 0, float("-inf"))
        glimpse = torch.einsum("bhn,hnd->bhd", torch.softmax(scores, -1), Vg).reshape(b, self.d)
        qc = self.Wout(glimpse)  # [B, d]
        logits = torch.einsum("bd,nd->bn", qc, precomp["Kc"]) / math.sqrt(self.d)  # [B, N+1]
        return logits.masked_fill(mask == 0, float("-inf"))

    def dist(self, context, precomp: dict, mask) -> Categorical:
        return Categorical(logits=self.logits(context, precomp, mask))
