"""Inference search (Phase 6b Step 3): sample-K take-best + PortfolioPlanner. NO training.

PortfolioPlanner collects candidates { sample-K(RL, temperature) ∪ RL-multistart-greedy ∪
greedy heuristic } and takes the best with the SINGLE scorer (`evaluate_solution` under the SAME
travel). Guarantee BY CONSTRUCTION: the result ≤ the greedy heuristic — the greedy candidate is
built by the same `greedy_routes(env=make_dynamic_env(inst, travel, fleet_size))` and scored by
the same scorer as the `greedy` row in the table → `min(candidates) ≤ greedy` identically (#3).
Latency is end-to-end (encode+decode+scoring), median over replicas on fixed hardware.
"""

from __future__ import annotations

import statistics
import time

from logistics_rl_gnn.baselines.greedy import greedy_routes
from logistics_rl_gnn.config import instance as im
from logistics_rl_gnn.env.events import make_dynamic_env
from logistics_rl_gnn.env.scoring import CostConfig, evaluate_solution
from logistics_rl_gnn.replan.local_search import polish
from logistics_rl_gnn.train.pomo import multistart_greedy


def take_best(candidates, instance, travel, cfg: CostConfig, scores_out=None) -> tuple:
    """Best candidate route by the SINGLE scorer. -> (routes, cost€, idx). cost = −reward.
    idx is the index in the ORIGINAL list (None candidates are skipped, numbering is preserved).
    scores_out (optional list) is filled with [(idx, cost€)] — the candidate table for free, out
    of what was already computed (fresh scoring inside the timed block would skew latency)."""
    scored = [
        (i, -evaluate_solution(r, instance, cfg, travel=travel)["reward"])
        for i, r in enumerate(candidates)
        if r is not None
    ]
    assert scored, "no valid candidates (greedy must always be there)"
    if scores_out is not None:
        scores_out.extend(scored)
    i, cost = min(scored, key=lambda ic: ic[1])
    return candidates[i], cost, i


# human-readable names of candidate sources (the demo prints the table with these labels)
SOURCE_RU = {"greedy": "greedy heuristic", "rl_greedy": "RL multistart", "sample": "RL sample-K"}
_RL_SOURCES = ("rl_greedy", "sample")  # candidates produced by the MODEL (ablation --no-model)


def candidate_rows(labels, scores) -> list[dict]:
    """[(idx, cost)] + labels → a per-source table: {source, n, cost(best), mean, polished}.

    polished=None → the source never reached polish (top-M): an honest "—", NOT the raw value.
    """
    agg: dict[str, dict] = {}
    for i, cost in scores:
        base, _, suffix = labels[i].partition("+")
        r = agg.setdefault(base, {"source": base, "n": 0, "cost": None, "sum": 0.0,
                                  "polished": None})
        if suffix:  # «X+polish»
            r["polished"] = cost if r["polished"] is None else min(r["polished"], cost)
        else:
            r["n"] += 1
            r["sum"] += cost
            r["cost"] = cost if r["cost"] is None else min(r["cost"], cost)
    rows = []
    for base in ("greedy", *_RL_SOURCES):
        r = agg.get(base)
        if r is None or r["n"] == 0:
            continue
        rows.append({"source": base, "n": r["n"], "cost": r["cost"],
                     "mean": r["sum"] / r["n"], "polished": r["polished"]})
    return rows


def rl_candidate_mean(rows) -> float | None:
    """Mean raw cost of the MODEL's candidates (weight-swap guard). No RL rows → None."""
    rl = [r for r in rows if r["source"] in _RL_SOURCES]
    n = sum(r["n"] for r in rl)
    return None if n == 0 else sum(r["mean"] * r["n"] for r in rl) / n


class PortfolioPlanner:
    """RL re-plan portfolio: sample-K ∪ RL-multistart-greedy ∪ greedy → best (≤ greedy).

    policy=None → ablation `--no-model`: RL candidates are NOT produced at all (the portfolio is
    greedy (+polish)); an honest counterfactual "what the model adds", not a silent fallback.
    """

    def __init__(
        self,
        policy,
        *,
        k_samples: int = 64,
        temperature: float = 1.0,
        rl_starts: int = 8,
        seed: int = 0,
        polish_budget_ms: float = 0.0,
        polish_top_m: int = 5,
    ):
        self.policy = policy
        self.k_samples = int(k_samples)
        self.temperature = float(temperature)
        self.rl_starts = int(rl_starts)
        self.seed = int(seed)
        self.polish_budget_ms = float(polish_budget_ms)  # 0 → polish disabled (Step 3.5)
        self.polish_top_m = int(polish_top_m)

    def _candidates(self, instance, travel, fleet_size: int):
        """All candidates (seed-deterministic): (greedy, rl-multistart, [K sample rollouts])."""
        # the greedy heuristic is IDENTICAL to the greedy row in the table (the ≤ greedy guarantee)
        gr = greedy_routes(env=make_dynamic_env(instance, travel=travel, fleet_size=fleet_size))
        if self.policy is None:  # --no-model: a portfolio WITHOUT model candidates
            return gr, None, []
        # RL multistart-greedy (POMO distinct-first-starts — the main source of quality)
        env = make_dynamic_env(instance, travel=travel, fleet_size=fleet_size)
        _, rl_routes = multistart_greedy(self.policy, env, self.rl_starts)
        # sample-K (temperature stochasticity) — BATCHED decode (one encode, K rollouts)
        envs = [
            make_dynamic_env(instance, travel=travel, fleet_size=fleet_size)
            for _ in range(self.k_samples)
        ]
        envs[0].reset(seed=0)  # encode on the shared static graph (sample_k re-resets all copies)
        enc = self.policy.encode(envs[0])
        sk = self.policy.sample_k(envs, enc, temperature=self.temperature, seed=self.seed)
        return gr, rl_routes, sk

    def _select(self, instance, travel, fleet_size: int, cfg: CostConfig) -> dict:
        """Candidates → (opt.) polish of top-M in a shared budget → best. Inside plan()'s timer."""
        gr, rl_routes, sk = self._candidates(instance, travel, fleet_size)
        cands = [gr, rl_routes, *sk]  # rl_routes=None when no feasible POMO start exists
        labels = ["greedy", "rl_greedy", *(["sample"] * len(sk))]
        greedy_cost = -evaluate_solution(gr, instance, cfg, travel=travel)["reward"]
        if self.polish_budget_ms > 0:  # Step 3.5: polish the top-M candidates in a SHARED budget
            scored = [
                (i, -evaluate_solution(c, instance, cfg, travel=travel)["reward"])
                for i, c in enumerate(cands)
                if c is not None
            ]
            top = [i for i, _ in sorted(scored, key=lambda ic: ic[1])[: self.polish_top_m]]
            per = self.polish_budget_ms / max(1, len(top))
            _, cap = im.fleet_of(instance)  # scenario Q (else the def-time polish default)
            for i in top:  # the original candidates stay in the pool → the ≤ greedy guarantee holds
                pr, _ = polish(cands[i], instance, travel, budget_ms=per, fleet_size=fleet_size,
                               vehicle_cap=cap)
                cands.append(pr)
                labels.append(labels[i] + "+polish")
        scores: list = []
        best_routes, best_cost, idx = take_best(cands, instance, travel, cfg, scores_out=scores)
        return {
            "routes": best_routes,
            "cost": best_cost,
            "greedy_cost": greedy_cost,  # guarantee: best_cost ≤ greedy_cost (same scorer)
            "source": labels[idx],
            "n_candidates": sum(c is not None for c in cands),
            "rows": candidate_rows(labels, scores),  # free, from what was computed (see take_best)
        }

    def plan(self, instance, travel, *, fleet_size: int, reps: int = 1, warmup: int = 0) -> dict:
        """Portfolio re-plan. -> {routes, cost, greedy_cost, source, n_candidates, latency_ms}."""
        cfg = CostConfig()
        for _ in range(warmup):  # absorb torch lazy-init (honest latency)
            self._select(instance, travel, fleet_size, cfg)
        ts: list[float] = []
        out: dict = {}
        for _ in range(max(1, reps)):  # seed-deterministic → replicas identical; we time them
            t0 = time.perf_counter()
            out = self._select(instance, travel, fleet_size, cfg)
            ts.append((time.perf_counter() - t0) * 1000.0)
        out["latency_ms"] = statistics.median(ts)
        return out
