"""Инференс-поиск (Phase 6b Шаг 3): sample-K take-best + PortfolioPlanner. БЕЗ обучения.

PortfolioPlanner собирает кандидаты { sample-K(RL, temperature) ∪ RL-multistart-greedy ∪
greedy-эвристика } и берёт лучший ЕДИНЫМ scorer'ом (`evaluate_solution` под ТЕМ ЖЕ travel).
Гарантия ПО ПОСТРОЕНИЮ: результат ≤ greedy-эвристика — greedy-кандидат строится тем же
`greedy_routes(env=make_dynamic_env(inst, travel, fleet_size))` и скорится тем же scorer'ом,
что метод `greedy` в таблице → `min(кандидаты) ≤ greedy` тождественно (запрет №3 цел).
Латентность end-to-end (encode+decode+scoring), медиана реплик на фикс. железе.
"""

from __future__ import annotations

import statistics
import time

from logistics_rl_gnn.baselines.greedy import greedy_routes
from logistics_rl_gnn.env.events import make_dynamic_env
from logistics_rl_gnn.env.scoring import CostConfig, evaluate_solution
from logistics_rl_gnn.train.pomo import multistart_greedy


def take_best(candidates, instance, travel, cfg: CostConfig) -> tuple[list, float, int]:
    """Лучший маршрут-кандидат ЕДИНЫМ scorer'ом. -> (routes, cost€, idx). cost = −reward.
    idx — индекс в ИСХОДНОМ списке (None-кандидаты пропускаем, но нумерацию сохраняем)."""
    scored = [
        (i, -evaluate_solution(r, instance, cfg, travel=travel)["reward"])
        for i, r in enumerate(candidates)
        if r is not None
    ]
    assert scored, "нет валидных кандидатов (greedy должен быть всегда)"
    i, cost = min(scored, key=lambda ic: ic[1])
    return candidates[i], cost, i


class PortfolioPlanner:
    """RL-портфель re-plan: sample-K ∪ RL-multistart-greedy ∪ greedy → best (≤ greedy)."""

    def __init__(
        self,
        policy,
        *,
        k_samples: int = 64,
        temperature: float = 1.0,
        rl_starts: int = 8,
        seed: int = 0,
    ):
        self.policy = policy
        self.k_samples = int(k_samples)
        self.temperature = float(temperature)
        self.rl_starts = int(rl_starts)
        self.seed = int(seed)

    def _candidates(self, instance, travel, fleet_size: int):
        """Все кандидаты (детерминированы seed): (greedy, rl-multistart, [K sample-роллаутов])."""
        # greedy-эвристика — ИДЕНТИЧНА методу greedy в таблице (на этом держится гарантия ≤ greedy)
        gr = greedy_routes(env=make_dynamic_env(instance, travel=travel, fleet_size=fleet_size))
        # RL multistart-greedy (POMO distinct-first-starts — основной источник качества)
        env = make_dynamic_env(instance, travel=travel, fleet_size=fleet_size)
        _, rl_routes = multistart_greedy(self.policy, env, self.rl_starts)
        # sample-K (temperature-стохастика) — БАТЧЕВЫЙ decode (один encode, K роллаутов)
        envs = [
            make_dynamic_env(instance, travel=travel, fleet_size=fleet_size)
            for _ in range(self.k_samples)
        ]
        envs[0].reset(seed=0)  # encode на общем статическом графе (sample_k пересбросит все копии)
        enc = self.policy.encode(envs[0])
        sk = self.policy.sample_k(envs, enc, temperature=self.temperature, seed=self.seed)
        return gr, rl_routes, sk

    def plan(self, instance, travel, *, fleet_size: int, reps: int = 1, warmup: int = 0) -> dict:
        """Re-plan портфелем. -> {routes, cost, greedy_cost, source, n_candidates, latency_ms}."""
        cfg = CostConfig()
        for _ in range(warmup):  # torch lazy-init гасим (честная латентность)
            self._candidates(instance, travel, fleet_size)
        ts: list[float] = []
        gr = rl_routes = sk = None
        for _ in range(max(1, reps)):  # детерминизм по seed → реплики идентичны; меряем время
            t0 = time.perf_counter()
            gr, rl_routes, sk = self._candidates(instance, travel, fleet_size)
            ts.append((time.perf_counter() - t0) * 1000.0)
        cands = [gr, rl_routes, *sk]  # rl_routes=None при отсутствии feasible POMO-старта
        labels = ["greedy", "rl_greedy", *(["sample"] * len(sk))]
        best_routes, best_cost, idx = take_best(cands, instance, travel, cfg)
        greedy_cost = -evaluate_solution(gr, instance, cfg, travel=travel)["reward"]
        return {
            "routes": best_routes,
            "cost": best_cost,
            "greedy_cost": greedy_cost,  # гарантия: best_cost ≤ greedy_cost (тот же scorer)
            "source": labels[idx],
            "n_candidates": sum(c is not None for c in cands),
            "latency_ms": statistics.median(ts),
        }
