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
from logistics_rl_gnn.config import instance as im
from logistics_rl_gnn.env.events import make_dynamic_env
from logistics_rl_gnn.env.scoring import CostConfig, evaluate_solution
from logistics_rl_gnn.replan.local_search import polish
from logistics_rl_gnn.train.pomo import multistart_greedy


def take_best(candidates, instance, travel, cfg: CostConfig, scores_out=None) -> tuple:
    """Лучший маршрут-кандидат ЕДИНЫМ scorer'ом. -> (routes, cost€, idx). cost = −reward.
    idx — индекс в ИСХОДНОМ списке (None-кандидаты пропускаем, но нумерацию сохраняем).
    scores_out (опц. список) наполняется [(idx, cost€)] — таблица кандидатов даром, из уже
    посчитанного (новый scoring внутри timed-блока исказил бы латентность)."""
    scored = [
        (i, -evaluate_solution(r, instance, cfg, travel=travel)["reward"])
        for i, r in enumerate(candidates)
        if r is not None
    ]
    assert scored, "нет валидных кандидатов (greedy должен быть всегда)"
    if scores_out is not None:
        scores_out.extend(scored)
    i, cost = min(scored, key=lambda ic: ic[1])
    return candidates[i], cost, i


# человекочитаемые имена источников кандидатов (демо печатает таблицу этими подписями)
SOURCE_RU = {"greedy": "greedy-эвристика", "rl_greedy": "RL multistart", "sample": "RL sample-K"}
_RL_SOURCES = ("rl_greedy", "sample")  # кандидаты, порождённые МОДЕЛЬЮ (ablation --no-model)


def candidate_rows(labels, scores) -> list[dict]:
    """[(idx, cost)] + метки → таблица по источникам: {source, n, cost(лучший), mean, polished}.

    polished=None → источник в polish не попал (top-M): честное «—», НЕ подстановка сырого.
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
    """Средняя сырая стоимость кандидатов МОДЕЛИ (weight-swap-страж). Нет RL-строк → None."""
    rl = [r for r in rows if r["source"] in _RL_SOURCES]
    n = sum(r["n"] for r in rl)
    return None if n == 0 else sum(r["mean"] * r["n"] for r in rl) / n


class PortfolioPlanner:
    """RL-портфель re-plan: sample-K ∪ RL-multistart-greedy ∪ greedy → best (≤ greedy).

    policy=None → ablation `--no-model`: RL-кандидаты НЕ порождаются вовсе (портфель = greedy
    (+polish)); честный контрфактуал «сколько даёт модель», а не тихий фолбэк.
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
        self.polish_budget_ms = float(polish_budget_ms)  # 0 → polish выключен (Шаг 3.5)
        self.polish_top_m = int(polish_top_m)

    def _candidates(self, instance, travel, fleet_size: int):
        """Все кандидаты (детерминированы seed): (greedy, rl-multistart, [K sample-роллаутов])."""
        # greedy-эвристика — ИДЕНТИЧНА методу greedy в таблице (на этом держится гарантия ≤ greedy)
        gr = greedy_routes(env=make_dynamic_env(instance, travel=travel, fleet_size=fleet_size))
        if self.policy is None:  # --no-model: портфель БЕЗ кандидатов модели
            return gr, None, []
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

    def _select(self, instance, travel, fleet_size: int, cfg: CostConfig) -> dict:
        """Кандидаты → (опц.) polish топ-M в общем бюджете → best. Внутри timed-блока plan()."""
        gr, rl_routes, sk = self._candidates(instance, travel, fleet_size)
        cands = [gr, rl_routes, *sk]  # rl_routes=None при отсутствии feasible POMO-старта
        labels = ["greedy", "rl_greedy", *(["sample"] * len(sk))]
        greedy_cost = -evaluate_solution(gr, instance, cfg, travel=travel)["reward"]
        if self.polish_budget_ms > 0:  # Шаг 3.5: полируем топ-M кандидатов в ОБЩЕМ бюджете
            scored = [
                (i, -evaluate_solution(c, instance, cfg, travel=travel)["reward"])
                for i, c in enumerate(cands)
                if c is not None
            ]
            top = [i for i, _ in sorted(scored, key=lambda ic: ic[1])[: self.polish_top_m]]
            per = self.polish_budget_ms / max(1, len(top))
            _, cap = im.fleet_of(instance)  # Q сценария (иначе def-time дефолт polish)
            for i in top:  # исходные кандидаты остаются в пуле → гарантия ≤ greedy цела
                pr, _ = polish(cands[i], instance, travel, budget_ms=per, fleet_size=fleet_size,
                               vehicle_cap=cap)
                cands.append(pr)
                labels.append(labels[i] + "+polish")
        scores: list = []
        best_routes, best_cost, idx = take_best(cands, instance, travel, cfg, scores_out=scores)
        return {
            "routes": best_routes,
            "cost": best_cost,
            "greedy_cost": greedy_cost,  # гарантия: best_cost ≤ greedy_cost (тот же scorer)
            "source": labels[idx],
            "n_candidates": sum(c is not None for c in cands),
            "rows": candidate_rows(labels, scores),  # даром из уже посчитанного (см. take_best)
        }

    def plan(self, instance, travel, *, fleet_size: int, reps: int = 1, warmup: int = 0) -> dict:
        """Re-plan портфелем. -> {routes, cost, greedy_cost, source, n_candidates, latency_ms}."""
        cfg = CostConfig()
        for _ in range(warmup):  # torch lazy-init гасим (честная латентность)
            self._select(instance, travel, fleet_size, cfg)
        ts: list[float] = []
        out: dict = {}
        for _ in range(max(1, reps)):  # детерминизм по seed → реплики идентичны; меряем время
            t0 = time.perf_counter()
            out = self._select(instance, travel, fleet_size, cfg)
            ts.append((time.perf_counter() - t0) * 1000.0)
        out["latency_ms"] = statistics.median(ts)
        return out
