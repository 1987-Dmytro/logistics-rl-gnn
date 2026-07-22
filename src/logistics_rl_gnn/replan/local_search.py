"""Local-search polish декодированных маршрутов (Phase 6b Шаг 3.5). БЕЗ обучения.

`polish` улучшает маршруты классическим local search:
  • intra-route: 2-opt (реверс внутр. сегмента) + Or-opt (перенос сегмента 1–3 внутри маршрута);
  • inter-route: relocate (клиент A→B) + swap (клиент A ↔ клиент B).
Стоимость — ТОЛЬКО `evaluate_solution` (полная переоценка кандидата → корректно под time-dependent
congestion, БЕЗ delta-cost допущений: реверс/перенос сдвигает все downstream-времена). Feasibility —
`check_feasible` (жёстко как env: cap/TW/T_max/fleet). First-improvement, циклы до сходимости или
`budget_ms`. **ИНВАРИАНТ: результат НЕ хуже входа** (best стартует входом, заменяется лишь строго
лучшим феасибл-кандидатом). Операторы перераспределяют по СУЩЕСТВУЮЩИМ слотам — маршрут не
добавляют (vehicles_used не растёт сверх fleet; пустые [0,0]-слоты дают relocate место под машину).
"""

from __future__ import annotations

import time
from itertools import chain

from logistics_rl_gnn.config import instance as im
from logistics_rl_gnn.env.scoring import CostConfig, check_feasible, evaluate_solution


def _cost(routes, instance, cfg, travel) -> float:
    return -evaluate_solution(routes, instance, cfg, travel=travel)["reward"]


def _normalize(routes, fleet_size):
    """Каждый маршрут → минимум [0,0]; добить пустыми слотами до fleet_size (место для relocate)."""
    norm = [list(r) if len(r) >= 2 else [0, 0] for r in routes]
    if fleet_size is not None:
        while len(norm) < fleet_size:
            norm.append([0, 0])
    return norm


def _two_opt(routes):
    """Реверс внутреннего сегмента одного маршрута (depot-концы не трогаем)."""
    for ri in range(len(routes)):
        r = routes[ri]
        for i in range(1, len(r) - 2):
            for j in range(i + 1, len(r) - 1):
                cand = [x[:] for x in routes]
                cand[ri][i : j + 1] = r[i : j + 1][::-1]
                yield cand


def _or_opt(routes):
    """Перенос сегмента длины 1–3 в другую позицию ТОГО ЖЕ маршрута (identity → отсеет strict<)."""
    for ri in range(len(routes)):
        r = routes[ri]
        for seg_len in (1, 2, 3):
            for p in range(1, len(r) - seg_len):  # сегмент [p, p+seg_len) — внутри интерьера
                seg = r[p : p + seg_len]
                rest = r[:p] + r[p + seg_len :]
                for q in range(1, len(rest)):  # вставка между depot-концами rest
                    cand = [x[:] for x in routes]
                    cand[ri] = rest[:q] + seg + rest[q:]
                    yield cand


def _relocate(routes):
    """Перенос клиента из маршрута A в маршрут B (B может быть пустым [0,0] → новая машина)."""
    for ai in range(len(routes)):
        ra = routes[ai]
        for p in range(1, len(ra) - 1):
            cust = ra[p]
            src = ra[:p] + ra[p + 1 :]
            for bi in range(len(routes)):
                if bi == ai:
                    continue
                rb = routes[bi]
                for q in range(1, len(rb)):
                    cand = [x[:] for x in routes]
                    cand[ai] = src
                    cand[bi] = rb[:q] + [cust] + rb[q:]
                    yield cand


def _swap(routes):
    """Обмен клиента маршрута A с клиентом маршрута B (пары ai<bi — без дублей)."""
    for ai in range(len(routes)):
        ra = routes[ai]
        for pa in range(1, len(ra) - 1):
            for bi in range(ai + 1, len(routes)):
                rb = routes[bi]
                for pb in range(1, len(rb) - 1):
                    cand = [x[:] for x in routes]
                    cand[ai] = ra[:pa] + [rb[pb]] + ra[pa + 1 :]
                    cand[bi] = rb[:pb] + [ra[pa]] + rb[pb + 1 :]
                    yield cand


def _moves(routes):
    """Кандидаты в ФИКСИРОВАННОМ порядке (детерминизм): 2-opt → Or-opt → relocate → swap."""
    return chain(_two_opt(routes), _or_opt(routes), _relocate(routes), _swap(routes))


def polish(
    routes,
    instance,
    travel=None,
    *,
    budget_ms: float = 200.0,
    t_max_min: float = im.T_MAX_MIN,
    vehicle_cap: float = im.VEHICLE_CAP,
    fleet_size: int | None = None,
    cfg: CostConfig | None = None,
) -> tuple[list, float]:
    """Полировка маршрутов local-search'ем. -> (routes, cost€). НЕ хуже входа; feasibility как env.

    budget_ms — потолок wall-clock (для сходимости в замерах ставь щедрый). fleet_size=None →
    выводим из входа (число слотов). Стоимость/feasibility — единый scorer/check_feasible.
    """
    cfg = cfg or CostConfig()
    best = _normalize(routes, fleet_size)
    fleet = fleet_size if fleet_size is not None else len(best)

    def feas(rs):
        return check_feasible(
            rs, instance, travel, t_max_min=t_max_min, vehicle_cap=vehicle_cap, fleet_size=fleet
        )

    best_cost = _cost(best, instance, cfg, travel)
    if not feas(best):  # вход инфеасибл → возвращаем как есть (инвариант: НЕ хуже входа)
        return best, best_cost
    deadline = time.perf_counter() + budget_ms / 1000.0
    improved = True
    while improved and time.perf_counter() < deadline:
        improved = False
        for cand in _moves(best):
            if time.perf_counter() >= deadline:
                break
            c = _cost(cand, instance, cfg, travel)
            if c < best_cost - 1e-9 and feas(cand):  # строго лучше И феасибл → first-improvement
                best, best_cost, improved = cand, c, True
                break  # рестарт enumerate с нового best
    return best, best_cost
