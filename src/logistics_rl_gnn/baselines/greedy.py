"""Greedy nearest-feasible — честный бейзлайн (запрет №3).

Формализация демонстратора: гоняем VRPEnv жадной политикой «ближайший feasible-стоп по
времени в пути» → маскинг cap/TW/T_max берём ИЗ среды (единый источник feasibility, без
расхождений). Нет feasible-стопа → закрыть маршрут (действие 0), среда стартует след. машину.
Детерминирован: инстанс сидируется, argmin с tie-break по индексу.
"""

from __future__ import annotations

from logistics_rl_gnn.env.vrp_env import VRPEnv


def greedy_routes(seed: int = 0, *, env: VRPEnv | None = None, **env_kw) -> list[list[int]]:
    """Маршруты nearest-feasible на инстансе `seed`. Возвращает routes (env-формат)."""
    env = env or VRPEnv(**env_kw)
    obs, _ = env.reset(seed=seed)
    info: dict = {}
    done = False
    while not done:
        mask = obs["action_mask"]
        cands = [j for j in range(1, env.k) if mask[j]]  # feasible аптеки (0 = депо)
        # ближайшая по времени в пути, tie → меньший индекс; нет feasible → 0 (депо/след. машина)
        a = min(cands, key=lambda j: (env.time_m[env.pos, j], j)) if cands else 0
        obs, _, term, trunc, info = env.step(a)
        done = term or trunc
    return info["routes"]
