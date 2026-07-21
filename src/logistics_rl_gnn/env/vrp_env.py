"""VRPEnv — gymnasium-среда динамического CVRPTW (Phase 3 core).

Инстанс из config.instance.generate_instance (реальные окна аптек). Машины строятся
ПОСЛЕДОВАТЕЛЬНО; каждая стартует и финиширует в депо (cur_time сбрасывается в 0),
ограничена вместимостью Q и временем тура T_max. Действие = следующий узел
(0 = депо / закрыть маршрут). Feasibility через action_mask. Reward разрежённый
терминальный (dense-shaping — опция). ЕДИНИЦА ВРЕМЕНИ ВНУТРИ СРЕДЫ — МИНУТЫ.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from logistics_rl_gnn.config import instance as cfg
from logistics_rl_gnn.env.scoring import CostConfig, evaluate_solution
from logistics_rl_gnn.env.travel import FreeFlowTravel, time_context

# признаков на узел: x, y, demand, e, l, service, visited, feasible, congestion
_NF = 9  # congestion = уровень пробки/инцидента у узла (0 под free-flow, Phase 6b Шаг 0)


class VRPEnv(gym.Env):
    """CVRPTW-среда. Действие Discrete(k), k = депо + аптеки инстанса."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        instance_fn=None,
        *,
        delivery_weekday: int = cfg.DELIVERY_WEEKDAY,
        fleet_size: int = cfg.FLEET_SIZE,
        vehicle_cap: float = cfg.VEHICLE_CAP,
        t_max_min: float = cfg.T_MAX_MIN,
        cost_vehicle: float = cfg.COST_PER_VEHICLE,
        cost_km: float = cfg.COST_PER_KM,
        cost_hour: float = cfg.COST_PER_HOUR,
        penalty_late: float = cfg.PENALTY_LATE_PER_MIN,
        penalty_unserved: float = cfg.PENALTY_UNSERVED,
        penalty_invalid: float = cfg.PENALTY_INVALID,
        dense_reward: bool = cfg.DENSE_REWARD,
        max_steps: int | None = None,
    ):
        super().__init__()
        self._instance_fn = instance_fn or (
            lambda seed: cfg.generate_instance(seed=seed, delivery_weekday=delivery_weekday)
        )
        self.K = int(fleet_size)
        self.dow = int(delivery_weekday)  # для time-context (congestion-фаза дня недели)
        self.Q = float(vehicle_cap)
        self.t_max_min = float(t_max_min)
        self.c_f, self.c_d, self.c_t = float(cost_vehicle), float(cost_km), float(cost_hour)
        self.p_late = float(penalty_late)
        self.p_unserved = float(penalty_unserved)
        self.p_invalid = float(penalty_invalid)
        self.dense = bool(dense_reward)
        self._cost_cfg = CostConfig(self.c_f, self.c_d, self.c_t, self.p_late, self.p_unserved)

        probe = self._instance_fn(0)
        self.k = len(probe.node_ids)  # стопов; фиксирован для weekday (seed не меняет исключения)
        self.max_steps = int(max_steps) if max_steps else 2 * self.k + self.K + 2

        self.action_space = spaces.Discrete(self.k)
        self.observation_space = spaces.Dict(
            {
                "node_features": spaces.Box(-1e6, 1e6, shape=(self.k, _NF), dtype=np.float32),
                "vehicle_state": spaces.Box(-1e6, 1e6, shape=(4,), dtype=np.float32),
                # time-context: [sin_h, cos_h, sin_dow, cos_dow] — фаза congestion (Phase 6b Шаг 0)
                "time_context": spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32),
                "action_mask": spaces.MultiBinary(self.k),
            }
        )
        self._load(probe)

    # ---------- инстанс ----------

    def _load(self, inst) -> None:
        self._inst = inst  # для единого оценщика на терминале
        self.time_m = np.asarray(inst.time_matrix, dtype=float) / 60.0  # сек→мин
        self.dist_km = np.asarray(inst.dist_matrix, dtype=float) / 1000.0  # м→км
        self.win = np.asarray(inst.windows, dtype=float) / 60.0  # сек→мин
        self.service_min = np.asarray(inst.service, dtype=float) / 60.0
        self.demand = np.asarray(inst.demand, dtype=float)
        self.coords = np.asarray(inst.coords, dtype=float)
        self.travel = FreeFlowTravel(self.time_m)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._load(self._instance_fn(0 if seed is None else int(seed)))
        self.pos = 0
        self.cur_time = 0.0
        self.rem_cap = self.Q
        self.vehicle_idx = 0
        self.visited = np.zeros(self.k, dtype=bool)
        self.visited[0] = True  # депо — не клиент
        self.steps = 0
        self.vehicles_used = 0
        self.total_km = 0.0
        self.total_time_min = 0.0
        self.total_late_min = 0.0
        self._emitted = 0.0  # сумма розданных per-step reward (dense); терминал добирает остаток
        self.routes: list[list[int]] = []
        self.current_route = [0]
        return self._obs(), self._info()

    @property
    def abs_minute(self) -> float:
        """Абсолютная минута дня для congestion-фазы: offset travel-модели + время тура."""
        return float(self.travel.offset_min) + float(self.cur_time)

    # ---------- feasibility ----------

    def _feasible(self, j: int) -> bool:
        if self.visited[j] or self.demand[j] > self.rem_cap:
            return False
        arrival = self.cur_time + self.travel.time(self.pos, j, self.cur_time)
        if arrival > self.win[j, 1]:  # позже позднего окна l_j
            return False
        start = max(arrival, self.win[j, 0])  # ждём до e_j при раннем приезде
        t_after = start + self.service_min[j]
        return t_after + self.travel.time(j, 0, t_after) <= self.t_max_min  # успеть вернуться

    def _mask(self) -> np.ndarray:
        m = np.zeros(self.k, dtype=np.int8)
        m[0] = 1  # депо всегда допустимо
        for j in range(1, self.k):
            if self._feasible(j):
                m[j] = 1
        return m

    # ---------- obs / info ----------

    def _obs(self):
        mask = self._mask()
        nf = np.zeros((self.k, _NF), dtype=np.float32)
        nf[:, 0:2] = self.coords
        nf[:, 2] = self.demand
        nf[:, 3:5] = self.win
        nf[:, 5] = self.service_min
        nf[:, 6] = self.visited
        nf[:, 7] = mask
        nf[:, 8] = self.travel.node_congestion(self.coords, self.cur_time)  # 0 под free-flow
        vs = np.array([self.pos, self.rem_cap, self.cur_time, self.vehicle_idx], dtype=np.float32)
        tctx = time_context(self.abs_minute, self.dow)
        return {
            "node_features": nf,
            "vehicle_state": vs,
            "time_context": tctx,
            "action_mask": mask,
        }

    def _n_unserved(self) -> int:
        return int((~self.visited[1:]).sum())

    def _info(self, extra=None):
        info = {
            "vehicles_used": self.vehicles_used,
            "total_km": self.total_km,
            "total_time_min": self.total_time_min,
            "n_unserved": self._n_unserved(),
            "late_min": self.total_late_min,
            "routes": [list(r) for r in self.routes],
        }
        if extra:
            info.update(extra)
        return info

    # ---------- динамика ----------

    def _close_route(self, returning: bool) -> None:
        """Финализировать текущий маршрут. returning → добавить возврат в депо."""
        if returning and self.pos != 0:
            dt = self.travel.time(self.pos, 0, self.cur_time)
            self.total_time_min += dt
            self.total_km += self.dist_km[self.pos, 0]
            self.cur_time += dt
            self.current_route.append(0)
            self.pos = 0
        if len(self.current_route) > 2:  # длиннее [0, 0] → обслужил ≥1 аптеку
            self.vehicles_used += 1
        self.routes.append(self.current_route)

    def step(self, action):
        a = int(action)
        mask = self._mask()
        if mask[a] == 0:  # невалидное действие → штраф + терминал
            return self._obs(), -self.p_invalid, True, False, self._info({"invalid_action": a})

        step_km = step_time = 0.0
        if a == 0:  # закрыть маршрут / вернуться в депо
            self._close_route(returning=True)
            self.vehicle_idx += 1
            if self.vehicle_idx < self.K:  # запуск следующей машины из депо
                self.pos = 0
                self.cur_time = 0.0
                self.rem_cap = self.Q
                self.current_route = [0]
        else:  # доехать до аптеки a
            dt = self.travel.time(self.pos, a, self.cur_time)
            arrival = self.cur_time + dt
            wait = max(0.0, self.win[a, 0] - arrival)
            self.total_late_min += max(0.0, arrival - self.win[a, 1])  # ≈0 под маской
            step_km = self.dist_km[self.pos, a]
            step_time = dt + wait + self.service_min[a]
            self.total_km += step_km
            self.total_time_min += step_time
            self.cur_time = arrival + wait + self.service_min[a]
            self.rem_cap -= self.demand[a]
            self.visited[a] = True
            self.pos = a
            self.current_route.append(a)

        self.steps += 1
        terminated = truncated = False
        if bool(self.visited[1:].all()):  # все аптеки обслужены → финальный возврат
            self._close_route(returning=True)
            terminated = True
        elif self.vehicle_idx >= self.K:  # машины кончились с необслуженными
            terminated = True
        if self.steps >= self.max_steps:
            truncated = True

        if terminated or truncated:
            reward = self._terminal_reward()
        elif self.dense:
            reward = -(self.c_d * step_km + self.c_t * step_time / 60.0)
            self._emitted += reward  # копим розданное → терминал добирает остаток
        else:
            reward = 0.0
        return self._obs(), float(reward), terminated, truncated, self._info()

    def _terminal_reward(self) -> float:
        # единый оценщик = источник истины для sparse И dense (та же формула, что у бейзлайнов).
        # dense роздал variable по шагам → терминал добирает остаток (total − уже_выданное), так что
        # суммарный reward эпизода == total в любом режиме; sparse: _emitted=0 → terminal=total.
        total = evaluate_solution(self.routes, self._inst, self._cost_cfg)["reward"]
        return total - self._emitted

    # ---------- helper ----------

    def sample_valid_action(self, obs) -> int:
        """Случайное допустимое действие по action_mask (через self.np_random)."""
        valid = np.flatnonzero(obs["action_mask"])
        return int(self.np_random.choice(valid))
