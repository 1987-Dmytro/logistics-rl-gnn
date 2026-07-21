"""Распределение инстансов для обучения (Phase 6).

Эпизод = случайное подмножество аптек (~n_range) реального Аугсбурга + сид-спрос. Окна —
реальные (из базовой генерации Phase 3; ASSUMED-фолбэк там же), матрицы/координаты режутся
из полного инстанса. Снапшот грузится ОДИН раз (кэш полного инстанса). Спрос добирается
жадно до target с контролем Σdemand ≤ K·Q → инстанс гарантированно feasible (+ страж на срезе).
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from logistics_rl_gnn.config import instance as im


class InstanceSampler:
    def __init__(self, n_range=(40, 62), *, delivery_weekday=im.DELIVERY_WEEKDAY, base_seed=0):
        self.full = im.generate_instance(
            seed=base_seed, delivery_weekday=delivery_weekday
        )  # 1 load
        self.n_ph = len(self.full.demand) - 1  # аптек в полном (депо = индекс 0)
        self.n_min, self.n_max = n_range
        self.cap = im.FLEET_SIZE * im.VEHICLE_CAP

    def sample(self, seed: int):
        """Инстанс-подмножество по сиду (детерминирован). demand[0]=депо=0."""
        rng = np.random.default_rng(seed)
        target = int(rng.integers(self.n_min, min(self.n_max, self.n_ph) + 1))
        order = rng.permutation(np.arange(1, self.n_ph + 1))  # случайный порядок аптек
        demand = rng.integers(
            im.DEMAND_RANGE[0], im.DEMAND_RANGE[1] + 1, size=self.n_ph + 1
        ).astype(float)
        demand[0] = 0.0
        chosen, total = [], 0.0
        for p in order:  # добираем до target, держа Σdemand ≤ K·Q (feasible by construction)
            if len(chosen) >= target:
                break
            if total + demand[p] <= self.cap:
                chosen.append(int(p))
                total += demand[p]
        return self._slice([0, *sorted(chosen)], demand)

    def _slice(self, idx, demand):
        f = self.full
        sub = replace(
            f,
            node_ids=[f.node_ids[i] for i in idx],
            snapshot_stops=[f.snapshot_stops[i] for i in idx],
            kinds=[f.kinds[i] for i in idx],
            tw_source=[f.tw_source[i] for i in idx],
            time_matrix=f.time_matrix[np.ix_(idx, idx)],
            dist_matrix=f.dist_matrix[np.ix_(idx, idx)],
            coords=f.coords[idx],
            windows=f.windows[idx],
            demand=np.array([demand[i] for i in idx], dtype=float),
            service=f.service[idx],
            meta={**f.meta, "n_sub": len(idx)},
        )
        # страж на срезе (ручной replace минует guard в generate_instance) → тихий баг = громкий
        im._check_feasibility(sub.demand, sub.service, sub.time_matrix, sub.windows[:, 0])
        return sub
