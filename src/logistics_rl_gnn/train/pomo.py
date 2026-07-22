"""POMO на статике (Phase 6b Шаг 1) — замена Kool rollout-baseline на multi-start + shared baseline.

На инстанс: encode ОДИН раз → N траекторий из N РАЗНЫХ первых узлов (допустимых на шаге 0),
sample-режим. shared baseline b = mean_N(cost_i); advantage_i = cost_i − b (центрирован, БЕЗ
std-норм: POMO baseline не вырожден, all-depot-патологии Phase 6 нет). loss = mean(adv·Σlogπ),
знак +Σlogπ → спуск ↓cost (как фикс Phase 6; флип → cost растёт, ловится smoke). Ни rollout-
baseline, ни t-test, ни frozen-copy (shared baseline их заменяет; p=nan уходит). grad-clip@1
нормирует шаг. Инференс: multi-start greedy (лучший из N стартов). 8x-augmentation НЕ применяем
(евклидова, у нас реальная travel-матрица) — см. decision 0006. Runtime-страж: |g|≈0 → обрыв.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

from logistics_rl_gnn.baselines.greedy import greedy_routes
from logistics_rl_gnn.config import congestion as cg
from logistics_rl_gnn.env.events import congestion_for, make_dynamic_env
from logistics_rl_gnn.env.scoring import evaluate_solution
from logistics_rl_gnn.env.travel import Incident
from logistics_rl_gnn.env.vrp_env import VRPEnv
from logistics_rl_gnn.train.instance_sampler import InstanceSampler

_FREEZE_PATIENCE = 3  # эпох подряд с grad_norm≈0 → обрыв (страж коллапса, как Phase 6)
_PROBE_N = 16  # фикс. train-инстансов в probe (train-side gauge gap-to-greedy; тренд, не точность)


def _ids_hash(id_lists) -> str:
    """sha1[:12] от node_id-наборов инстансов — детектор свежести (RNG продвигается → хеш иной)."""
    h = hashlib.sha1()
    for ids in id_lists:
        h.update(repr(tuple(ids)).encode())
    return h.hexdigest()[:12]


def make_env(instance) -> VRPEnv:
    """VRPEnv на фиксированном инстансе (instance_fn игнорит seed → reset даёт тот же инстанс)."""
    return VRPEnv(instance_fn=lambda s: instance)


def sample_congestion_travel(inst, seed: int, cfg):
    """CongestionTravel на инстансе (Шаг 2): dow=delivery, offset+инциденты из seed (детерминир.).

    Инциденты в УЗЛАХ-аптеках (на графе → задевают маршрутные рёбра = coverage) + долгоживущие
    (сигнал держится эпизод, не затухает до приезда). t0-снимок кодируется энкодером один раз;
    reward через evaluate_solution — полное time-dependent время (диурнал+затухание корректны).
    """
    rng = np.random.default_rng(int(seed) + 4242)  # декоррелируем от instance/train RNG
    coords = np.asarray(inst.coords, dtype=float)
    n = len(coords)
    offset = float(rng.uniform(0.0, cfg.cong_offset_max_min))
    n_inc = int(rng.integers(cfg.cong_incidents[0], cfg.cong_incidents[1] + 1))
    incidents = []
    for _ in range(n_inc):
        center = tuple(coords[int(rng.integers(1, n))])  # аптека (не депо=0) → на маршруте
        closure = bool(rng.random() < cg.INCIDENT_CLOSURE_PROB)
        mag = np.inf if closure else float(rng.uniform(*cg.INCIDENT_MAG_RANGE))
        dur = float(rng.uniform(*cfg.cong_inc_dur_min))
        # t_start=offset (абс-время): инцидент активен с t0 эпизода (decay=1 → виден энкодеру);
        # иначе offset>dur оставлял бы инцидент истёкшим к старту (coverage-дыра).
        incidents.append(Incident(center, cg.INCIDENT_RADIUS_KM, mag, offset, dur))
    return congestion_for(inst, dow=cfg.cong_dow, offset_min=offset, incidents=incidents)


def congestion_coverage(envs) -> dict:
    """Доля инстансов, где ИНЦИДЕНТ реально задевает граф (advisor-гейт, диурнал не в счёт).

    node: node_congestion>0 у ≥1 узла (инцидент в зоне) — чистый инцидент-индикатор (диурнал его
    не ставит). edge: рёбра выше диурнального фона (median мультипликатора) = инцидент на рёбрах.
    -> доли ∈ [0,1] + средняя доля задетых рёбер. Мало → поднять cong_incidents/mag/radius.
    """
    inc_node, inc_edge, edge_frac = 0, 0, []
    for env in envs:
        env.reset(seed=0)
        ff = np.asarray(env.time_m, dtype=float)
        tm = np.asarray(env.travel.matrix(env.cur_time), dtype=float)
        off = ff > 0
        mult = np.where(np.isinf(tm[off]), 1e9, tm[off]) / ff[off]  # travel/ff на недиаг. рёбрах
        base = float(np.median(mult))  # ≈ равномерный диурнал c (инциденты — выбросы вверх)
        hit = mult > base * (1.0 + 1e-3)  # рёбра выше фона = инцидент
        inc_edge += int(hit.any())
        edge_frac.append(float(hit.mean()))
        inc_node += int(float(env.travel.node_congestion(env.coords, env.cur_time).max()) > 0.0)
    m = len(envs)
    return {
        "inc_node_cov": inc_node / m,  # доля с инцидентом в зоне узла (≈1 by construction)
        "inc_edge_cov": inc_edge / m,  # доля с инцидентом на рёбрах
        "mean_inc_edge_frac": float(np.mean(edge_frac)),  # ср. доля задетых рёбер
    }


def _heuristic_greedy_cost(env) -> float:
    """Стоимость эвристики nearest-feasible (Phase 4) под travel СРЕДЫ — фикс. референс gap.

    travel=env.travel: под congestion greedy-маршрут оценён congestion-временем (честный baseline);
    free-flow → travel.time==time_m → бит-паритет с прежним (evaluate_solution стр.58).
    """
    routes = greedy_routes(env=env)
    return -evaluate_solution(routes, env._inst, env._cost_cfg, travel=env.travel)["reward"]


def feasible_starts(env, obs, max_starts: int) -> list[int]:
    """Детерминированный набор допустимых-НА-ШАГЕ-0 первых узлов (≤ max_starts).

    Берём только mask==1 (иначе форс инфеасибл → log_prob=−inf → NaN). Прореживаем равномерно.
    """
    feas = [j for j in range(1, env.k) if obs["action_mask"][j] == 1]
    if len(feas) <= max_starts:
        return feas
    pick = np.linspace(0, len(feas) - 1, max_starts).round().astype(int)
    return [feas[i] for i in sorted(set(pick.tolist()))]


def _decode(policy, env, enc, start: int, mode: str, *, reset_seed: int = 0):
    """Траектория из ФОРСИРОВАННОГО первого узла start, дальше mode∈{sample,greedy}.

    enc — общий (encode ОДИН раз на инстанс, переживает reset: статический граф). start взят из
    feasible_starts (допустим). Форс-шаг ИСКЛЮЧЁН из sum_logp/энтропии (канон POMO: prob=1 у
    навязанного действия → вклад в градиент 0). -> (cost, sum_logp, mean_ent, routes).
    """
    obs, _ = env.reset(seed=reset_seed)
    logps, ents = [], []
    forced = start
    done = False
    while not done:
        dist = policy.action_dist(env, obs, enc)
        forced_step = forced is not None
        if forced_step:
            a = torch.as_tensor(forced, device=policy.device)
            forced = None
        elif mode == "greedy":
            a = dist.probs.argmax()
        else:
            a = dist.sample()
        if not forced_step:  # форс-старт — не решение π (prob=1) → вне градиента (канон POMO)
            logps.append(dist.log_prob(a))
            p = dist.probs  # ручная энтропия: маскед p=0 → 0·log≈0 (без NaN от −inf логитов)
            ents.append(-(p * (p + 1e-12).log()).sum())
        obs, _, term, trunc, info = env.step(int(a.item()))
        done = term or trunc
    # travel=env.travel: reward congestion-aware (учит congestion-роутинг); free-flow → паритет
    cost = -evaluate_solution(info["routes"], env._inst, env._cost_cfg, travel=env.travel)["reward"]
    slp = torch.stack(logps).sum() if logps else torch.zeros((), device=policy.device)
    ent = torch.stack(ents).mean() if ents else torch.zeros((), device=policy.device)
    return cost, slp, ent, info["routes"]


def multistart_greedy(policy, env, max_starts: int, *, reset_seed: int = 0):
    """Инференс POMO: greedy-decode из N допустимых стартов → (лучшая cost, routes). Быстро."""
    obs, _ = env.reset(seed=reset_seed)
    enc = policy.encode(env)  # ОДИН encode на инстанс
    starts = feasible_starts(env, obs, max_starts)
    best_c, best_r = float("inf"), None
    with torch.no_grad():
        for st in starts:
            c, _, _, r = _decode(policy, env, enc, st, "greedy", reset_seed=reset_seed)
            if c < best_c:
                best_c, best_r = c, r
    return best_c, best_r


class POMOTrainer:
    def __init__(self, policy, cfg, *, val_ort: float | None = None):
        self.policy, self.cfg = policy, cfg
        self.opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
        self.sampler = InstanceSampler(n_range=cfg.n_range)  # train-инстансы
        # eval_sampler: val/test на n_range (лычаг val_n_range → deployment-размер, если val слаб)
        self.eval_sampler = (
            self.sampler if cfg.val_n_range is None else InstanceSampler(n_range=cfg.val_n_range)
        )
        self.val_envs = [self._wrap_env(self.eval_sampler.sample(s), s) for s in cfg.val_seeds()]
        self.val_heur = float(np.mean([_heuristic_greedy_cost(e) for e in self.val_envs]))
        # train-probe: фикс. train-инстансы, gauge train-side gap (memorization = train↓ vs val→)
        probe_seeds = range(cfg.train_range[0], cfg.train_range[0] + _PROBE_N)
        self.probe_envs = [self._wrap_env(self.sampler.sample(s), s) for s in probe_seeds]
        self.probe_heur = float(np.mean([_heuristic_greedy_cost(e) for e in self.probe_envs]))
        self.val_ort = val_ort  # OR-Tools-референс (инъекция из скрипта; None в тестах)
        self._test_built = False

    def _wrap_env(self, inst, seed):
        """Env инстанса: congestion-travel (Шаг 2, seed→детерм.) либо free-flow."""
        if self.cfg.congestion:
            ct = sample_congestion_travel(inst, int(seed), self.cfg)
            return make_dynamic_env(inst, travel=ct)
        return make_env(inst)

    def _batch_instance_env(self, seed):
        """(node_ids, env) одного инстанса батча. Переопределяется residual-куррикулумом (микс)."""
        inst = self.sampler.sample(int(seed))
        return inst.node_ids, self._wrap_env(inst, int(seed))

    def train_batch(self, seeds) -> dict:
        """Один шаг: на каждый инстанс — N форс-стартов, shared baseline, градиент-аккумуляция."""
        self.opt.zero_grad()
        b = len(seeds)
        ents, cost_vecs, sampled = [], [], []
        for s in seeds:
            ids, env = self._batch_instance_env(int(s))
            sampled.append(ids)  # для freshness-хеша эпохи
            obs, _ = env.reset(seed=0)
            starts = feasible_starts(env, obs, self.cfg.max_starts)
            if len(starts) < 2:
                continue  # shared baseline требует ≥2 старта
            enc = self.policy.encode(env)  # encode ОДИН раз (переживает reset в _decode)
            costs, logps, tents = [], [], []
            for st in starts:
                c, slp, ent, _ = _decode(self.policy, env, enc, st, "sample")
                costs.append(c)
                logps.append(slp)
                tents.append(ent)
            cost_t = torch.tensor(costs, dtype=torch.float32)
            adv = (cost_t - cost_t.mean()).detach()  # shared baseline (центрирован)
            loss = (adv * torch.stack(logps)).mean()  # +Σlogπ → ↓cost
            if self.cfg.entropy_beta:
                loss = loss - self.cfg.entropy_beta * torch.stack(tents).mean()
            (loss / b).backward()  # аккумулируем (граф инстанса освобождается сразу → память O(1))
            ents.append(float(torch.stack(tents).mean().detach()))
            cost_vecs.append(costs)
        if not cost_vecs:
            raise RuntimeError("в батче нет инстансов с ≥2 допустимыми стартами")
        gnorm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.grad_clip)
        self.opt.step()
        flat = [c for v in cost_vecs for c in v]
        return {
            "cost": float(np.mean(flat)),
            "entropy": float(np.mean(ents)),
            "grad_norm": float(gnorm),
            "start_std": float(np.mean([np.std(v) for v in cost_vecs])),  # разброс стартов
            "cost_vecs": cost_vecs,  # для теста non-degeneracy
            "inst_hash": _ids_hash(sampled),  # свежесть инстансов шага (RNG продвигается)
        }

    def _validate(self) -> dict:
        costs = [multistart_greedy(self.policy, e, self.cfg.max_starts)[0] for e in self.val_envs]
        m = float(np.mean(costs))
        rec = {"val_cost": m, "gap_greedy": m / self.val_heur - 1.0}
        if self.val_ort is not None:
            rec["gap_ort"] = m / self.val_ort - 1.0
        return rec

    def _train_probe(self) -> float:
        """Gap-to-greedy на ФИКС. train-инстансах (apples-to-apples к val: оба greedy multistart).

        train_gap↓ при застывшем val_gap = memorization. Отдельно от freshness (probe — константный
        gauge, freshness — над свежими train-драйвами).
        """
        costs = [multistart_greedy(self.policy, e, self.cfg.max_starts)[0] for e in self.probe_envs]
        return float(np.mean(costs)) / self.probe_heur - 1.0

    def test_eval(self, policy=None) -> dict:
        """Held-out TEST (ленивое построение) — gap-to-greedy лучшей политики. НЕ для отбора."""
        pol = self.policy if policy is None else policy
        if not self._test_built:
            self.test_envs = [
                self._wrap_env(self.eval_sampler.sample(s), s) for s in self.cfg.test_seeds()
            ]
            self.test_heur = float(np.mean([_heuristic_greedy_cost(e) for e in self.test_envs]))
            self._test_built = True
        costs = [multistart_greedy(pol, e, self.cfg.max_starts)[0] for e in self.test_envs]
        m = float(np.mean(costs))
        return {"test_cost": m, "test_gap_greedy": m / self.test_heur - 1.0}

    def fit(self, log_fn=None) -> list[dict]:
        rng = np.random.default_rng(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)
        best, history, frozen, since = float("inf"), [], 0, 0
        if self.cfg.warm_start:  # floor (advisor): деплой НЕ хуже warm-start под congestion.
            best = self._validate()["val_cost"]  # стартовый val = планка; бьём только если ниже
            if self.cfg.ckpt:  # warm-start как floor-чекпойнт (нулевой исход = warm-start)
                Path(self.cfg.ckpt).parent.mkdir(parents=True, exist_ok=True)
                torch.save(self.policy.state_dict(), self.cfg.ckpt)
        for epoch in range(self.cfg.epochs):
            ep = [
                self.train_batch(rng.integers(*self.cfg.train_range, size=self.cfg.batch))
                for _ in range(self.cfg.steps_per_epoch)
            ]
            rec = {
                "epoch": epoch,
                "train_cost": float(np.mean([e["cost"] for e in ep])),
                "entropy": float(np.mean([e["entropy"] for e in ep])),
                "grad_norm": float(np.mean([e["grad_norm"] for e in ep])),
                "start_std": float(np.mean([e["start_std"] for e in ep])),  # baseline жив (>0)
                # freshness эпохи: хеш шаговых хешей; 3 эпохи подряд должны различаться
                "inst_hash": _ids_hash([e["inst_hash"] for e in ep]),
                "train_gap": self._train_probe(),  # train-side gap-to-greedy (probe)
                **self._validate(),  # val_cost, gap_greedy[, gap_ort]
            }
            rec["mem_gap"] = rec["gap_greedy"] - rec["train_gap"]  # растёт (val↑/train↓) → memorize
            history.append(rec)
            if rec["val_cost"] < best - 1e-9:  # best-by-val → чекпойнт (отбор ТОЛЬКО по val)
                best, since = rec["val_cost"], 0
                if self.cfg.ckpt:
                    Path(self.cfg.ckpt).parent.mkdir(parents=True, exist_ok=True)
                    torch.save(self.policy.state_dict(), self.cfg.ckpt)
            else:
                since += 1
            rec["since_improve"] = since
            if log_fn:
                log_fn(rec)
            # runtime-страж коллапса: |g|≈0 = насыщение (Phase 6). Обрыв на epoch ~3, НЕ epochs_max.
            frozen = frozen + 1 if rec["grad_norm"] < 1e-6 else 0
            if frozen >= _FREEZE_PATIENCE:
                raise RuntimeError(
                    f"обучение заморожено: grad_norm≈0 {frozen} эпох подряд (насыщение/коллапс) "
                    f"— прогон прерван на epoch {epoch} (не {self.cfg.epochs})"
                )
            if since >= self.cfg.patience:  # early-stop: patience эпох без улучшения val
                break
        return history
