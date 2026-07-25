"""POMO on statics (Phase 6b Step 1) — Kool rollout baseline replaced by multi-start + shared
baseline.

Per instance: encode ONCE → N trajectories from N DIFFERENT first nodes (allowed at step 0),
sample mode. shared baseline b = mean_N(cost_i); advantage_i = cost_i − b (centred, WITHOUT
std-norm: the POMO baseline is not degenerate, the all-depot pathology of Phase 6 is gone).
loss = mean(adv·Σlogπ), sign +Σlogπ → descent ↓cost (as the Phase 6 fix; a flip → cost grows,
caught by smoke). No rollout baseline, no t-test, no frozen copy (the shared baseline replaces
them; p=nan disappears). grad-clip@1 normalises the step. Inference: multi-start greedy (best of N
starts). 8x augmentation is NOT used (euclidean; ours is a real travel matrix) — see decision 0006.
Runtime guard: |g|≈0 → abort.
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

_FREEZE_PATIENCE = 3  # epochs in a row with grad_norm≈0 → abort (collapse guard, as Phase 6)
_PROBE_N = 16  # fixed train instances in the probe (train-side gauge gap-to-greedy; trend only)


def _ids_hash(id_lists) -> str:
    """sha1[:12] of the instances' node_id sets — freshness detector (RNG advances → new hash)."""
    h = hashlib.sha1()
    for ids in id_lists:
        h.update(repr(tuple(ids)).encode())
    return h.hexdigest()[:12]


def make_env(instance) -> VRPEnv:
    """VRPEnv on a fixed instance (instance_fn ignores seed → reset yields the same instance)."""
    return VRPEnv(instance_fn=lambda s: instance)


def sample_congestion_travel(inst, seed: int, cfg):
    """CongestionTravel on an instance (Step 2): dow=delivery, offset+incidents from seed (det.).

    Incidents sit on pharmacy NODES (on the graph → they hit route edges = coverage) and are
    long-lived (the signal lasts an episode instead of fading before arrival). The t0 snapshot is
    encoded once; reward via evaluate_solution — full time-dependent time (diurnal+decay correct).
    """
    rng = np.random.default_rng(int(seed) + 4242)  # decorrelate from the instance/train RNG
    coords = np.asarray(inst.coords, dtype=float)
    n = len(coords)
    offset = float(rng.uniform(0.0, cfg.cong_offset_max_min))
    n_inc = int(rng.integers(cfg.cong_incidents[0], cfg.cong_incidents[1] + 1))
    incidents = []
    for _ in range(n_inc):
        center = tuple(coords[int(rng.integers(1, n))])  # a pharmacy (not depot=0) → on a route
        closure = bool(rng.random() < cg.INCIDENT_CLOSURE_PROB)
        mag = np.inf if closure else float(rng.uniform(*cg.INCIDENT_MAG_RANGE))
        dur = float(rng.uniform(*cfg.cong_inc_dur_min))
        # t_start=offset (absolute time): the incident is active from episode t0 (decay=1 → visible
        # to the encoder); otherwise offset>dur would leave it expired at the start (coverage hole).
        incidents.append(Incident(center, cg.INCIDENT_RADIUS_KM, mag, offset, dur))
    return congestion_for(inst, dow=cfg.cong_dow, offset_min=offset, incidents=incidents)


def congestion_coverage(envs) -> dict:
    """Share of instances where an INCIDENT hits the graph (advisor gate; diurnal excluded).

    node: node_congestion>0 at ≥1 node (incident inside the zone) — a pure incident indicator (the
    diurnal never sets it). edge: edges above the diurnal background (median multiplier) = incident.
    -> shares ∈ [0,1] + the mean share of hit edges. Too low → raise cong_incidents/mag/radius.
    """
    inc_node, inc_edge, edge_frac = 0, 0, []
    for env in envs:
        env.reset(seed=0)
        ff = np.asarray(env.time_m, dtype=float)
        tm = np.asarray(env.travel.matrix(env.cur_time), dtype=float)
        off = ff > 0
        mult = np.where(np.isinf(tm[off]), 1e9, tm[off]) / ff[off]  # travel/ff on off-diag edges
        base = float(np.median(mult))  # ≈ the uniform diurnal c (incidents are upward outliers)
        hit = mult > base * (1.0 + 1e-3)  # edges above the background = incident
        inc_edge += int(hit.any())
        edge_frac.append(float(hit.mean()))
        inc_node += int(float(env.travel.node_congestion(env.coords, env.cur_time).max()) > 0.0)
    m = len(envs)
    return {
        "inc_node_cov": inc_node / m,  # share with an incident in a node zone (≈1 by construction)
        "inc_edge_cov": inc_edge / m,  # share with an incident on the edges
        "mean_inc_edge_frac": float(np.mean(edge_frac)),  # mean share of hit edges
    }


def _heuristic_greedy_cost(env) -> float:
    """Cost of the nearest-feasible heuristic (Phase 4) under the ENV travel — fixed gap reference.

    travel=env.travel: under congestion the greedy route is scored with congestion time (an honest
    baseline); free-flow → travel.time==time_m → bit-parity with the old path (scoring line 58).
    """
    routes = greedy_routes(env=env)
    return -evaluate_solution(routes, env._inst, env._cost_cfg, travel=env.travel)["reward"]


def feasible_starts(env, obs, max_starts: int) -> list[int]:
    """Deterministic set of first nodes allowed AT STEP 0 (≤ max_starts).

    Only mask==1 is taken (else forcing an infeasible → log_prob=−inf → NaN). Thinned uniformly.
    """
    feas = [j for j in range(1, env.k) if obs["action_mask"][j] == 1]
    if len(feas) <= max_starts:
        return feas
    pick = np.linspace(0, len(feas) - 1, max_starts).round().astype(int)
    return [feas[i] for i in sorted(set(pick.tolist()))]


def _decode(policy, env, enc, start: int, mode: str, *, reset_seed: int = 0):
    """Trajectory from a FORCED first node start, then mode∈{sample,greedy}.

    enc is shared (encode ONCE per instance, survives reset: the graph is static). start comes from
    feasible_starts (allowed). The forced step is EXCLUDED from sum_logp/entropy (POMO canon: prob=1
    for an imposed action → gradient contribution 0). -> (cost, sum_logp, mean_ent, routes).
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
        if not forced_step:  # a forced start is not a π decision (prob=1) → outside the gradient
            logps.append(dist.log_prob(a))
            p = dist.probs  # manual entropy: masked p=0 → 0·log≈0 (no NaN from −inf logits)
            ents.append(-(p * (p + 1e-12).log()).sum())
        obs, _, term, trunc, info = env.step(int(a.item()))
        done = term or trunc
    # travel=env.travel: reward is congestion-aware (teaches congestion routing); free-flow → parity
    cost = -evaluate_solution(info["routes"], env._inst, env._cost_cfg, travel=env.travel)["reward"]
    slp = torch.stack(logps).sum() if logps else torch.zeros((), device=policy.device)
    ent = torch.stack(ents).mean() if ents else torch.zeros((), device=policy.device)
    return cost, slp, ent, info["routes"]


def multistart_greedy(policy, env, max_starts: int, *, reset_seed: int = 0):
    """POMO inference: greedy decode from N allowed starts → (best cost, routes). Fast."""
    obs, _ = env.reset(seed=reset_seed)
    enc = policy.encode(env)  # ONE encode per instance
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
        self.sampler = InstanceSampler(n_range=cfg.n_range)  # train instances
        # eval_sampler: val/test on n_range (lever val_n_range → deployment size if val is weak)
        self.eval_sampler = (
            self.sampler if cfg.val_n_range is None else InstanceSampler(n_range=cfg.val_n_range)
        )
        self.val_envs = [self._wrap_env(self.eval_sampler.sample(s), s) for s in cfg.val_seeds()]
        self.val_heur = float(np.mean([_heuristic_greedy_cost(e) for e in self.val_envs]))
        # train probe: fixed train instances, gauges the train-side gap (memorisation = train↓ val→)
        probe_seeds = range(cfg.train_range[0], cfg.train_range[0] + _PROBE_N)
        self.probe_envs = [self._wrap_env(self.sampler.sample(s), s) for s in probe_seeds]
        self.probe_heur = float(np.mean([_heuristic_greedy_cost(e) for e in self.probe_envs]))
        self.val_ort = val_ort  # OR-Tools reference (injected by the script; None in tests)
        self._test_built = False

    def _wrap_env(self, inst, seed):
        """Env for an instance: congestion travel (Step 2, seed→det.) or free-flow."""
        if self.cfg.congestion:
            ct = sample_congestion_travel(inst, int(seed), self.cfg)
            return make_dynamic_env(inst, travel=ct)
        return make_env(inst)

    def _batch_instance_env(self, seed):
        """(node_ids, env) of one batch instance. Overridden by the residual curriculum (mix)."""
        inst = self.sampler.sample(int(seed))
        return inst.node_ids, self._wrap_env(inst, int(seed))

    def train_batch(self, seeds) -> dict:
        """One step: per instance — N forced starts, shared baseline, gradient accumulation."""
        self.opt.zero_grad()
        b = len(seeds)
        ents, cost_vecs, sampled = [], [], []
        for s in seeds:
            ids, env = self._batch_instance_env(int(s))
            sampled.append(ids)  # for the epoch freshness hash
            obs, _ = env.reset(seed=0)
            starts = feasible_starts(env, obs, self.cfg.max_starts)
            if len(starts) < 2:
                continue  # the shared baseline needs ≥2 starts
            enc = self.policy.encode(env)  # encode ONCE (survives the reset inside _decode)
            costs, logps, tents = [], [], []
            for st in starts:
                c, slp, ent, _ = _decode(self.policy, env, enc, st, "sample")
                costs.append(c)
                logps.append(slp)
                tents.append(ent)
            cost_t = torch.tensor(costs, dtype=torch.float32)
            adv = (cost_t - cost_t.mean()).detach()  # shared baseline (centred)
            loss = (adv * torch.stack(logps)).mean()  # +Σlogπ → ↓cost
            if self.cfg.entropy_beta:
                loss = loss - self.cfg.entropy_beta * torch.stack(tents).mean()
            (loss / b).backward()  # accumulate (the instance graph is freed at once → memory O(1))
            ents.append(float(torch.stack(tents).mean().detach()))
            cost_vecs.append(costs)
        if not cost_vecs:
            raise RuntimeError("no instance in the batch has ≥2 allowed starts")
        gnorm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.grad_clip)
        self.opt.step()
        flat = [c for v in cost_vecs for c in v]
        return {
            "cost": float(np.mean(flat)),
            "entropy": float(np.mean(ents)),
            "grad_norm": float(gnorm),
            "start_std": float(np.mean([np.std(v) for v in cost_vecs])),  # spread across starts
            "cost_vecs": cost_vecs,  # for the non-degeneracy test
            "inst_hash": _ids_hash(sampled),  # freshness of the step's instances (RNG advances)
        }

    def _validate(self) -> dict:
        costs = [multistart_greedy(self.policy, e, self.cfg.max_starts)[0] for e in self.val_envs]
        m = float(np.mean(costs))
        rec = {"val_cost": m, "gap_greedy": m / self.val_heur - 1.0}
        if self.val_ort is not None:
            rec["gap_ort"] = m / self.val_ort - 1.0
        return rec

    def _train_probe(self) -> float:
        """Gap-to-greedy on FIXED train instances (apples-to-apples with val: both multistart).

        train_gap↓ with a frozen val_gap = memorisation. Separate from freshness (the probe is a
        constant gauge, freshness watches the fresh train draws).
        """
        costs = [multistart_greedy(self.policy, e, self.cfg.max_starts)[0] for e in self.probe_envs]
        return float(np.mean(costs)) / self.probe_heur - 1.0

    def test_eval(self, policy=None) -> dict:
        """Held-out TEST (built lazily) — gap-to-greedy of the best policy. NOT for selection."""
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
        if self.cfg.warm_start:  # floor (advisor): deployment is NOT worse than warm start.
            best = self._validate()["val_cost"]  # the starting val is the bar; beat it or nothing
            if self.cfg.ckpt:  # warm start as the floor checkpoint (zero outcome = warm start)
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
                "start_std": float(np.mean([e["start_std"] for e in ep])),  # baseline alive (>0)
                # epoch freshness: hash of step hashes; 3 epochs in a row must differ
                "inst_hash": _ids_hash([e["inst_hash"] for e in ep]),
                "train_gap": self._train_probe(),  # train-side gap-to-greedy (probe)
                **self._validate(),  # val_cost, gap_greedy[, gap_ort]
            }
            rec["mem_gap"] = rec["gap_greedy"] - rec["train_gap"]  # grows (val↑/train↓) → memorise
            history.append(rec)
            if rec["val_cost"] < best - 1e-9:  # best-by-val → checkpoint (selection ONLY by val)
                best, since = rec["val_cost"], 0
                if self.cfg.ckpt:
                    Path(self.cfg.ckpt).parent.mkdir(parents=True, exist_ok=True)
                    torch.save(self.policy.state_dict(), self.cfg.ckpt)
            else:
                since += 1
            rec["since_improve"] = since
            if log_fn:
                log_fn(rec)
            # runtime collapse guard: |g|≈0 = saturation (Phase 6). Abort at epoch ~3, not the max.
            frozen = frozen + 1 if rec["grad_norm"] < 1e-6 else 0
            if frozen >= _FREEZE_PATIENCE:
                raise RuntimeError(
                    f"training frozen: grad_norm≈0 for {frozen} epochs (saturation/collapse) "
                    f"— aborted at epoch {epoch} (not {self.cfg.epochs})"
                )
            if since >= self.cfg.patience:  # early-stop: patience epochs without val improvement
                break
        return history
