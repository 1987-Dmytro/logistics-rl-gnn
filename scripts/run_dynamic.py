"""Phase 7 dynamic run: an event stream over M seeds → RL vs OR-Tools vs greedy on re-plan
LATENCY and QUALITY. Writes results/dynamic.json (+ see decision 0004).

Model: 8 vehicles execute the initial greedy plan (in parallel from 08:00 under diurnal congestion).
At every trigger event (traffic/breakdown/urgent) we take the partial state (served — from the
timeline of the executed plan) and re-plan what REMAINS with each method.

Reproducibility (prohibition #4): both latency AND quality of OR-Tools are wall-clock-dependent (GLS
churns neighbourhoods until the deadline → faster/idler hardware fits more iterations → a DIFFERENT
route → different cost/on-time/unserved; verified: the same seed+config yields different OR-Tools
outcomes). Only RL/greedy quality is deterministic from seed+config (+ a fixed weight checkpoint).
All numbers are median/mean on fixed hardware, tied to config+versions+checkpoint hash in the json.
Run: python scripts/run_dynamic.py [--seeds N] [--deadline S] [--events K].
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import numpy as np
import torch

from logistics_rl_gnn.baselines.greedy import greedy_routes
from logistics_rl_gnn.config import congestion as cg
from logistics_rl_gnn.config import instance as im
from logistics_rl_gnn.env.events import (
    DynamicState,
    congestion_for,
    event_stream,
    make_dynamic_env,
    residual_instance,
    served_by,
)
from logistics_rl_gnn.models.policy import VRPPolicy
from logistics_rl_gnn.replan.compare import compare_replan

_CKPT = Path("results/policy_best.pt")
_OUT = Path("results/dynamic.json")
_METHODS = ("rl", "greedy", "ortools")


def _pkg_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _provenance(ckpt: Path) -> dict:
    """Metric provenance: RL checkpoint hash (outside git) + solver versions (prohibition #4)."""
    return {
        "checkpoint": {
            "path": str(ckpt),
            "sha256_16": hashlib.sha256(ckpt.read_bytes()).hexdigest()[:16],
        },
        "torch": torch.__version__,
        "numpy": np.__version__,
        "ortools": _pkg_version("ortools"),
        "platform": platform.platform(),
    }


def _load_policy(ckpt: Path) -> VRPPolicy:
    if not ckpt.exists():
        raise FileNotFoundError(f"no weights {ckpt} — train first (train_pomo.py)")
    pol = VRPPolicy()
    pol.load_state_dict(torch.load(ckpt, weights_only=True))
    pol.eval()
    return pol


def iter_events(seeds, *, n_events: int):
    """Iterator of the 0004 harness re-plan events → (seed, ev, res, travel, fleet_size).

    The same stream as run(): the diurnal plan is executed from 08:00, at every TRIGGERED event we
    take served from the timeline and re-plan the remainder. Split out so that ablation
    (run_ablation.py) runs THE SAME events with identical provenance but its own systems."""
    dow = im.DELIVERY_WEEKDAY
    base_k = im.FLEET_SIZE
    for seed in seeds:
        inst = im.generate_instance(seed=seed)
        exec_travel = congestion_for(inst, dow=dow)  # diurnal (no incidents) = the plan timeline
        exec_routes = greedy_routes(env=make_dynamic_env(inst, travel=exec_travel))
        state = DynamicState(inst, dow)
        for ev in event_stream(seed, inst, dow, n_events=n_events):
            state.now_min = float(ev.at_min)
            state.served = served_by(exec_routes, inst, exec_travel, ev.at_min)
            if not ev.apply(state):  # mutates incidents/broken/urgent; the diurnal → False
                continue
            n = len(inst.demand)
            pending = [i for i in range(1, n) if i not in state.served]
            if not pending and not state.urgent:
                continue  # everything served → nothing to re-plan
            res = residual_instance(state)
            travel = congestion_for(
                res, dow=dow, offset_min=state.now_min, incidents=state.incidents
            )
            yield seed, ev, res, travel, state.fleet(base_k)


def run(seeds, *, deadline_s: int, n_events: int, ckpt: Path, rl_planner=None) -> dict:
    pol = _load_policy(ckpt)
    records: list[dict] = []
    for seed, ev, res, travel, fleet in iter_events(seeds, n_events=n_events):
        out = compare_replan(
            res, travel, pol, fleet_size=fleet, deadline_s=deadline_s, rl_planner=rl_planner
        )
        for m in _METHODS:
            records.append(
                {
                    "seed": int(seed),
                    "event": ev.kind,
                    "at_min": round(ev.at_min, 1),
                    "n_pending": len(res.demand) - 1,
                    "fleet": fleet,
                    "method": m,
                    **out[m],
                }
            )
    return {"records": records}


def _agg(records, method: str) -> dict:
    r = [x for x in records if x["method"] == method]
    lat = np.array([x["latency_ms"] for x in r])
    cost = np.array([-x["reward"] for x in r])  # € (positive)
    return {
        "n": len(r),
        "latency_ms_median": float(np.median(lat)),
        "latency_ms_mean": float(lat.mean()),
        "cost_eur_mean": float(cost.mean()),
        "on_time_pct_mean": float(np.mean([x["on_time_pct"] for x in r])),
        "unserved_mean": float(np.mean([x["unserved"] for x in r])),
    }


def _print_table(agg: dict) -> None:
    rl_lat = agg["rl"]["latency_ms_median"]
    print("\n=== Phase 7 'after' on reaction speed (re-plan, mean over events) ===")
    print(
        f"{'method':8s} | {'latency (median)':>22s} | {'cost,€':>9s} | "
        f"{'on-time%':>8s} | {'unserved':>8s} | {'× slower than RL':>14s}"
    )
    print("-" * 92)
    for m in _METHODS:
        a = agg[m]
        slow = a["latency_ms_median"] / rl_lat
        unit = f"{a['latency_ms_median']:8.1f} ms"
        print(
            f"{m:8s} | {unit:>22s} | {a['cost_eur_mean']:9.1f} | "
            f"{a['on_time_pct_mean']:8.1f} | {a['unserved_mean']:8.2f} | {slow:13.0f}×"
        )
    print("-" * 92)
    print(
        f"MAIN GATE: RL latency {rl_lat:.1f}ms << OR-Tools "
        f"{agg['ortools']['latency_ms_median']:.0f}ms "
        f"(×{agg['ortools']['latency_ms_median'] / rl_lat:.0f})."
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3, help="number of seeds (0..N-1)")
    ap.add_argument("--deadline", type=int, default=2, help="OR-Tools re-solve deadline, s")
    ap.add_argument("--events", type=int, default=6, help="events per seed")
    ap.add_argument("--ckpt", type=str, default=str(_CKPT), help="RL checkpoint (Step 2)")
    ap.add_argument("--out", type=str, default=str(_OUT), help="output JSON")
    args = ap.parse_args()
    if args.seeds < 1:
        ap.error("--seeds ≥ 1")

    ckpt, out = Path(args.ckpt), Path(args.out)
    seeds = list(range(args.seeds))
    res = run(seeds, deadline_s=args.deadline, n_events=args.events, ckpt=ckpt)
    agg = {m: _agg(res["records"], m) for m in _METHODS}
    _print_table(agg)

    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "seeds": seeds,
            "deadline_s": args.deadline,
            "n_events": args.events,
            "fleet_size": im.FLEET_SIZE,
            "vehicle_cap": im.VEHICLE_CAP,
            "t_max_min": im.T_MAX_MIN,
            "delivery_weekday": im.DELIVERY_WEEKDAY,
            "congestion_tag": cg.CALIBRATION_TAG,
        },
        "provenance": _provenance(ckpt),
        "aggregates": agg,
        "records": res["records"],
        "note": "both latency AND quality of OR-Tools are wall-clock-dependent (GLS until the "
        "deadline → faster hardware = another route = another cost/on-time/unserved). Only "
        "RL/greedy quality is seed+config-deterministic (+ checkpoint hash). RL free-flow-trained.",
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"\nsummary → {out} ({len(res['records'])} records)")


if __name__ == "__main__":
    main()
