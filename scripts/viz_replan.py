"""Phase 8 — re-plan animation: an event → instant re-optimisation (GIF).

Two scenarios on seed 0 (real Augsburg): (1) traffic — a jam/closure on a main road, the hit edges
in red; (2) breakdown — a vehicle drops out and its stops are redistributed. Frames: plan → event →
re-plan. Caption: RL reaction ~14–19ms (forward pass) vs OR-Tools 2001ms (×~130). GIF → docs/assets/
(small, in git); no heavy mp4/frames are written. Deterministic (seed 0). The latency numbers are
(ablation rl_raw / polish_summary ortools).

Run: python scripts/viz_replan.py [--seed 0]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import osmnx as ox  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import run_dynamic as rd  # noqa: E402

from logistics_rl_gnn.baselines.greedy import greedy_routes  # noqa: E402
from logistics_rl_gnn.config import congestion as cg  # noqa: E402
from logistics_rl_gnn.config import instance as im  # noqa: E402
from logistics_rl_gnn.env.events import (  # noqa: E402
    BreakdownEvent,
    DynamicState,
    congestion_for,
    make_dynamic_env,
    residual_instance,
    served_by,
)
from logistics_rl_gnn.env.travel import Incident  # noqa: E402

_SNAP = Path("data/snapshots/augsburg_20260720")
_CKPT = Path("results/policy_pomo_congestion.pt")
_DOW = im.DELIVERY_WEEKDAY
_COLORS = plt.cm.tab10.colors
_LAT_RL = "~14–19 ms"  # neural forward pass: the speed CEILING (but quality-inferior, 0010)
_LAT_SYS = "689 ms"  # deployed portfolio+polish system: ×2.9 vs OR AT +3.4% quality (0009)
_LAT_OR = "2001 ms"  # polish_summary ortools


def _rl_replan(pol, residual, travel, fleet):
    env = make_dynamic_env(residual, travel=travel, fleet_size=fleet)
    with torch.no_grad():
        return pol.rollout(env, mode="greedy")[0]


def _near(coords, a, b, center, radius_deg):
    """Edge (a,b) passes near the incident centre (roughly: either end inside the radius)."""
    for n in (a, b):
        dx, dy = abs(coords[n][0] - center[0]), abs(coords[n][1] - center[1])
        if dx < radius_deg and dy < radius_deg:
            return True
    return False


def _frame(graph, inst, routes, title, *, served=None, incident=None, hit_center=None):
    """One frame → an RGB array. served (grey), incident (red circle + hit edges in red)."""
    fig, ax = plt.subplots(figsize=(7.4, 7.4))
    ox.plot_graph(graph, ax=ax, node_size=0, edge_color="#e3e3e3", edge_linewidth=0.4,
                  show=False, close=False, bgcolor="white")
    c = inst.coords
    served = served or set()
    v = 0
    for route in routes:
        if len(route) <= 2:
            continue
        col = _COLORS[v % len(_COLORS)]
        rad = cg.INCIDENT_RADIUS_KM / 90.0
        for a, b in zip(route[:-1], route[1:], strict=False):
            red = hit_center is not None and _near(c, a, b, hit_center, rad)
            ax.plot([c[a][0], c[b][0]], [c[a][1], c[b][1]],
                    "-", color="red" if red else col, lw=2.2 if red else 1.3,
                    alpha=0.95, zorder=4 if red else 3)
        v += 1
    ph = [i for i in range(1, len(inst.demand))]
    sc = [i for i in ph if i in served]
    pend = [i for i in ph if i not in served]
    if sc:
        ax.scatter([c[i][0] for i in sc], [c[i][1] for i in sc], s=12, c="#bbb", zorder=4)
    ax.scatter([c[i][0] for i in pend], [c[i][1] for i in pend], s=16, c="#333", zorder=5)
    ax.scatter([c[0][0]], [c[0][1]], s=180, marker="*", c="crimson", edgecolors="k", zorder=6)
    if incident is not None:
        ax.scatter([incident[0]], [incident[1]], s=520, marker="o", facecolors="none",
                   edgecolors="red", linewidths=2.5, zorder=7)
    ax.set_title(title, fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[..., :3].copy()
    plt.close(fig)
    return img


def _scenario_traffic(graph, inst, exec_routes, exec_travel, pol, now_min):
    served = served_by(exec_routes, inst, exec_travel, now_min)
    # incident centre: the pharmacy stop closest to the city centre (median of coordinates)
    ph = np.array([inst.coords[i] for i in range(1, len(inst.demand))])
    center = tuple(ph[np.argmin(np.hypot(*(ph - ph.mean(0)).T))])
    inc = Incident(center, cg.INCIDENT_RADIUS_KM, np.inf, now_min, 240.0)
    st = DynamicState(inst, _DOW, now_min=now_min)
    st.served = served
    st.incidents = [inc]
    res = residual_instance(st)
    travel = congestion_for(res, dow=_DOW, offset_min=now_min, incidents=st.incidents)
    new = _rl_replan(pol, res, travel, st.fleet(im.FLEET_SIZE))
    # residual routes in the original numbering (to draw them on the full instance)
    idx = [0] + sorted(i for i in range(1, len(inst.demand)) if i not in served)
    new_full = [[idx[n] for n in r] for r in new]
    return [
        _frame(graph, inst, exec_routes, "1. Plan at the start (8 vehicles, 62 pharmacies)"),
        _frame(graph, inst, exec_routes, f"2. Execution (~{now_min:.0f} min, partly served)",
               served=served),
        _frame(graph, inst, exec_routes, "3. EVENT: closure/jam on a main road",
               served=served, incident=center, hit_center=center),
        _frame(graph, inst, new_full,
               f"4. RE-PLAN: neural start {_LAT_RL} (forward pass, no polish)",
               served=served, incident=center),
        _frame(graph, inst, new_full,
               f"neural {_LAT_RL} (ceiling) · system {_LAT_SYS} = ×2.9 vs OR ({_LAT_OR})",
               served=served, incident=center),
    ]


def _scenario_breakdown(graph, inst, exec_routes, exec_travel, pol, now_min):
    served = served_by(exec_routes, inst, exec_travel, now_min)
    st = DynamicState(inst, _DOW, now_min=now_min)
    st.served = served
    BreakdownEvent(now_min).apply(st)  # −1 vehicle; its stops are already in the unserved pool
    res = residual_instance(st)
    travel = congestion_for(res, dow=_DOW, offset_min=now_min)
    new = _rl_replan(pol, res, travel, st.fleet(im.FLEET_SIZE))
    idx = [0] + sorted(i for i in range(1, len(inst.demand)) if i not in served)
    new_full = [[idx[n] for n in r] for r in new]
    return [
        _frame(graph, inst, exec_routes, "1. Plan at the start (8 vehicles)"),
        _frame(graph, inst, exec_routes, f"2. Execution (~{now_min:.0f} min)", served=served),
        _frame(graph, inst, exec_routes, "3. EVENT: a vehicle broke down — stops orphaned",
               served=served),
        _frame(graph, inst, new_full, f"4. RE-PLAN {_LAT_RL}: 7 vehicles, stops redistributed",
               served=served),
        _frame(graph, inst, new_full,
               f"neural {_LAT_RL} (ceiling) · system {_LAT_SYS} = ×2.9 vs OR ({_LAT_OR})",
               served=served),
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 8 — GIF re-plan")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--now-min", type=float, default=60.0)
    ap.add_argument("--outdir", default="docs/assets")
    args = ap.parse_args()

    torch.manual_seed(0)
    inst = im.generate_instance(seed=args.seed)
    exec_travel = congestion_for(inst, dow=_DOW)
    exec_routes = greedy_routes(env=make_dynamic_env(inst, travel=exec_travel))
    pol = rd._load_policy(_CKPT)
    graph = ox.load_graphml(_SNAP / "graph.graphml")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for name, fn in (("traffic", _scenario_traffic), ("breakdown", _scenario_breakdown)):
        frames = fn(graph, inst, exec_routes, exec_travel, pol, args.now_min)
        durations = [1.3, 1.3, 1.8, 1.8, 2.2]  # hold the event/re-plan frames longer
        out = outdir / f"replan_{name}.gif"
        imageio.mimsave(out, frames, duration=durations, loop=0)
        print(f"→ {out}  ({out.stat().st_size / 1024:.0f} KB, {len(frames)} frames)")


if __name__ == "__main__":
    main()
