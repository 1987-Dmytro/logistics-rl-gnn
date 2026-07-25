"""Phase 8 — Augsburg delivery map: "before" (greedy) vs "after" (polished portfolio).

OSM network (grey) + the Phoenix depot (star) + 62 pharmacies (dots) + routes per vehicle (colours).
Two panels on ONE instance (seed 0, full-62, free-flow) — an honest before/after. PNG (README, in
git) + an interactive folium HTML (docs/, heavy → outside git). Deterministic (seed 0).

Run: python scripts/viz_routes.py [--seed 0] [--budget-ms 10000]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import osmnx as ox  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import route_sheet as rs  # noqa: E402
import run_dynamic as rd  # noqa: E402
from eval_system import system_routes  # noqa: E402

from logistics_rl_gnn.baselines.greedy import greedy_routes  # noqa: E402
from logistics_rl_gnn.config import instance as im  # noqa: E402
from logistics_rl_gnn.env.events import make_dynamic_env  # noqa: E402
from logistics_rl_gnn.env.scoring import CostConfig, evaluate_solution  # noqa: E402

_SNAP = Path("data/snapshots/augsburg_20260720")
_CKPT = Path("results/policy_pomo_congestion.pt")
_CFG = CostConfig()
_COLORS = plt.cm.tab10.colors


def _cost(routes, inst):
    return -evaluate_solution(routes, inst, _CFG)["reward"]


def _draw_routes(ax, inst, routes, title):
    coords = inst.coords
    v = 0
    for route in routes:
        stops = [n for n in route]
        if len(stops) <= 2:  # empty route (depot-depot)
            continue
        xs = [coords[n][0] for n in stops]
        ys = [coords[n][1] for n in stops]
        ax.plot(xs, ys, "-", color=_COLORS[v % len(_COLORS)], lw=1.3, alpha=0.9, zorder=3)
        v += 1
    ph = [i for i in range(1, len(inst.demand))]
    ax.scatter([coords[i][0] for i in ph], [coords[i][1] for i in ph],
               s=14, c="#333", zorder=4, label="pharmacies (62)")
    ax.scatter([coords[0][0]], [coords[0][1]], s=180, marker="*", c="crimson",
               edgecolors="k", zorder=5, label="Phoenix depot")
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="lower left", fontsize=8)


def _folium_map(inst, routes, out: Path, names=None):
    """Interactive map: polylines per vehicle + numbered stops in visiting order,
    popup (pharmacy name/window/ETA). ETA/window — the same walk as route_sheet (one semantics)."""
    import folium

    names = names or {}
    c = inst.coords
    base = inst.start_datetime
    m = folium.Map(location=[c[0][1], c[0][0]], zoom_start=12, tiles="cartodbpositron")
    folium.Marker([c[0][1], c[0][0]], tooltip="PHOENIX depot",
                  icon=folium.Icon(color="red", icon="star")).add_to(m)
    pal = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#17becf"]
    v = 0
    for route in routes:
        if len(route) <= 2:
            continue
        col = pal[v % len(pal)]
        folium.PolyLine([[c[n][1], c[n][0]] for n in route], color=col,
                        weight=3, opacity=0.85, tooltip=f"vehicle {v + 1}").add_to(m)
        stops, _ = rs.walk_route(route, inst)
        for k, s in enumerate(stops, start=1):  # k = visiting order within this vehicle
            n = s["n"]
            popup = folium.Popup(
                f"<b>{k}. {rs._label(s['snap'], names)}</b><br>veh. {v + 1} · "
                f"ETA {rs._clock(base, s['arr_min'])}<br>window "
                f"{rs._clock(base, s['e_min'])}–{rs._clock(base, s['l_min'])}",
                max_width=260,
            )
            folium.Marker(
                [c[n][1], c[n][0]], popup=popup,
                icon=folium.DivIcon(html=(
                    f'<div style="font-size:10px;color:#fff;background:{col};border-radius:50%;'
                    f'width:18px;height:18px;text-align:center;line-height:18px;'
                    f'border:1px solid #333">{k}</div>'), icon_size=(18, 18), icon_anchor=(9, 9)),
            ).add_to(m)
        v += 1
    out.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out))


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 8 — before/after map")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget-ms", type=float, default=10000.0)
    ap.add_argument("--out", default="docs/assets/routes_before_after.png")
    ap.add_argument("--folium", default="docs/routes_map_after.html")
    args = ap.parse_args()

    torch.manual_seed(0)
    # instance on THE SAME snapshot as the graph background and names (else coordinates/labels
    # desynchronise across snapshots); _SNAP — the snapshot whose graph.graphml is loaded below
    inst = im.generate_instance(snapshot_dir=_SNAP, seed=args.seed)
    gr = greedy_routes(env=make_dynamic_env(inst, travel=None, fleet_size=im.FLEET_SIZE))
    pol = rd._load_policy(_CKPT)
    sysr = system_routes(pol, inst, budget_ms=args.budget_ms, k_samples=128, temp=1.0, rl_starts=16)
    gc, sc = _cost(gr, inst), _cost(sysr, inst)

    graph = ox.load_graphml(_SNAP / "graph.graphml")
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.5))
    for ax in axes:  # OSM network as a grey background
        ox.plot_graph(graph, ax=ax, node_size=0, edge_color="#dddddd", edge_linewidth=0.4,
                      show=False, close=False, bgcolor="white")
    _draw_routes(axes[0], inst, gr, f"Before — greedy: {gc:.0f} €")
    _draw_routes(axes[1], inst, sysr, f"After — polished: {sc:.0f} € ({sc / gc - 1:+.0%})")
    fig.suptitle(f"Augsburg delivery (seed {args.seed}, 62 pharmacies, 1 depot)", fontsize=13)
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=115, bbox_inches="tight")
    plt.close(fig)
    print(f"→ {out}  (greedy {gc:.0f}€ vs system {sc:.0f}€, {sc / gc - 1:+.1%})")

    fol = Path(args.folium)
    _folium_map(inst, sysr, fol, names=rs.load_names(_SNAP))
    kb = fol.stat().st_size / 1024
    print(f"→ {fol}  ({kb:.0f} KB{' — heavy, outside git' if kb > 200 else ''})")


if __name__ == "__main__":
    main()
