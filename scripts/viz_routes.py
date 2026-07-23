"""Phase 8 — карта доставки Аугсбурга: «Было» (greedy) vs «Стало» (полир. портфель).

OSM-сеть (серым) + депо Phoenix (звезда) + 62 аптеки (точки) + маршруты по машинам (цвета).
Две панели на ОДНОМ инстансе (seed 0, full-62, free-flow) — честное Было/Стало. PNG (README,
в git) + интерактивный folium HTML (docs/, тяжёлый → вне git). Детерминирован (seed 0).

Запуск: python scripts/viz_routes.py [--seed 0] [--budget-ms 10000]
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
        if len(stops) <= 2:  # пустой маршрут (депо-депо)
            continue
        xs = [coords[n][0] for n in stops]
        ys = [coords[n][1] for n in stops]
        ax.plot(xs, ys, "-", color=_COLORS[v % len(_COLORS)], lw=1.3, alpha=0.9, zorder=3)
        v += 1
    ph = [i for i in range(1, len(inst.demand))]
    ax.scatter([coords[i][0] for i in ph], [coords[i][1] for i in ph],
               s=14, c="#333", zorder=4, label="аптеки (62)")
    ax.scatter([coords[0][0]], [coords[0][1]], s=180, marker="*", c="crimson",
               edgecolors="k", zorder=5, label="депо Phoenix")
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="lower left", fontsize=8)


def _folium_map(inst, routes, out: Path):
    import folium

    c = inst.coords
    m = folium.Map(location=[c[0][1], c[0][0]], zoom_start=12, tiles="cartodbpositron")
    folium.Marker([c[0][1], c[0][0]], tooltip="Депо Phoenix",
                  icon=folium.Icon(color="red", icon="star")).add_to(m)
    for i in range(1, len(inst.demand)):
        folium.CircleMarker([c[i][1], c[i][0]], radius=3, color="#333", fill=True,
                            tooltip=f"аптека {i}").add_to(m)
    v = 0
    pal = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#17becf"]
    for route in routes:
        if len(route) <= 2:
            continue
        folium.PolyLine([[c[n][1], c[n][0]] for n in route], color=pal[v % len(pal)],
                        weight=3, opacity=0.85, tooltip=f"машина {v + 1}").add_to(m)
        v += 1
    out.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out))


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 8 — карта Было/Стало")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget-ms", type=float, default=10000.0)
    ap.add_argument("--out", default="docs/assets/routes_before_after.png")
    ap.add_argument("--folium", default="docs/routes_map_after.html")
    args = ap.parse_args()

    torch.manual_seed(0)
    inst = im.generate_instance(seed=args.seed)
    gr = greedy_routes(env=make_dynamic_env(inst, travel=None, fleet_size=im.FLEET_SIZE))
    pol = rd._load_policy(_CKPT)
    sysr = system_routes(pol, inst, budget_ms=args.budget_ms, k_samples=128, temp=1.0, rl_starts=16)
    gc, sc = _cost(gr, inst), _cost(sysr, inst)

    graph = ox.load_graphml(_SNAP / "graph.graphml")
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.5))
    for ax in axes:  # OSM-сеть серым фоном
        ox.plot_graph(graph, ax=ax, node_size=0, edge_color="#dddddd", edge_linewidth=0.4,
                      show=False, close=False, bgcolor="white")
    _draw_routes(axes[0], inst, gr, f"Было — greedy: {gc:.0f} €")
    _draw_routes(axes[1], inst, sysr, f"Стало — полир. портфель: {sc:.0f} € ({sc / gc - 1:+.0%})")
    fig.suptitle(f"Доставка в аптеки Аугсбурга (seed {args.seed}, 62 аптеки, 1 депо)", fontsize=13)
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=115, bbox_inches="tight")
    plt.close(fig)
    print(f"→ {out}  (greedy {gc:.0f}€ vs система {sc:.0f}€, {sc / gc - 1:+.1%})")

    fol = Path(args.folium)
    _folium_map(inst, sysr, fol)
    kb = fol.stat().st_size / 1024
    print(f"→ {fol}  ({kb:.0f} KB{' — тяжёлый, вне git' if kb > 200 else ''})")


if __name__ == "__main__":
    main()
