"""Phase 8 — training curves (Phase 6 REINFORCE → POMO → congestion → residual).

Parses results/*.log (train cost + entropy per epoch), draws the phase progress in two panels and
marks the events (loss sign-fix, early-stop, gate 0011 FAIL). PNG → docs/assets/ (small, in git —
a documented exception to prohibition #1). Idempotent, no RNG, results/*.json untouched.

Run: python scripts/viz_training.py [--out docs/assets/training_curves.png]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless (CPU server/CI)
import matplotlib.pyplot as plt  # noqa: E402

_EP = re.compile(r"(?:epoch|ep)\s+(\d+)")
_COST = re.compile(r"train(?:_cost)?\s+([\d.]+)")
_H = re.compile(r"\bH\s+([\d.]+)")

# phase chronology: (log, label, colour)
_PHASES = [
    ("results/train_20260721.log", "Phase 6 · REINFORCE", "#9467bd"),
    ("results/pomo_20260721_1914.log", "Phase 6b · POMO static", "#1f77b4"),
    ("results/pomo_congestion_20260722_1019.log", "Path A · congestion", "#2ca02c"),
    ("results/residual_20260722.log", "Path B · residual", "#d62728"),
]


def parse_log(path: str):
    """-> (epochs, costs, entropies) from a train log (regex, robust to phase formatting)."""
    eps, costs, hs = [], [], []
    for line in Path(path).read_text().splitlines():
        m, c, h = _EP.search(line), _COST.search(line), _H.search(line)
        if m and c:
            eps.append(int(m.group(1)))
            costs.append(float(c.group(1)))
            hs.append(float(h.group(1)) if h else float("nan"))
    return eps, costs, hs


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 8 — training curves")
    ap.add_argument("--out", default="docs/assets/training_curves.png")
    args = ap.parse_args()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.5, 7.2))
    plotted = []
    for path, label, color in _PHASES:
        if not Path(path).exists():
            continue
        eps, costs, hs = parse_log(path)
        if not eps:
            continue
        ax1.plot(eps, costs, color=color, lw=1.7, label=label)
        ax2.plot(eps, hs, color=color, lw=1.7, label=label)
        plotted.append((label, color, eps, costs))

    # --- events ---
    if plotted:
        # Phase 6: the loss sign-fix (+Σlogπ) removed the collapse — start of a healthy descent
        p6 = next((p for p in plotted if "REINFORCE" in p[0]), None)
        if p6:
            ax1.annotate(
                "loss sign-fix +Σlogπ\n(Phase 6 collapse removed)",
                xy=(p6[2][0], p6[3][0]), xytext=(8, -38), textcoords="offset points",
                fontsize=7.5, color=p6[1],
                arrowprops=dict(arrowstyle="->", color=p6[1], lw=0.9),
            )
        # early-stop at the end of congestion/residual + the gate FAIL on residual
        for label, color, eps, costs in plotted:
            if "Path A" in label or "Path B" in label:
                ax1.axvline(eps[-1], color=color, ls=":", lw=1.0, alpha=0.6)
                tag = "early-stop\n+ gate 0011 FAIL" if "Path B" in label else "early-stop"
                ax1.annotate(
                    f"{tag}\n(ep{eps[-1]})", xy=(eps[-1], costs[-1]),
                    xytext=(-4, 24), textcoords="offset points", fontsize=7.0, color=color,
                    ha="right", arrowprops=dict(arrowstyle="->", color=color, lw=0.8),
                )

    ax1.set_ylabel("train cost, €")
    ax1.set_title("Training curves — cost / epoch (Phase 6 → POMO → Path A → Path B)")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8, loc="upper right")
    ax2.set_ylabel("policy entropy H")
    ax2.set_xlabel("epoch (within the phase)")
    ax2.set_title("Entropy — training health (not 0 = no collapse, not max = still learning)")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8, loc="upper right")
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"→ {out}  ({', '.join(p[0] for p in plotted)})")


if __name__ == "__main__":
    main()
