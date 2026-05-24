"""Render a run into a static high-density dashboard PNG (no browser needed).

Mirrors the web dashboard's key panels — manifold map, equity, drawdown, signals —
straight from the persisted run.json, so the result can be viewed on any device.

Usage: python scripts/render_snapshot.py [run_id] [out.png]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BG = "#0a0d12"
PANEL = "#0c0f14"
FG = "#c9d3df"
DIM = "#6b7682"
GOOD = "#59a14f"
BAD = "#e15759"


def _style(ax, title: str) -> None:
    ax.set_facecolor(PANEL)
    ax.set_title(title, color=FG, fontsize=10, loc="left", fontfamily="monospace")
    ax.tick_params(colors=DIM, labelsize=7)
    for s in ax.spines.values():
        s.set_color("#1c2530")


def main() -> None:
    run_id = sys.argv[1] if len(sys.argv) > 1 else "vix"
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(f"data/snapshots/{run_id}.png")
    out.parent.mkdir(parents=True, exist_ok=True)

    run = json.loads(Path(f"data/runs/{run_id}/run.json").read_text())
    steps = run["steps"]
    reg_path = Path(f"data/runs/{run_id}/regimes.json")
    regimes = json.loads(reg_path.read_text()) if reg_path.exists() else []
    cmap = {r["regime_id"]: r["color"] for r in regimes}

    ex = [s["manifold"]["embedding"][0] for s in steps]
    ey = [s["manifold"]["embedding"][1] for s in steps]
    colors = [cmap.get(s["manifold"]["regime_id"], "#888888") for s in steps]
    equity = [s["portfolio"]["equity"] for s in steps]
    dd = [s["portfolio"]["drawdown"] * 100 for s in steps]
    tw = [s["target"]["target_weight"] for s in steps]
    anom = [s["manifold"]["anomaly_score"] for s in steps]

    symbol = steps[0]["bar"]["symbol"]
    d0 = steps[0]["bar"]["ts"][:10]
    d1 = steps[-1]["bar"]["ts"][:10]
    total_ret = (equity[-1] / equity[0] - 1.0) * 100 if equity[0] else 0.0
    max_dd = min(dd) if dd else 0.0
    n_trades = sum(1 for s in steps if s.get("fill"))

    fig = plt.figure(figsize=(16, 9), facecolor=BG)
    gs = fig.add_gridspec(3, 2, width_ratios=[1.7, 1], hspace=0.45, wspace=0.18)

    axm = fig.add_subplot(gs[:, 0])
    _style(axm, f"manifold map · {len(steps):,} market states")
    axm.scatter(ex, ey, c=colors, s=6, alpha=0.5, linewidths=0)
    axm.plot(ex[-80:], ey[-80:], color=FG, lw=0.8, alpha=0.7)
    axm.scatter([ex[-1]], [ey[-1]], s=80, facecolor="none", edgecolor=FG, lw=1.5)
    axm.set_xticks([])
    axm.set_yticks([])

    axe = fig.add_subplot(gs[0, 1])
    _style(axe, "equity")
    axe.plot(equity, color="#76b7b2", lw=1.2)

    axd = fig.add_subplot(gs[1, 1])
    _style(axd, "drawdown %")
    axd.fill_between(range(len(dd)), dd, 0, color=BAD, alpha=0.45)

    axs = fig.add_subplot(gs[2, 1])
    _style(axs, "target weight (teal) · anomaly (orange)")
    axs.plot(tw, color="#76b7b2", lw=0.7)
    axs.plot(anom, color="#f28e2b", lw=0.7, alpha=0.8)

    tone = GOOD if total_ret >= 0 else BAD
    fig.text(0.012, 0.965, "MANNO·FOLD", color=FG, fontsize=16, fontweight="bold", fontfamily="monospace")
    fig.text(
        0.012,
        0.94,
        f"{symbol}   {d0} → {d1}   |   return ",
        color=DIM,
        fontsize=10,
        fontfamily="monospace",
    )
    fig.text(0.255, 0.94, f"{total_ret:+.1f}%", color=tone, fontsize=10, fontweight="bold", fontfamily="monospace")
    fig.text(
        0.33,
        0.94,
        f"   max dd {max_dd:.1f}%   trades {n_trades:,}   regimes {len(regimes)}",
        color=DIM,
        fontsize=10,
        fontfamily="monospace",
    )

    fig.savefig(out, dpi=110, facecolor=BG, bbox_inches="tight")
    print(f"wrote {out}  ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
