"""Render a run as an animated GIF: the manifold filling in + equity building.

The closest thing to watching the live simulation when no browser/preview is
available. Usage: python scripts/render_animation.py [run_id] [out.gif]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

BG = "#0a0d12"
PANEL = "#0c0f14"
FG = "#c9d3df"
DIM = "#6b7682"
N_FRAMES = 90


def _style(ax, title):
    ax.set_facecolor(PANEL)
    ax.set_title(title, color=FG, fontsize=10, loc="left", fontfamily="monospace")
    ax.tick_params(colors=DIM, labelsize=7)
    for s in ax.spines.values():
        s.set_color("#1c2530")


def main() -> None:
    run_id = sys.argv[1] if len(sys.argv) > 1 else "vix"
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(f"data/snapshots/{run_id}.gif")
    out.parent.mkdir(parents=True, exist_ok=True)

    run = json.loads(Path(f"data/runs/{run_id}/run.json").read_text())
    steps = run["steps"]
    reg = Path(f"data/runs/{run_id}/regimes.json")
    cmap = {r["regime_id"]: r["color"] for r in json.loads(reg.read_text())} if reg.exists() else {}

    # Downsample to keep the GIF small and quick to render.
    stride = max(1, len(steps) // 1600)
    s = steps[::stride]
    ex = np.array([p["manifold"]["embedding"][0] for p in s])
    ey = np.array([p["manifold"]["embedding"][1] for p in s])
    colors = [cmap.get(p["manifold"]["regime_id"], "#888888") for p in s]
    equity = np.array([p["portfolio"]["equity"] for p in s])
    n = len(s)
    per = max(1, n // N_FRAMES)

    fig = plt.figure(figsize=(11, 5.2), facecolor=BG)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1], wspace=0.2)
    axm = fig.add_subplot(gs[0, 0])
    axe = fig.add_subplot(gs[0, 1])
    _style(axm, "manifold map — building")
    _style(axe, "equity")
    axm.set_xlim(ex.min() - 0.5, ex.max() + 0.5)
    axm.set_ylim(ey.min() - 0.5, ey.max() + 0.5)
    axm.set_xticks([])
    axm.set_yticks([])
    axe.set_xlim(0, n)
    axe.set_ylim(equity.min() * 0.98, equity.max() * 1.02)

    scat = axm.scatter([], [], s=7, linewidths=0)
    (line,) = axe.plot([], [], color="#76b7b2", lw=1.3)
    head = axm.scatter([], [], s=90, facecolor="none", edgecolor=FG, lw=1.5)
    fig.text(0.012, 0.94, "MANNO·FOLD", color=FG, fontsize=14, fontweight="bold", fontfamily="monospace")
    sym = s[0]["bar"]["symbol"]
    cap = fig.text(0.012, 0.02, "", color=DIM, fontsize=9, fontfamily="monospace")

    def frame(i):
        m = min(n, (i + 1) * per)
        scat.set_offsets(np.c_[ex[:m], ey[:m]])
        scat.set_facecolors(colors[:m])
        head.set_offsets(np.c_[[ex[m - 1]], [ey[m - 1]]])
        line.set_data(range(m), equity[:m])
        ret = (equity[m - 1] / equity[0] - 1) * 100 if equity[0] else 0
        cap.set_text(f"{sym}  ·  {s[m-1]['bar']['ts'][:10]}  ·  {m:,}/{n:,} states  ·  return {ret:+.1f}%")
        return scat, line, head, cap

    anim = FuncAnimation(fig, frame, frames=N_FRAMES, blit=False)
    anim.save(out, writer=PillowWriter(fps=12), dpi=80)
    print(f"wrote {out}  ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
