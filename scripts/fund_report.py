"""Extensive fund report: every engine as a portfolio across the real S&P universe.

Runs each registered strategy through the unified engine across a broad universe
of real tickers, equal-weights into one daily-rebalanced book at a return-seeking
exposure, and ranks engines by CAGR — flagging which actually BEAT an
equal-weight buy-and-hold benchmark of the same universe. Writes fund_report.json
for the dashboard and prints a desk-style summary.

Usage: python scripts/fund_report.py [n_tickers] [target_vol]
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

from mannofold.analytics.portfolio import run_portfolio
from mannofold.feed.sp500_universe import UNIVERSE, load_universe
from mannofold.signals.strategies import discover

OUT = Path("web/public/runs/fund_report.json")


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    target_vol = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05
    universe = load_universe(UNIVERSE[:n])
    print(f"universe: {len(universe)} real tickers @ target_vol={target_vol}\n")

    engines = discover()
    rows: list[dict] = []
    benchmark: dict | None = None
    for e in engines:
        r = run_portfolio(e.build, universe, target_vol=target_vol)
        benchmark = r["benchmark"]
        p, v = r["portfolio"], r["vs_benchmark"]
        rows.append({
            "name": e.name,
            "cagr": p["cagr"], "total_return": p["total_return"],
            "ann_vol": p["ann_vol"], "sharpe": p["sharpe"], "sortino": p["sortino"],
            "max_drawdown": p["max_drawdown"], "calmar": p["calmar"],
            "hit_rate": p["hit_rate"],
            "alpha_annual": v["alpha_annual"], "beta": v["beta"],
            "info_ratio": v["info_ratio"], "correlation": v["correlation"],
            "beats_benchmark": p["cagr"] > benchmark["cagr"],
        })
        print(f"{e.name:24s} CAGR {p['cagr']*100:+6.1f}%  Sh {p['sharpe']:+.2f}  "
              f"maxDD {p['max_drawdown']*100:+6.1f}%  IR {v['info_ratio']:+.2f}  "
              f"{'BEATS' if rows[-1]['beats_benchmark'] else '     '} bench")

    rows.sort(key=lambda x: x["cagr"], reverse=True)
    n_beat = sum(1 for r in rows if r["beats_benchmark"])
    payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "universe_size": len(universe),
        "target_vol": target_vol,
        "benchmark": benchmark,
        "n_engines": len(rows),
        "n_beat_benchmark": n_beat,
        "rows": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(payload, indent=2)
    OUT.write_text(body)
    Path("data").mkdir(exist_ok=True)
    Path("data/fund_report.json").write_text(body)
    bench_cagr = (benchmark or {}).get("cagr", 0.0)
    print(f"\nbenchmark buy&hold CAGR {bench_cagr*100:+.1f}%  ·  "
          f"{n_beat}/{len(rows)} engines beat it")
    print("TOP 5 by CAGR:", [r["name"] for r in rows[:5]])


if __name__ == "__main__":
    main()
