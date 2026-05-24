"""Vigorous speed backtest: time the engine over real + large synthetic series.

Reports wall time and throughput (bars/sec). Real data comes from free
GitHub-hosted datasets; the synthetic stress test pushes raw throughput.

Usage: python scripts/benchmark.py [synthetic_n]   (default 100000)
"""

from __future__ import annotations

import sys
import time

from threadpoolctl import threadpool_limits

from mannofold.engine import Engine, EngineConfig, compute_metrics
from mannofold.feed.github_csv import DATASETS, GithubCsvFeed
from mannofold.feed.historical import HistoricalReplayFeed
from mannofold.feed.synthetic import SyntheticConfig, generate_bars

ROW = "{:<16} {:>9} {:>9} {:>9} {:>12} {:>10} {:>9}"


def _time_run(bars, cfg, label, periods_per_year=252 * 26) -> dict:
    t0 = time.perf_counter()
    with threadpool_limits(limits=1):
        result = Engine(config=cfg, run_id=f"bench_{label}").run(HistoricalReplayFeed(bars))
    wall = time.perf_counter() - t0
    m = compute_metrics(result.results, periods_per_year=periods_per_year)
    return {
        "label": label,
        "bars": len(bars),
        "steps": len(result.results),
        "wall_s": wall,
        "bars_per_s": len(bars) / wall if wall else 0.0,
        "ret": m.get("total_return", 0.0),
        "sharpe": m.get("sharpe", 0.0),
    }


def _print(rows: list[dict]) -> None:
    print(ROW.format("dataset", "bars", "steps", "wall_s", "bars/sec", "return", "sharpe"))
    print("-" * 82)
    for r in rows:
        print(
            ROW.format(
                r["label"],
                r["bars"],
                r["steps"],
                f"{r['wall_s']:.2f}",
                f"{r['bars_per_s']:,.0f}",
                f"{r['ret'] * 100:.2f}%",
                f"{r['sharpe']:.2f}",
            )
        )


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000
    rows: list[dict] = []

    for name in ("vix", "aapl"):
        feed = GithubCsvFeed(name)
        bars = list(feed.stream())
        cfg = EngineConfig(train_size=min(500, len(bars) // 3), refit_every=250, max_train=1000)
        rows.append(_time_run(bars, cfg, f"{name}({DATASETS[name].symbol})", periods_per_year=252))

    print(f"\nGenerating {n:,} synthetic bars for the speed stress test...")
    big, _ = generate_bars(SyntheticConfig(n_bars=n, seed=3))
    fast = EngineConfig(train_size=2000, refit_every=4000, max_train=4000)
    rows.append(_time_run(big, fast, f"synth({n // 1000}k)"))

    print()
    _print(rows)


if __name__ == "__main__":
    main()
