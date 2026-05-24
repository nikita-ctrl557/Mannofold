"""Run a backtest on synthetic data and persist artifacts for the dashboard.

Usage: python scripts/run_backtest.py [run_id]
"""

from __future__ import annotations

import json
import sys

from mannofold.engine import Engine, EngineConfig, compute_metrics
from mannofold.feed.historical import HistoricalReplayFeed
from mannofold.feed.synthetic import SyntheticConfig, generate_bars
from mannofold.persist.store import LocalStateStore


def main() -> None:
    run_id = sys.argv[1] if len(sys.argv) > 1 else "backtest"
    bars, _ = generate_bars(SyntheticConfig(n_bars=4000, seed=7))
    feed = HistoricalReplayFeed(bars)
    engine = Engine(config=EngineConfig(), store=LocalStateStore(), run_id=run_id)
    result = engine.run(feed)
    metrics = compute_metrics(result.results)
    print(f"run_id={run_id}  steps={len(result.results)}  regimes={len(result.regimes)}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
