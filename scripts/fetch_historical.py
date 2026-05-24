"""Download a free historical dataset from GitHub and run a backtest on it.

Usage: python scripts/fetch_historical.py [dataset] [run_id]
       dataset in {vix, aapl, sp500}  (default: vix)
"""

from __future__ import annotations

import json
import sys

from mannofold.engine import Engine, EngineConfig, compute_metrics
from mannofold.feed.github_csv import DATASETS, GithubCsvFeed
from mannofold.persist.store import LocalStateStore


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "vix"
    run_id = sys.argv[2] if len(sys.argv) > 2 else name
    if name not in DATASETS:
        raise SystemExit(f"unknown dataset {name!r}; choose from {sorted(DATASETS)}")

    feed = GithubCsvFeed(name)
    start, end = feed.date_range()
    print(f"{name}: {len(feed)} real bars  {start.date()} .. {end.date()}")
    print(f"   {DATASETS[name].description}")

    # Smaller train/refit windows suit shorter daily series than the synthetic default.
    cfg = EngineConfig(train_size=min(500, len(feed) // 3), refit_every=250, max_train=1000)
    result = Engine(config=cfg, store=LocalStateStore(), run_id=run_id).run(feed)
    # Annualize Sharpe by the dataset's true bar frequency (daily / monthly).
    periods = {"sp500": 12}.get(name, 252)
    print(f"run_id={run_id}  steps={len(result.results)}  regimes={len(result.regimes)}")
    print(json.dumps(compute_metrics(result.results, periods_per_year=periods), indent=2))


if __name__ == "__main__":
    main()
