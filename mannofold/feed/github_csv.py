"""Free historical-data feed sourced from public GitHub-hosted CSV datasets.

This sandbox's network policy only permits GitHub (raw.githubusercontent.com) —
Yahoo/Stooq/Binance/etc. are blocked — so we pull real OHLC(V) history from
well-known free CSV datasets on GitHub. Files are cached under data/cache/ after
first download. Conforms to the DataFeed Protocol (mode = BACKTEST).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pandas as pd

from mannofold.contracts.models import Bar, Mode

CACHE_DIR = Path("data/cache")


@dataclass(frozen=True)
class Dataset:
    url: str
    symbol: str
    date_col: str
    open_col: str | None = None
    high_col: str | None = None
    low_col: str | None = None
    close_col: str = "CLOSE"
    volume_col: str | None = None
    description: str = ""


# Curated free datasets reachable from this environment (GitHub raw CSV).
DATASETS: dict[str, Dataset] = {
    "vix": Dataset(
        url="https://raw.githubusercontent.com/datasets/finance-vix/main/data/vix-daily.csv",
        symbol="VIX",
        date_col="DATE",
        open_col="OPEN",
        high_col="HIGH",
        low_col="LOW",
        close_col="CLOSE",
        description="CBOE VIX daily OHLC since 1990 (~9,200 bars).",
    ),
    "aapl": Dataset(
        url="https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv",
        symbol="AAPL",
        date_col="Date",
        open_col="AAPL.Open",
        high_col="AAPL.High",
        low_col="AAPL.Low",
        close_col="AAPL.Close",
        volume_col="AAPL.Volume",
        description="Apple daily OHLCV 2014-2016 (~500 bars).",
    ),
    "sp500": Dataset(
        url="https://raw.githubusercontent.com/datasets/s-and-p-500/master/data/data.csv",
        symbol="SP500",
        date_col="Date",
        close_col="SP500",
        description="S&P 500 monthly index (close only) since 1870 (~1,800 bars).",
    ),
}


def _download(ds: Dataset, force: bool = False) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"{ds.symbol}.csv"
    if cache.exists() and not force:
        return cache
    resp = httpx.get(ds.url, timeout=30.0, follow_redirects=True)
    resp.raise_for_status()
    cache.write_bytes(resp.content)
    return cache


def load_bars(name_or_dataset: str | Dataset, force: bool = False) -> list[Bar]:
    """Load (and cache) a dataset's bars in chronological order."""
    ds = name_or_dataset if isinstance(name_or_dataset, Dataset) else DATASETS[name_or_dataset]
    df = pd.read_csv(_download(ds, force=force))
    df = df.dropna(subset=[ds.date_col, ds.close_col])

    bars: list[Bar] = []
    for _, row in df.iterrows():
        close = float(row[ds.close_col])
        open_ = float(row[ds.open_col]) if ds.open_col else close
        high = float(row[ds.high_col]) if ds.high_col else max(open_, close)
        low = float(row[ds.low_col]) if ds.low_col else min(open_, close)
        volume = float(row[ds.volume_col]) if ds.volume_col else 0.0
        ts = pd.to_datetime(row[ds.date_col]).to_pydatetime().replace(tzinfo=UTC)
        bars.append(
            Bar(ts=ts, symbol=ds.symbol, open=open_, high=high, low=low, close=close, volume=volume)
        )
    bars.sort(key=lambda b: b.ts)
    return bars


class GithubCsvFeed:
    """Backtest feed backed by a free GitHub-hosted historical CSV dataset."""

    mode = Mode.BACKTEST

    def __init__(self, dataset: str = "vix", force_download: bool = False):
        self._bars = load_bars(dataset, force=force_download)
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self._bars)

    def date_range(self) -> tuple[datetime, datetime]:
        return self._bars[0].ts, self._bars[-1].ts

    def stream(self) -> Iterator[Bar]:
        yield from self._bars
