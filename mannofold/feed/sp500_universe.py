"""S&P-500 universe feed: ~100+ real US large-cap tickers, 5y daily (2013-2018).

Sourced from a GitHub-hosted per-ticker copy of the well-known Kaggle "sandp500"
dataset (the runtime network policy allows raw.githubusercontent.com). Each
ticker is a small plain CSV (date,open,high,low,close,volume,Name); files are
downloaded on first use and cached under data/cache/sp500/. Missing tickers
(404) are skipped gracefully so the universe degrades to whatever is reachable.

This is the breadth feed for portfolio-style ("hedge fund") backtests across
many names — distinct from feed/github_csv.py (a few curated index series).
"""

from __future__ import annotations

import csv
from datetime import UTC, datetime
from pathlib import Path

import httpx

from mannofold.contracts.models import Bar

# Pinned commit of a repo holding the full individual_stocks_5yr/ set.
_BASE = (
    "https://raw.githubusercontent.com/DayuanTan/prepare_data_4knn/"
    "ffa123f20aa6e5ac1535588f7aef752c9f2cbfa1/individual_stocks_5yr"
)
_CACHE = Path("data/cache/sp500")

# Curated large-cap universe spanning sectors (all in the 2013-2018 index set).
# 110+ names; the loader skips any that 404 so the effective universe is robust.
UNIVERSE: tuple[str, ...] = (
    # Tech / comms
    "AAPL", "MSFT", "AMZN", "GOOGL", "GOOG", "FB", "INTC", "CSCO", "ORCL", "IBM",
    "QCOM", "TXN", "ADBE", "CRM", "NFLX", "NVDA", "AMD", "MU", "HPQ", "ADI",
    "AMAT", "ADP", "INTU", "EBAY", "CTSH", "GLW", "WDC", "STX", "FISV",
    # Financials
    "JPM", "BAC", "WFC", "C", "GS", "MS", "AXP", "USB", "PNC", "COF", "MET",
    "AIG", "BK", "SCHW", "BLK", "SPGI", "CME", "TRV", "ALL", "PRU",
    # Healthcare
    "JNJ", "PFE", "MRK", "UNH", "ABT", "AMGN", "GILD", "BMY", "LLY", "MDT",
    "DHR", "CVS", "CI", "ANTM", "BAX", "BDX", "SYK", "ISRG", "BIIB", "CELG",
    # Consumer
    "WMT", "PG", "KO", "PEP", "HD", "MCD", "NKE", "DIS", "COST", "SBUX",
    "LOW", "TGT", "CL", "KMB", "MO", "PM", "MDLZ", "GIS", "K", "EL", "YUM",
    # Industrials / energy / materials
    "BA", "MMM", "GE", "CAT", "HON", "UPS", "UNP", "LMT", "RTN", "GD", "EMR",
    "FDX", "DE", "XOM", "CVX", "COP", "SLB", "EOG", "OXY", "HAL", "PSX", "VLO",
    "DD", "DOW", "LYB", "NEM", "FCX", "APD", "SHW",
    # Utilities / real estate / telecom / autos / travel
    "NEE", "DUK", "SO", "D", "EXC", "AEP", "VZ", "T", "SPG", "PLD", "AMT",
    "CCI", "F", "GM", "DAL", "AAL", "UAL", "LUV", "MAR", "CCL",
)


def _download(ticker: str) -> Path | None:
    _CACHE.mkdir(parents=True, exist_ok=True)
    cache = _CACHE / f"{ticker}.csv"
    if cache.exists() and cache.stat().st_size > 0:
        return cache
    try:
        r = httpx.get(f"{_BASE}/{ticker}_data.csv", timeout=20.0)
        if r.status_code != 200 or not r.text.startswith(("date", "Date")):
            return None
        cache.write_text(r.text)
        return cache
    except Exception:
        return None


def load_ticker_bars(ticker: str) -> list[Bar]:
    """Load (and cache) one ticker's daily bars; empty list if unavailable."""
    path = _download(ticker)
    if path is None:
        return []
    bars: list[Bar] = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            g = {k.lower(): v for k, v in row.items()}
            try:
                close = float(g["close"])
                bars.append(
                    Bar(
                        ts=datetime.fromisoformat(g["date"]).replace(tzinfo=UTC),
                        symbol=ticker,
                        open=float(g.get("open") or close),
                        high=float(g.get("high") or close),
                        low=float(g.get("low") or close),
                        close=close,
                        volume=float(g.get("volume") or 0.0),
                    )
                )
            except (ValueError, KeyError):
                continue
    bars.sort(key=lambda b: b.ts)
    return bars


def load_universe(tickers: tuple[str, ...] | list[str] | None = None) -> dict[str, list[Bar]]:
    """Load all reachable tickers in the universe -> {ticker: bars}."""
    out: dict[str, list[Bar]] = {}
    for t in tickers or UNIVERSE:
        bars = load_ticker_bars(t)
        if len(bars) > 250:  # need at least ~1y to be useful
            out[t] = bars
    return out
