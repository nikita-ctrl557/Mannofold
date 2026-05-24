"""Data feeds — the only components that drive the clock forward.

``HistoricalReplayFeed`` (backtest) and ``LiveReplayFeed`` (paper) yield the
exact same ``Bar`` objects; only the cadence differs. ``AlpacaFeed`` (added by
the live-data workstream) plugs in here behind the same Protocol.
"""

from mannofold.feed.alpaca import AlpacaFeed
from mannofold.feed.github_csv import DATASETS, GithubCsvFeed, load_bars
from mannofold.feed.historical import HistoricalReplayFeed
from mannofold.feed.live_replay import LiveReplayFeed
from mannofold.feed.synthetic import SyntheticConfig, generate_bars

__all__ = [
    "AlpacaFeed",
    "DATASETS",
    "GithubCsvFeed",
    "HistoricalReplayFeed",
    "LiveReplayFeed",
    "SyntheticConfig",
    "generate_bars",
    "load_bars",
]
