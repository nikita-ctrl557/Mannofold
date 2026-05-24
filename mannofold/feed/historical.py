"""Backtest feed: replays a fixed list of bars as fast as possible."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from mannofold.contracts.models import Bar, Mode


class HistoricalReplayFeed:
    """Yields bars in time order with no delay. ``mode == BACKTEST``."""

    mode = Mode.BACKTEST

    def __init__(self, bars: Sequence[Bar]):
        self._bars = sorted(bars, key=lambda b: b.ts)

    def stream(self) -> Iterator[Bar]:
        yield from self._bars
