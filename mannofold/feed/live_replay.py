"""Paper feed: replays the same bars on a wall-clock cadence.

Yields identical ``Bar`` objects to :class:`HistoricalReplayFeed` — only the
timing differs — which is what makes backtest ≡ paper provable. ``speed`` lets a
demo replay e.g. a day of 15-min bars in seconds; the engine never reads the wall
clock, so output is unaffected by ``speed``.
"""

from __future__ import annotations

import time
from collections.abc import Iterator, Sequence

from mannofold.contracts.models import Bar, Mode


class LiveReplayFeed:
    mode = Mode.PAPER

    def __init__(self, bars: Sequence[Bar], speed: float = 0.0):
        """``speed`` = seconds of sleep between bars (0.0 = as fast as possible)."""
        self._bars = sorted(bars, key=lambda b: b.ts)
        self._speed = max(speed, 0.0)

    def stream(self) -> Iterator[Bar]:
        for bar in self._bars:
            yield bar
            if self._speed:
                time.sleep(self._speed)
