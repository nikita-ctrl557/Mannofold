"""Regime-switching synthetic market generator.

A 3-state Markov-switching geometric Brownian motion:

* state 0 — low-vol trending (positive drift, low vol)
* state 1 — high-vol mean-reverting (zero drift, high vol)
* state 2 — crash (strong negative drift, very high vol, rare/short)

The known ground-truth regime labels let the regime-recovery test check that the
manifold pipeline actually rediscovers them. This makes the entire stack runnable
in an ephemeral container with zero external data or secrets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import numpy as np

from mannofold.contracts.models import Bar


@dataclass
class SyntheticConfig:
    symbol: str = "SYNTH"
    n_bars: int = 4000
    seed: int = 7
    start: datetime = field(default_factory=lambda: datetime(2023, 1, 1, tzinfo=UTC))
    bar_minutes: int = 15
    start_price: float = 100.0
    # (drift_per_bar, vol_per_bar) for each state
    drifts: tuple[float, float, float] = (0.00025, 0.0, -0.0020)
    vols: tuple[float, float, float] = (0.004, 0.011, 0.025)
    # Markov transition matrix rows must sum to 1; crash is rare and short-lived.
    transition: tuple[tuple[float, ...], ...] = (
        (0.990, 0.009, 0.001),
        (0.020, 0.978, 0.002),
        (0.060, 0.140, 0.800),
    )


def generate_bars(cfg: SyntheticConfig | None = None) -> tuple[list[Bar], list[int]]:
    """Return ``(bars, regime_labels)`` where ``regime_labels[i]`` is the true state."""
    cfg = cfg or SyntheticConfig()
    rng = np.random.default_rng(cfg.seed)
    trans = np.array(cfg.transition)

    state = 0
    price = cfg.start_price
    bars: list[Bar] = []
    labels: list[int] = []
    ts = cfg.start

    for _ in range(cfg.n_bars):
        drift = cfg.drifts[state]
        vol = cfg.vols[state]
        ret = drift + vol * rng.standard_normal()
        new_price = max(price * float(np.exp(ret)), 0.01)

        open_ = price
        close = new_price
        intrabar = abs(vol) * price * 0.5
        high = max(open_, close) + abs(rng.standard_normal()) * intrabar
        low = min(open_, close) - abs(rng.standard_normal()) * intrabar
        low = max(low, 0.01)
        base_vol = 1000.0 * (1.0 + 4.0 * vol / cfg.vols[0])
        volume = float(max(base_vol * (1.0 + 0.3 * rng.standard_normal()), 1.0))

        bars.append(
            Bar(
                ts=ts,
                symbol=cfg.symbol,
                open=round(open_, 4),
                high=round(high, 4),
                low=round(low, 4),
                close=round(close, 4),
                volume=round(volume, 2),
            )
        )
        labels.append(state)

        price = new_price
        ts = ts + timedelta(minutes=cfg.bar_minutes)
        state = int(rng.choice(3, p=trans[state]))

    return bars, labels
