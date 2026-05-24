"""Volatility-band filter strategy.

Only trades when fwd_return_std sits within a comfortable moderate band relative
to its per-symbol EMA — avoiding both dead-calm (no opportunity) and chaotic
(unpredictable) regimes.  This keeps monthly returns more consistent.

Band: [LOW_RATIO * ema_std, HIGH_RATIO * ema_std]  (default 0.6x .. 1.4x)
Inside the band  -> take the trend position.
Outside the band -> flat (either too quiet or too wild).

weight = tanh(GAIN * tanh(sharpe)) * band_gate * confidence
confidence = regime_prob * (1 - anomaly_score)
"""

from __future__ import annotations

import math

from mannofold.contracts.interfaces import Strategy
from mannofold.contracts.models import (
    ANOMALY_REGIME,
    ManifoldState,
    SignalSet,
    TargetPosition,
)

NAME = "vol_band_filter"
DESCRIPTION = (
    "Trades only when realised vol is within a moderate band around its EMA "
    "(0.6x–1.4x), avoiding both dead-calm and chaotic regimes; Sharpe-directional "
    "sizing with confidence gating and dead-band suppression for month-over-month "
    "consistency."
)

# Tunable knobs
_GAIN = 3.0             # outer tanh amplifier
_ANOMALY_THRESH = 0.6   # anomaly_score above this -> flat immediately
_DEAD_BAND = 0.04       # collapse |weight| below this to 0
_EMA_ALPHA = 0.05       # smoothing factor for the per-symbol vol EMA
_LOW_RATIO = 0.6        # lower bound: current_std must be >= LOW_RATIO * ema_std
_HIGH_RATIO = 1.4       # upper bound: current_std must be <= HIGH_RATIO * ema_std


class VolBandFilterStrategy:
    """Only trades when current volatility is within a moderate band of its EMA."""

    def __init__(self) -> None:
        # per-symbol EMA of fwd_return_std (initialised lazily on first bar)
        self._vol_ema: dict[str, float] = {}

    # ------------------------------------------------------------------ #
    #  signals()                                                           #
    # ------------------------------------------------------------------ #
    def signals(self, state: ManifoldState) -> SignalSet:
        confidence = state.regime_prob * (1.0 - state.anomaly_score)
        confidence = max(0.0, min(1.0, confidence))

        sharpe = state.fwd_return_mean / (state.fwd_return_std + 1e-9)

        return SignalSet(
            ts=state.ts,
            symbol=state.symbol,
            momentum=sharpe,
            expected_return=state.fwd_return_mean,
            anomaly=state.anomaly_score,
            regime_id=state.regime_id,
            confidence=confidence,
        )

    # ------------------------------------------------------------------ #
    #  target()                                                            #
    # ------------------------------------------------------------------ #
    def target(self, signals: SignalSet) -> TargetPosition:
        sym = signals.symbol

        # Flat unconditionally on anomalous regime or high anomaly score
        if signals.regime_id == ANOMALY_REGIME or signals.anomaly > _ANOMALY_THRESH:
            return TargetPosition(ts=signals.ts, symbol=sym, target_weight=0.0)

        # Recover current_std from sharpe = mean/std
        er = signals.expected_return
        sharpe = signals.momentum
        if abs(sharpe) > 1e-12:
            current_std = abs(er / sharpe)
        else:
            current_std = abs(er) + 1e-9

        # Update per-symbol EMA of vol (no lookahead — update before gating)
        if sym not in self._vol_ema:
            self._vol_ema[sym] = current_std
        else:
            self._vol_ema[sym] = (
                _EMA_ALPHA * current_std + (1.0 - _EMA_ALPHA) * self._vol_ema[sym]
            )

        ema_vol = self._vol_ema[sym]

        # Band gate: only trade when vol is within [LOW * ema, HIGH * ema]
        lo = _LOW_RATIO * ema_vol
        hi = _HIGH_RATIO * ema_vol
        band_gate = 1.0 if lo <= current_std <= hi else 0.0

        if band_gate == 0.0:
            return TargetPosition(ts=signals.ts, symbol=sym, target_weight=0.0)

        # Direction + size via Sharpe with confidence gating
        raw = math.tanh(_GAIN * math.tanh(sharpe)) * band_gate * signals.confidence

        # Dead-band
        if abs(raw) < _DEAD_BAND:
            raw = 0.0

        # Clamp to [-1, 1]
        weight = max(-1.0, min(1.0, raw))
        return TargetPosition(ts=signals.ts, symbol=sym, target_weight=weight)


def build() -> Strategy:
    """Return a fresh VolBandFilterStrategy instance."""
    return VolBandFilterStrategy()
