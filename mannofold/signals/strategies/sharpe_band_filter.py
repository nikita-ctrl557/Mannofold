"""Sharpe-band filter strategy: only trade when neighbourhood Sharpe is in a productive band.

Below the lower band the signal is indistinguishable from noise (flat).
Above the upper band the state is likely an unstable outlier (tapered weight).
Within the band: weight = tanh(gain * tanh(sharpe)) * confidence.
Flattens on anomalous regime or high anomaly score.
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

NAME = "sharpe_band_filter"
DESCRIPTION = (
    "Trade only when neighbourhood Sharpe falls in a productive band [0.15, 2.0]; "
    "noise below, tapered above; flat on anomaly."
)

# Tunable knobs
_GAIN = 2.5             # amplifier inside the outer tanh
_ANOMALY_THRESH = 0.6   # anomaly_score above this -> flat
_DEAD_BAND = 0.04       # collapse |weight| below this to 0
_SHARPE_LOW = 0.15      # below this -> noise, flat
_SHARPE_HIGH = 2.0      # above this -> unstable, taper
_TAPER_SCALE = 0.3      # multiplicative reduction in the taper region


class SharpeBandFilterStrategy:
    """Filter trades to productive Sharpe magnitude band on the manifold neighbourhood."""

    # ------------------------------------------------------------------ #
    #  signals()                                                          #
    # ------------------------------------------------------------------ #
    def signals(self, state: ManifoldState) -> SignalSet:
        sharpe = state.fwd_return_mean / (state.fwd_return_std + 1e-9)

        momentum = math.tanh(sharpe)

        confidence = state.regime_prob * (1.0 - state.anomaly_score)
        confidence = max(0.0, min(1.0, confidence))

        return SignalSet(
            ts=state.ts,
            symbol=state.symbol,
            momentum=momentum,
            expected_return=state.fwd_return_mean,
            anomaly=state.anomaly_score,
            regime_id=state.regime_id,
            confidence=confidence,
        )

    # ------------------------------------------------------------------ #
    #  target()                                                           #
    # ------------------------------------------------------------------ #
    def target(self, signals: SignalSet) -> TargetPosition:
        # Flat on anomalous regime or high anomaly score
        if signals.regime_id == ANOMALY_REGIME or signals.anomaly > _ANOMALY_THRESH:
            return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=0.0)

        # Recover Sharpe from tanh(sharpe) stored as momentum
        # signals.momentum = tanh(sharpe) => sharpe ~ atanh(clipped momentum)
        clipped = max(-0.9999, min(0.9999, signals.momentum))
        sharpe = math.atanh(clipped)
        abs_sharpe = abs(sharpe)

        # Below productive band -> noise, flat
        if abs_sharpe < _SHARPE_LOW:
            return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=0.0)

        # Compute raw weight from tanh-gain-tanh chain
        raw = math.tanh(_GAIN * signals.momentum) * signals.confidence

        # Above productive band -> taper (reduce size for unstable outlier states)
        if abs_sharpe > _SHARPE_HIGH:
            raw *= _TAPER_SCALE

        # Dead-band: suppress small, noisy weights
        if abs(raw) < _DEAD_BAND:
            raw = 0.0

        # Clamp to [-1, 1]
        raw = max(-1.0, min(1.0, raw))

        return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=raw)


def build() -> Strategy:
    """Return a fresh SharpeBandFilterStrategy instance."""
    return SharpeBandFilterStrategy()
