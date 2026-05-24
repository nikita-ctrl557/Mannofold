"""High-volatility selective strategy.

Only engages in HIGH-volatility regimes where moves are large enough to trade
profitably. Tracks a per-symbol EMA of fwd_return_std; only takes a position
when current std exceeds that EMA (i.e. volatility is elevated vs its own
history). In quiet/low-vol states stays flat.

Direction is set by Sharpe: weight = tanh(GAIN * tanh(sharpe)) * confidence
gated by the high-vol filter.
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

NAME = "highvol_selective"
DESCRIPTION = (
    "Engages only when realised vol exceeds its own EMA; Sharpe-directional "
    "sizing with confidence gating and dead-band suppression."
)

# Tunable knobs
_GAIN = 3.0            # outer tanh amplifier
_ANOMALY_THRESH = 0.6  # anomaly_score above this -> flat immediately
_DEAD_BAND = 0.04      # collapse |weight| below this to 0
_EMA_ALPHA = 0.05      # smoothing factor for the per-symbol vol EMA
_VOL_RATIO = 1.0       # current_std must be > _VOL_RATIO * ema_std to engage


class HighVolSelectiveStrategy:
    """Only trades when current volatility is above its smoothed EMA."""

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

        # We need the raw fwd_return_std for the vol gate; it is embedded in the
        # Sharpe (momentum = mean/std) and expected_return (mean), so recover std:
        #   std = mean / sharpe  when sharpe != 0, else use a tiny sentinel
        er = signals.expected_return
        sharpe = signals.momentum  # already mean/std
        if abs(sharpe) > 1e-12:
            current_std = abs(er / sharpe)
        else:
            current_std = abs(er) + 1e-9

        # Update per-symbol EMA of vol
        if sym not in self._vol_ema:
            self._vol_ema[sym] = current_std
        else:
            self._vol_ema[sym] = (
                _EMA_ALPHA * current_std + (1.0 - _EMA_ALPHA) * self._vol_ema[sym]
            )

        ema_vol = self._vol_ema[sym]

        # High-vol gate: only trade when current vol > EMA vol
        if current_std <= _VOL_RATIO * ema_vol:
            return TargetPosition(ts=signals.ts, symbol=sym, target_weight=0.0)

        # Direction + size via Sharpe with confidence gating
        raw = math.tanh(_GAIN * math.tanh(sharpe)) * signals.confidence

        # Dead-band
        if abs(raw) < _DEAD_BAND:
            raw = 0.0

        # Clamp to [-1, 1]
        weight = max(-1.0, min(1.0, raw))
        return TargetPosition(ts=signals.ts, symbol=sym, target_weight=weight)


def build() -> Strategy:
    """Return a fresh HighVolSelectiveStrategy instance."""
    return HighVolSelectiveStrategy()
