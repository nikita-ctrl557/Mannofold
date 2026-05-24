"""Dispersion-breakout strategy: captures volatility-expansion / crisis onset.

Keeps a per-symbol EMA of neighbourhood dispersion (fwd_return_std). A breakout
occurs when current dispersion >> EMA (ratio > BREAKOUT_RATIO). On breakout,
takes a full-conviction position in the drift direction; when dispersion is
normal or contracting, takes only a small position. Gated by regime_prob.
Flat on ANOMALY_REGIME.
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

NAME = "dispersion_breakout"
DESCRIPTION = (
    "Per-symbol EMA dispersion tracker: full conviction when std >> EMA "
    "(volatility expansion / crisis onset), small when normal/contracting."
)

# Tunable knobs
_EMA_ALPHA = 0.08       # smoothing factor for per-symbol EMA of fwd_return_std
_BREAKOUT_RATIO = 1.3   # current_std / ema_std must exceed this for full conviction
_GAIN = 3.0             # tanh amplifier: tanh(gain * sharpe)
_FULL_SCALE = 1.0       # conviction multiplier on breakout
_QUIET_SCALE = 0.12     # conviction multiplier when no breakout
_DEAD_BAND = 0.04       # collapse |weight| below this to 0


class DispersionBreakoutStrategy:
    """Directional strategy sized by dispersion-breakout conviction."""

    def __init__(self) -> None:
        # per-symbol EMA of fwd_return_std and dispersion ratio
        self._ema_std: dict[str, float] = {}
        self._disp_ratio: dict[str, float] = {}

    # ------------------------------------------------------------------ #
    #  signals()                                                           #
    # ------------------------------------------------------------------ #
    def signals(self, state: ManifoldState) -> SignalSet:
        sym = state.symbol
        current_std = max(state.fwd_return_std, 1e-9)

        # Update per-symbol EMA of dispersion
        if sym not in self._ema_std:
            self._ema_std[sym] = current_std
            self._disp_ratio[sym] = 1.0
        else:
            prev_ema = self._ema_std[sym]
            self._ema_std[sym] = _EMA_ALPHA * current_std + (1.0 - _EMA_ALPHA) * prev_ema
            self._disp_ratio[sym] = current_std / (self._ema_std[sym] + 1e-9)

        # Sharpe of the neighbourhood
        sharpe = state.fwd_return_mean / (current_std + 1e-9)

        # Confidence gated by regime_prob only
        confidence = max(0.0, min(1.0, state.regime_prob))

        # Store dispersion ratio in momentum field (alongside sharpe via product)
        # Use a separate encoding: pass sharpe via momentum, ratio via confidence
        # We use a trick: encode disp_ratio as extra in the confidence slot
        # by multiplying: effective_confidence = regime_prob; disp in momentum side
        return SignalSet(
            ts=state.ts,
            symbol=sym,
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

        # Flat unconditionally on anomalous regime
        if signals.regime_id == ANOMALY_REGIME:
            return TargetPosition(ts=signals.ts, symbol=sym, target_weight=0.0)

        # Retrieve the current dispersion ratio for this symbol
        disp_ratio = self._disp_ratio.get(sym, 1.0)
        is_breakout = disp_ratio > _BREAKOUT_RATIO

        scale = _FULL_SCALE if is_breakout else _QUIET_SCALE

        # Directional conviction: tanh(gain * sharpe) * confidence * scale
        direction = math.tanh(_GAIN * signals.momentum)
        weight = direction * signals.confidence * scale

        # Dead-band
        if abs(weight) < _DEAD_BAND:
            weight = 0.0

        # Clamp to [-1, 1]
        weight = max(-1.0, min(1.0, weight))

        return TargetPosition(ts=signals.ts, symbol=sym, target_weight=weight)


def build() -> Strategy:
    """Return a fresh DispersionBreakoutStrategy instance."""
    return DispersionBreakoutStrategy()
