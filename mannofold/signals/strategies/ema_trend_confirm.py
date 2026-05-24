"""EMA Trend Confirmation strategy: agree between instant drift and its EMA.

Only takes a position when the INSTANT drift (fwd_return_mean) sign AGREES with
the EMA of that drift (past-only, per-symbol). Sizes by agreement strength —
product of both drift magnitudes passed through tanh. Flat when they disagree.
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

NAME = "ema_trend_confirm"
DESCRIPTION = (
    "Trend-confirmation: take positions only when instant drift agrees with its "
    "per-symbol EMA; size by agreement strength scaled by regime confidence."
)

# Tunable knobs
_EMA_ALPHA = 0.15        # smoothing factor for drift EMA (higher = faster)
_GAIN = 2.5              # amplifier inside inner tanh(gain * sharpe)
_ANOMALY_THRESH = 0.6    # anomaly_score above this -> flat
_DEAD_BAND = 0.04        # collapse |weight| below this to 0


class EmaTrendConfirmStrategy:
    """EMA trend-confirmation strategy with per-symbol drift EMA state."""

    def __init__(self) -> None:
        # per-symbol EMA of fwd_return_mean; None until first observation
        self._ema: dict[str, float] = {}

    # ------------------------------------------------------------------ #
    #  signals()                                                          #
    # ------------------------------------------------------------------ #
    def signals(self, state: ManifoldState) -> SignalSet:
        sym = state.symbol
        drift = state.fwd_return_mean

        # Update past-only EMA (uses previous EMA, not current drift yet)
        if sym not in self._ema:
            # Initialise to current drift; will be updated next call
            self._ema[sym] = drift
            ema_drift = drift
        else:
            prev_ema = self._ema[sym]
            ema_drift = prev_ema  # the EMA *before* this bar (no lookahead)
            # Update EMA for future bars
            self._ema[sym] = _EMA_ALPHA * drift + (1.0 - _EMA_ALPHA) * prev_ema

        # Confidence gate
        confidence = state.regime_prob * (1.0 - state.anomaly_score)
        confidence = max(0.0, min(1.0, confidence))

        # Agreement: both instant drift and EMA drift have same sign
        agree = 1.0 if drift * ema_drift > 0.0 else 0.0

        # Agreement strength: product of magnitudes (normalised via tanh of Sharpe)
        sharpe = drift / (state.fwd_return_std + 1e-9)
        ema_sharpe = ema_drift / (state.fwd_return_std + 1e-9)
        agree_strength = agree * abs(math.tanh(_GAIN * sharpe)) * abs(math.tanh(_GAIN * ema_sharpe))

        # momentum slot: signed weight hint — sign from instant drift
        direction = math.copysign(1.0, drift) if drift != 0.0 else 0.0
        momentum = direction * agree_strength

        return SignalSet(
            ts=state.ts,
            symbol=sym,
            momentum=momentum,
            expected_return=drift,
            anomaly=state.anomaly_score,
            regime_id=state.regime_id,
            confidence=confidence,
        )

    # ------------------------------------------------------------------ #
    #  target()                                                           #
    # ------------------------------------------------------------------ #
    def target(self, signals: SignalSet) -> TargetPosition:
        sym = signals.symbol

        # Flat on anomalous regime or high anomaly score
        if signals.regime_id == ANOMALY_REGIME or signals.anomaly > _ANOMALY_THRESH:
            return TargetPosition(ts=signals.ts, symbol=sym, target_weight=0.0)

        # weight = sign(drift) * agree_strength * confidence (already encoded in momentum)
        raw = signals.momentum * signals.confidence

        # Dead-band: suppress small noisy weights
        if abs(raw) < _DEAD_BAND:
            raw = 0.0

        # Clamp to [-1, 1]
        raw = max(-1.0, min(1.0, raw))

        return TargetPosition(ts=signals.ts, symbol=sym, target_weight=raw)


def build() -> Strategy:
    """Return a fresh EmaTrendConfirmStrategy instance."""
    return EmaTrendConfirmStrategy()
