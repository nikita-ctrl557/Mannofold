"""Momentum-of-Confidence strategy: trade the RATE OF CHANGE of regime conviction.

A regime that is *solidifying* (confidence rising) is a stronger signal than a
regime that has already fully committed.  We keep a per-symbol EMA of
``regime_prob`` and scale position size by how fast confidence is climbing.
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

NAME = "momentum_of_confidence"
DESCRIPTION = (
    "Scales directional conviction by the rate-of-change of regime confidence: "
    "a solidifying regime adds risk; a dissolving regime trims toward flat."
)

# ── tunable knobs ──────────────────────────────────────────────────────────────
_EMA_ALPHA    = 0.15   # smoothing for the regime_prob EMA (shorter = more reactive)
_GAIN         = 3.0    # amplifier inside tanh(gain * sharpe)
_K            = 2.0    # scaling for the delta term in the confidence multiplier
_BASE         = 0.3    # base confidence multiplier (floor before delta kicks in)
_ANOMALY_FLAT = 0.6    # anomaly_score above this -> flat immediately
_DEAD_BAND    = 0.04   # collapse |weight| below this to 0
# ──────────────────────────────────────────────────────────────────────────────


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


class MomentumOfConfidenceStrategy:
    """Directional sizing amplified by the momentum of regime confidence."""

    def __init__(self) -> None:
        # per-symbol EMA of regime_prob
        self._ema: dict[str, float] = {}

    # ------------------------------------------------------------------ #
    #  signals()                                                          #
    # ------------------------------------------------------------------ #
    def signals(self, state: ManifoldState) -> SignalSet:
        sym = state.symbol
        prob = state.regime_prob

        # initialise EMA on first observation
        prev_ema = self._ema.get(sym, prob)
        new_ema  = _EMA_ALPHA * prob + (1.0 - _EMA_ALPHA) * prev_ema
        self._ema[sym] = new_ema

        # delta: positive when confidence is RISING (regime solidifying)
        delta = prob - new_ema

        # Sharpe-normalised expected drift — the directional signal
        sharpe = state.fwd_return_mean / (state.fwd_return_std + 1e-9)

        # Pack both ingredients into SignalSet fields
        # momentum  <- sharpe (direction + magnitude)
        # confidence <- delta  (rate-of-change of conviction)
        return SignalSet(
            ts=state.ts,
            symbol=sym,
            momentum=sharpe,
            expected_return=state.fwd_return_mean,
            anomaly=state.anomaly_score,
            regime_id=state.regime_id,
            confidence=delta,          # repurposed: delta in regime_prob
        )

    # ------------------------------------------------------------------ #
    #  target()                                                           #
    # ------------------------------------------------------------------ #
    def target(self, signals: SignalSet) -> TargetPosition:
        sym = signals.symbol

        # Flat unconditionally on anomalous regime or high anomaly score
        if signals.regime_id == ANOMALY_REGIME or signals.anomaly > _ANOMALY_FLAT:
            return TargetPosition(ts=signals.ts, symbol=sym, target_weight=0.0)

        # Recover the stored EMA to compute the level term
        ema_level = self._ema.get(sym, 0.0)
        delta     = signals.confidence   # delta stored in confidence field

        # Confidence multiplier: rising delta → scale up; also weighted by level
        conf_mult = _clamp(_BASE + _K * delta, 0.0, 1.0)

        # Reduce anomaly exposure continuously
        anomaly_scale = 1.0 - signals.anomaly

        # Core weight: tanh of Sharpe-amplified drift, modulated by confidence momentum
        raw = (
            math.tanh(_GAIN * math.tanh(signals.momentum))
            * conf_mult
            * anomaly_scale
        )

        # Dead-band: treat tiny weights as zero
        if abs(raw) < _DEAD_BAND:
            raw = 0.0

        return TargetPosition(ts=signals.ts, symbol=sym, target_weight=raw)


def build() -> Strategy:
    """Return a fresh MomentumOfConfidenceStrategy instance."""
    return MomentumOfConfidenceStrategy()
