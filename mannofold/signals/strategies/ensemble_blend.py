"""Ensemble-blend strategy: weighted combination of drift, momentum, and carry sub-signals.

Blends three complementary views of the manifold state:
  drift   — direction of expected return (Sharpe-like),
  momentum — velocity-scaled version of drift direction,
  carry   — risk-adjusted expected return (carry-like ratio).
Composite weight = tanh(gain * blend) * confidence, gated on anomaly/regime.
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

NAME = "ensemble_blend"
DESCRIPTION = "Weighted blend of drift, momentum, and carry sub-signals from manifold geometry."

# Sub-signal blend weights (must sum to 1.0)
_W_DRIFT  = 0.4
_W_MOM    = 0.3
_W_CARRY  = 0.3

# Gain applied to composite before outer tanh
_GAIN = 2.5

# Momentum velocity gain (separate scale on drift sign * velocity magnitude)
_MOM_GAIN = 1.5

# Carry clamp before tanh (prevents carry from dominating on tiny std)
_CARRY_CLAMP = 5.0

_EPS = 1e-9

_ANOMALY_THRESH = 0.6   # anomaly_score above this -> flat
_DEAD_BAND      = 0.04  # |weight| below this -> 0


class EnsembleBlendStrategy:
    """Ensemble blend of drift, momentum-velocity, and carry sub-signals."""

    # ------------------------------------------------------------------ #
    #  signals()                                                          #
    # ------------------------------------------------------------------ #
    def signals(self, state: ManifoldState) -> SignalSet:
        mu  = state.fwd_return_mean
        sig = state.fwd_return_std

        # (a) Drift: Sharpe-like direction
        drift = math.tanh(mu / (sig + _EPS))

        # (b) Momentum: sign of drift amplified by a separate velocity gain
        #     Uses the same Sharpe proxy but with a different gain to give a
        #     distinct contribution (trajectory-momentum flavour).
        momentum = math.tanh(_MOM_GAIN * mu / (sig + _EPS))

        # (c) Carry: risk-adjusted return ~ mean / var, clamped then tanh
        carry_raw = mu / (sig ** 2 + _EPS)
        carry_clamped = max(-_CARRY_CLAMP, min(_CARRY_CLAMP, carry_raw))
        carry = math.tanh(carry_clamped)

        # Composite sub-signal
        composite = _W_DRIFT * drift + _W_MOM * momentum + _W_CARRY * carry

        # Confidence = regime certainty * non-anomalousness, clamped [0,1]
        confidence = max(0.0, min(1.0, state.regime_prob * (1.0 - state.anomaly_score)))

        return SignalSet(
            ts=state.ts,
            symbol=state.symbol,
            momentum=composite,          # reuse momentum field for composite
            expected_return=mu,
            anomaly=state.anomaly_score,
            regime_id=state.regime_id,
            confidence=confidence,
        )

    # ------------------------------------------------------------------ #
    #  target()                                                           #
    # ------------------------------------------------------------------ #
    def target(self, signals: SignalSet) -> TargetPosition:
        # Hard gates: anomalous regime or high anomaly score -> flat
        if signals.regime_id == ANOMALY_REGIME or signals.anomaly > _ANOMALY_THRESH:
            return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=0.0)

        # composite is stored in signals.momentum
        raw = math.tanh(_GAIN * signals.momentum) * signals.confidence

        # Dead-band: suppress small noisy weights
        if abs(raw) < _DEAD_BAND:
            raw = 0.0

        # Clamp to [-1, 1]
        raw = max(-1.0, min(1.0, raw))

        return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=raw)


def build() -> Strategy:
    """Return a fresh EnsembleBlendStrategy instance."""
    return EnsembleBlendStrategy()
