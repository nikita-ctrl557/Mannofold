"""Balanced Kelly strategy: balanced_allrounder composite + Kelly inverse-variance sizing.

Composite = 0.4*momentum + 0.3*carry + 0.3*reversion (same as balanced_allrounder).
Direction = sign(composite); magnitude scaled by Kelly criterion.

target_weight = sign(composite) * tanh(gain * |composite| * |kelly|) * confidence
  where kelly = fwd_return_mean / (fwd_return_std^2 + eps), clamped to [-kelly_cap, kelly_cap]
  and   confidence = regime_prob * (1 - anomaly_score)

Density gate suppresses reversion in typical (high density) states.
Light EMA smoothing reduces turnover. Flat on ANOMALY_REGIME or anomaly > 0.6.
Dead-band: |weight| < 0.04 -> 0.
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

NAME = "balanced_kelly"
DESCRIPTION = (
    "Balanced allrounder composite (momentum+carry+density-gated reversion) with "
    "Kelly inverse-variance position sizing for improved risk-adjusted returns."
)

# Composite blend weights (must sum to 1.0)
_W_MOM   = 0.40
_W_CARRY = 0.30
_W_REV   = 0.30

# Gain applied to |composite| * |kelly| before outer tanh
_GAIN = 2.0

# Carry clamp before tanh
_CARRY_CLAMP = 5.0

# Density gate (logistic): gate=0.5 at _DENSITY_MID
_DENSITY_MID   = 1.0
_DENSITY_SCALE = 2.0
_DENSITY_CLAMP = 50.0

# Kelly parameters
_KELLY_CAP = 3.0

# EMA smoothing (higher alpha = more smoothing)
_EMA_ALPHA = 0.25

# Gates and thresholds
_ANOMALY_THRESH = 0.60
_DEAD_BAND      = 0.04
_EPS            = 1e-9


def _density_gate(density: float) -> float:
    """Smooth gate in [0,1]: low density -> ~0, high density -> ~1."""
    d = max(0.0, min(density, _DENSITY_CLAMP))
    return 1.0 / (1.0 + math.exp(-_DENSITY_SCALE * (d - _DENSITY_MID)))


class BalancedKellyStrategy:
    """Composite momentum+carry+reversion with Kelly inverse-variance sizing."""

    def __init__(self) -> None:
        self._ema: dict[str, float] = {}

    def signals(self, state: ManifoldState) -> SignalSet:
        mu  = state.fwd_return_mean
        sig = state.fwd_return_std
        sharpe = mu / (sig + _EPS)

        # Momentum: tanh of Sharpe
        momentum = math.tanh(sharpe)

        # Carry: risk-adjusted return clamped then tanh
        carry_raw = mu / (sig ** 2 + _EPS)
        carry = math.tanh(max(-_CARRY_CLAMP, min(_CARRY_CLAMP, carry_raw)))

        # Reversion: counter-trend weighted by LOW density (atypical states only)
        dgate = _density_gate(state.density)
        reversion = -math.tanh(sharpe) * (1.0 - dgate)

        composite = _W_MOM * momentum + _W_CARRY * carry + _W_REV * reversion

        # Kelly ratio: mu / sigma^2, clamped
        kelly_raw = mu / (sig ** 2 + _EPS)
        kelly = max(-_KELLY_CAP, min(_KELLY_CAP, kelly_raw))

        # EMA smoothing of composite per symbol
        sym = state.symbol
        prev_ema = self._ema.get(sym, composite)
        smoothed = _EMA_ALPHA * composite + (1.0 - _EMA_ALPHA) * prev_ema
        self._ema[sym] = smoothed

        # Confidence: regime certainty * non-anomalousness
        confidence = max(0.0, min(1.0, state.regime_prob * (1.0 - state.anomaly_score)))

        # Pack kelly into expected_return field for use in target()
        return SignalSet(
            ts=state.ts,
            symbol=sym,
            momentum=smoothed,
            expected_return=kelly,
            anomaly=state.anomaly_score,
            regime_id=state.regime_id,
            confidence=confidence,
        )

    def target(self, signals: SignalSet) -> TargetPosition:
        sym = signals.symbol

        # Hard gates: anomalous regime or high anomaly score -> flat
        if signals.regime_id == ANOMALY_REGIME or signals.anomaly > _ANOMALY_THRESH:
            return TargetPosition(ts=signals.ts, symbol=sym, target_weight=0.0)

        composite = signals.momentum   # EMA-smoothed composite
        kelly     = signals.expected_return  # clamped Kelly ratio

        comp_sign = math.copysign(1.0, composite) if composite != 0.0 else 0.0
        magnitude = abs(composite) * abs(kelly)

        # target = sign(composite) * tanh(gain * magnitude) * confidence
        raw = comp_sign * math.tanh(_GAIN * magnitude) * signals.confidence

        # Dead-band
        if abs(raw) < _DEAD_BAND:
            raw = 0.0

        weight = max(-1.0, min(1.0, raw))
        return TargetPosition(ts=signals.ts, symbol=sym, target_weight=weight)


def build() -> Strategy:
    """Return a fresh BalancedKellyStrategy instance."""
    return BalancedKellyStrategy()
