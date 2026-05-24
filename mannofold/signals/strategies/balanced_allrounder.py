"""Balanced allrounder strategy: robust kitchen-sink composite for all-weather use.

Composite = 0.4*momentum + 0.3*carry + 0.3*reversion, where:
  momentum  = tanh(sharpe)                       directional drift
  carry     = tanh(clamp(mu / (sig^2 + eps)))    risk-adjusted expected return
  reversion = -tanh(sharpe) * low_density_weight reversion only in atypical states

A density typicality gate suppresses sizing in typical dense regions for the
reversion term, entry/exit hysteresis cuts churn, and light EMA smoothing
prevents rapid oscillation. Designed for steady, all-weather behaviour.
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

NAME = "balanced_allrounder"
DESCRIPTION = (
    "Tuned kitchen-sink composite of momentum, carry, and density-gated reversion "
    "with hysteresis and EMA smoothing for steady all-weather behaviour."
)

# Sub-signal blend weights (must sum to 1.0)
_W_MOM  = 0.40
_W_CARRY = 0.30
_W_REV  = 0.30

# Gain applied to composite before outer tanh
_GAIN = 2.8

# Carry clamp before tanh
_CARRY_CLAMP = 5.0

# Density gate parameters (logistic): gate=0.5 at _DENSITY_MID
_DENSITY_MID   = 1.0
_DENSITY_SCALE = 2.0
_DENSITY_CLAMP = 50.0

# EMA smoothing coefficient (higher = more smoothing)
_EMA_ALPHA = 0.25

# Hysteresis thresholds
_ENTRY_THRESH = 0.10
_EXIT_THRESH  = 0.05

_ANOMALY_THRESH = 0.60
_DEAD_BAND      = 0.04
_EPS            = 1e-9


def _density_gate(density: float) -> float:
    """Smooth gate in [0,1]: low density -> ~0, high density -> ~1."""
    d = max(0.0, min(density, _DENSITY_CLAMP))
    return 1.0 / (1.0 + math.exp(-_DENSITY_SCALE * (d - _DENSITY_MID)))


class BalancedAllrounderStrategy:
    """Composite momentum + carry + density-gated reversion with EMA and hysteresis."""

    def __init__(self) -> None:
        # per-symbol EMA of composite signal
        self._ema: dict[str, float] = {}
        # per-symbol hysteresis stance (+1, -1, 0)
        self._stance: dict[str, int] = {}

    def signals(self, state: ManifoldState) -> SignalSet:
        mu  = state.fwd_return_mean
        sig = state.fwd_return_std
        sharpe = mu / (sig + _EPS)

        # Momentum: tanh of Sharpe
        momentum = math.tanh(sharpe)

        # Carry: risk-adjusted return ~ mean / var, clamped then tanh
        carry_raw = mu / (sig ** 2 + _EPS)
        carry = math.tanh(max(-_CARRY_CLAMP, min(_CARRY_CLAMP, carry_raw)))

        # Reversion: counter-trend, weighted by LOW density (atypical states)
        dgate = _density_gate(state.density)
        low_density_weight = 1.0 - dgate  # 1 when atypical, 0 when typical
        reversion = -math.tanh(sharpe) * low_density_weight

        composite = _W_MOM * momentum + _W_CARRY * carry + _W_REV * reversion

        # EMA smoothing per symbol
        sym = state.symbol
        prev_ema = self._ema.get(sym, composite)
        smoothed = _EMA_ALPHA * composite + (1.0 - _EMA_ALPHA) * prev_ema
        self._ema[sym] = smoothed

        # Confidence: regime certainty * non-anomalousness
        confidence = max(0.0, min(1.0, state.regime_prob * (1.0 - state.anomaly_score)))

        return SignalSet(
            ts=state.ts,
            symbol=sym,
            momentum=smoothed,          # composite (EMA-smoothed) stored here
            expected_return=mu,
            anomaly=state.anomaly_score,
            regime_id=state.regime_id,
            confidence=confidence,
        )

    def target(self, signals: SignalSet) -> TargetPosition:
        sym = signals.symbol

        # Hard gates: anomalous regime or high anomaly score -> flat
        if signals.regime_id == ANOMALY_REGIME or signals.anomaly > _ANOMALY_THRESH:
            self._stance[sym] = 0
            return TargetPosition(ts=signals.ts, symbol=sym, target_weight=0.0)

        # Density gate on the full composite (from signals.confidence already
        # encodes regime_prob * (1-anomaly); multiply by tanh(gain*composite))
        raw = math.tanh(_GAIN * signals.momentum) * signals.confidence

        # Dead-band
        if abs(raw) < _DEAD_BAND:
            raw = 0.0

        # Hysteresis
        current_stance = self._stance.get(sym, 0)
        desired_sign = 1 if raw > 0 else (-1 if raw < 0 else 0)

        if current_stance == 0:
            if abs(raw) >= _ENTRY_THRESH:
                self._stance[sym] = desired_sign
            else:
                return TargetPosition(ts=signals.ts, symbol=sym, target_weight=0.0)
        else:
            if desired_sign != 0 and desired_sign != current_stance and abs(raw) >= _ENTRY_THRESH:
                self._stance[sym] = desired_sign
            elif abs(raw) < _EXIT_THRESH:
                self._stance[sym] = 0
                return TargetPosition(ts=signals.ts, symbol=sym, target_weight=0.0)

        weight = abs(raw) * self._stance[sym]
        weight = max(-1.0, min(1.0, weight))

        return TargetPosition(ts=signals.ts, symbol=sym, target_weight=weight)


def build() -> Strategy:
    """Return a fresh BalancedAllrounderStrategy instance."""
    return BalancedAllrounderStrategy()
