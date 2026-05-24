"""Relativistic momentum strategy.

Special-relativity framing: classical momentum p = m*v grows unboundedly, but
relativistic momentum p = m*v*gamma(v) saturates as velocity v -> c (speed limit).

Here we set the natural speed limit c = 1 and map the Sharpe ratio through tanh
so that signal velocity v = tanh(sharpe) is already in [-1, 1].

Relativistic momentum: p = v / sqrt(1 - 0.98*v^2)

As |v| -> 1, the denominator -> sqrt(0.02) ~ 0.14, so p saturates near ~7*v
rather than blowing up -- strong conviction, but bounded.  We then squash through
tanh(gain * p) and temper by confidence, giving a weight that resists over-leveraging
on extreme signals while still following strong trends.

target_weight = tanh(gain * p_relativistic) * confidence
confidence    = regime_prob * (1 - anomaly_score)
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

NAME = "relativistic_momentum"
DESCRIPTION = (
    "Special-relativity momentum strategy: signal velocity v = tanh(sharpe) "
    "is relativistically boosted via p = v/sqrt(1-0.98*v^2), saturating near the "
    "speed limit c=1.  Conviction grows strongly but can never exceed the light "
    "cone, preventing over-leveraging on extreme signals. "
    "target_weight = tanh(gain*p) * confidence."
)

_EPS = 1e-9

# Lorentz factor denominator coefficient: 1 - beta*v^2.
# beta=0.98 keeps the denominator from hitting zero at |v|=1 while still
# giving strong relativistic amplification.
_BETA = 0.98

# Outer gain applied before the final tanh squash.
_GAIN = 1.8

# Anomaly threshold above which we go flat.
_ANOMALY_GATE = 0.6

# Dead-band: suppress |w| < this to 0.
_DEADBAND = 0.04


def _relativistic_momentum(v: float, beta: float = _BETA) -> float:
    """Compute p = v / sqrt(1 - beta*v^2).

    v should be in (-1, 1).  beta < 1 ensures the denominator never reaches 0.
    """
    denom = math.sqrt(max(1.0 - beta * v * v, _EPS))
    return v / denom


class RelativisticMomentumStrategy:
    """Momentum follower whose conviction saturates relativistically near the speed limit."""

    def __init__(
        self,
        gain: float = _GAIN,
        beta: float = _BETA,
        anomaly_gate: float = _ANOMALY_GATE,
        deadband: float = _DEADBAND,
    ) -> None:
        self._gain = gain
        self._beta = beta
        self._anomaly_gate = anomaly_gate
        self._deadband = deadband

    # ------------------------------------------------------------------ #
    #  signals()                                                           #
    # ------------------------------------------------------------------ #
    def signals(self, state: ManifoldState) -> SignalSet:
        sharpe = state.fwd_return_mean / (state.fwd_return_std + _EPS)

        # Signal velocity in natural units c=1; tanh maps sharpe to (-1, 1).
        v = math.tanh(sharpe)

        # Relativistic boost: p saturates as |v| -> 1.
        p = _relativistic_momentum(v, self._beta)

        confidence = max(0.0, state.regime_prob * (1.0 - state.anomaly_score))

        return SignalSet(
            ts=state.ts,
            symbol=state.symbol,
            momentum=p,           # store boosted momentum as the signal
            expected_return=state.fwd_return_mean,
            anomaly=state.anomaly_score,
            regime_id=state.regime_id,
            confidence=confidence,
        )

    # ------------------------------------------------------------------ #
    #  target()                                                            #
    # ------------------------------------------------------------------ #
    def target(self, signals: SignalSet) -> TargetPosition:
        # Hard-off: anomalous regime or excessive anomaly score.
        if signals.regime_id == ANOMALY_REGIME or signals.anomaly > self._anomaly_gate:
            return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=0.0)

        # Squash relativistic momentum through tanh and temper by confidence.
        w = math.tanh(self._gain * signals.momentum) * signals.confidence

        # Dead-band: suppress small, noisy weights.
        if abs(w) < self._deadband:
            w = 0.0

        w = max(-1.0, min(1.0, w))
        return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=w)


def build() -> Strategy:
    """Construct and return a RelativisticMomentumStrategy with default parameters."""
    return RelativisticMomentumStrategy()
