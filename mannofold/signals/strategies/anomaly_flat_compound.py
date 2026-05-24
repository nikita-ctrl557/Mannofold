"""Anomaly-flat-compound strategy: sit out turbulence, compound the calm.

Month-over-month CONSISTENCY theme. Goes fully flat whenever anomaly_score
is even moderately elevated (>0.35). Turbulent periods cause the big losing
months; by skipping them entirely and only participating in calm, low-anomaly
states the strategy accumulates a steady sequence of positive months.

Signal:
    sharpe      = fwd_return_mean / (fwd_return_std + eps)
    confidence  = regime_prob * (1 - anomaly_score)
    weight      = tanh(gain * tanh(sharpe)) * confidence   if anomaly < 0.35
                = 0                                         otherwise

Gates:
    flat on ANOMALY_REGIME (-1)
    flat when anomaly_score >= 0.35
    dead-band |w| < 0.04 -> 0
    no lookahead; build() takes no args
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

NAME = "anomaly_flat_compound"
DESCRIPTION = (
    "Goes fully flat whenever anomaly_score > 0.35 to avoid turbulent losing months; "
    "in calm, low-anomaly states takes a steady drift position using "
    "tanh(gain*tanh(sharpe))*confidence, compounding consistent positive months."
)

_EPS = 1e-9
_GAIN = 2.5           # amplifier inside the double-tanh
_ANOMALY_THRESHOLD = 0.35   # hard flat above this — even mild turbulence is excluded
_DEAD_BAND = 0.04     # collapse |weight| below this to zero


class AnomalyFlatCompoundStrategy:
    """Compound calm periods; go fully flat on any elevated anomaly.

    The asymmetry of participating only in calm, regime-confirmed states is
    the core edge: the large negative months come from turbulent periods, so
    simply sitting them out creates a positively-skewed monthly return stream.
    """

    def signals(self, state: ManifoldState) -> SignalSet:
        sharpe = state.fwd_return_mean / (state.fwd_return_std + _EPS)
        sharpe = max(-10.0, min(10.0, sharpe))

        # confidence collapses when either regime is uncertain OR anomaly is rising
        confidence = max(0.0, state.regime_prob * (1.0 - state.anomaly_score))

        return SignalSet(
            ts=state.ts,
            symbol=state.symbol,
            momentum=math.tanh(sharpe),           # tanh(sharpe) stored for target()
            expected_return=state.fwd_return_mean,
            anomaly=state.anomaly_score,
            regime_id=state.regime_id,
            confidence=confidence,
        )

    def target(self, signals: SignalSet) -> TargetPosition:
        sym = signals.symbol

        # Hard flat: anomalous regime sentinel or any elevated anomaly score
        if signals.regime_id == ANOMALY_REGIME or signals.anomaly >= _ANOMALY_THRESHOLD:
            return TargetPosition(ts=signals.ts, symbol=sym, target_weight=0.0)

        # Double-tanh: outer tanh bounds result; inner tanh(sharpe) from signals()
        base = math.tanh(_GAIN * signals.momentum)

        # Final weight: base * confidence
        weight = base * signals.confidence

        # Dead-band: suppress small noisy weights to reduce unnecessary trading
        if abs(weight) < _DEAD_BAND:
            weight = 0.0

        weight = max(-1.0, min(1.0, weight))
        return TargetPosition(ts=signals.ts, symbol=sym, target_weight=weight)


def build() -> Strategy:
    """Return a fresh AnomalyFlatCompoundStrategy instance."""
    return AnomalyFlatCompoundStrategy()
