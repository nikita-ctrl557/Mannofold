"""Anomaly-defensive strategy: neighbourhood-drift trading with aggressive de-grossing.

Trades the neighbourhood drift but aggressively reduces exposure as the state
drifts off-manifold. The anomaly haircut decays the position smoothly to zero
well before full anomaly, avoiding sharp regime boundaries.
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

NAME = "anomaly_defensive"
DESCRIPTION = "Neighbourhood-drift conviction with smooth anomaly haircut; flat on anomaly regime."

# Tunable knobs
_GAIN = 2.5          # amplifier inside the double-tanh: tanh(gain * tanh(sharpe))
_ANOMALY_CAP = 0.6   # anomaly_score >= this -> weight decays to 0
_DEAD_BAND = 0.04    # collapse |weight| below this to 0


class AnomalyDefensiveStrategy:
    """Trade neighbourhood drift but aggressively de-gross as state drifts off-manifold.

    Key mechanic: a multiplicative haircut ``factor = max(0, 1 - anomaly_score / anomaly_cap)``
    ensures the position smoothly decays to zero as the anomaly_score approaches
    ``anomaly_cap``, well before reaching full anomaly. Additionally the strategy
    goes flat whenever the regime is ANOMALY_REGIME (-1).
    """

    def signals(self, state: ManifoldState) -> SignalSet:
        # Neighbourhood Sharpe: expected forward return per unit of its spread
        sharpe = state.fwd_return_mean / (state.fwd_return_std + 1e-9)

        # Double-tanh momentum: outer tanh bounds the result, inner tanh bounds sharpe
        momentum = math.tanh(_GAIN * math.tanh(sharpe))

        # Confidence is purely regime probability (anomaly handled separately in target)
        confidence = max(0.0, min(1.0, state.regime_prob))

        return SignalSet(
            ts=state.ts,
            symbol=state.symbol,
            momentum=momentum,
            expected_return=state.fwd_return_mean,
            anomaly=state.anomaly_score,
            regime_id=state.regime_id,
            confidence=confidence,
        )

    def target(self, signals: SignalSet) -> TargetPosition:
        # Hard flat on anomalous regime sentinel
        if signals.regime_id == ANOMALY_REGIME:
            return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=0.0)

        # Smooth multiplicative anomaly haircut: decays linearly to 0 at anomaly_cap
        factor = max(0.0, 1.0 - signals.anomaly / _ANOMALY_CAP)

        # Final weight: momentum * haircut * confidence
        weight = signals.momentum * factor * signals.confidence

        # Dead-band: suppress small noisy weights
        if abs(weight) < _DEAD_BAND:
            weight = 0.0

        return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=weight)


def build() -> Strategy:
    """Return a fresh AnomalyDefensiveStrategy instance."""
    return AnomalyDefensiveStrategy()
