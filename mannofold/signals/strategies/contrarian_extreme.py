"""Contrarian-extreme strategy: selective fader of extreme neighbourhood Sharpe.

Stays completely flat unless the neighbourhood Sharpe is *extreme* (|sharpe| > THR).
When the condition fires, it bets on reversion — taking a contrarian position sized
by how far Sharpe exceeds the threshold, gated by regime confidence.  Few trades,
each one a high-conviction mean-reversion call.
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

NAME = "contrarian_extreme"
DESCRIPTION = (
    "Selective fader: flat unless |neighbourhood Sharpe| exceeds a high threshold, "
    "then takes a contrarian (reversion) position scaled by excess and gated by "
    "regime confidence."
)

_EPS = 1e-9
_THR = 1.0           # Sharpe threshold — only trade when |sharpe| > this
_GAIN = 2.5          # tanh gain applied to (|sharpe| - thr) excess
_ANOMALY_GATE = 0.6  # anomaly_score above this -> flat
_DEADBAND = 0.04     # |weight| below this collapses to 0


class ContrarianExtremeStrategy:
    """Selective contrarian fade of extreme neighbourhood Sharpe.

    target_weight = -sign(sharpe) * tanh(GAIN * (|sharpe| - THR)) * confidence
                    when |sharpe| > THR, else 0

    where confidence = regime_prob * (1 - anomaly_score)

    Goes flat when:
      - regime_id == ANOMALY_REGIME
      - anomaly_score > _ANOMALY_GATE
      - |sharpe| <= _THR  (not extreme enough to fade)
      - |weight| < _DEADBAND  (micro-position dead-band)
    """

    def signals(self, state: ManifoldState) -> SignalSet:
        sharpe = state.fwd_return_mean / (state.fwd_return_std + _EPS)
        confidence = max(0.0, min(1.0, state.regime_prob * (1.0 - state.anomaly_score)))
        return SignalSet(
            ts=state.ts,
            symbol=state.symbol,
            momentum=sharpe,           # raw Sharpe carried into target()
            expected_return=state.fwd_return_mean,
            anomaly=state.anomaly_score,
            regime_id=state.regime_id,
            confidence=confidence,
        )

    def target(self, signals: SignalSet) -> TargetPosition:
        zero = TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=0.0)

        # Hard gates: anomalous regime or high anomaly score -> flat
        if signals.regime_id == ANOMALY_REGIME or signals.anomaly > _ANOMALY_GATE:
            return zero

        sharpe = signals.momentum  # raw Sharpe passed through from signals()
        abs_sharpe = abs(sharpe)

        # Selectivity gate: only fade when Sharpe is truly extreme
        if abs_sharpe <= _THR:
            return zero

        # Contrarian weight: fade direction, sized by excess above threshold
        excess = abs_sharpe - _THR
        direction = -1.0 if sharpe > 0 else 1.0   # contrarian sign
        raw_weight = direction * math.tanh(_GAIN * excess)

        # Scale by confidence (regime certainty × non-anomalousness)
        weight = raw_weight * signals.confidence

        # Dead-band: suppress micro positions
        if abs(weight) < _DEADBAND:
            return zero

        weight = max(-1.0, min(1.0, weight))
        return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=weight)


def build() -> Strategy:
    """Return a fresh ContrarianExtremeStrategy with default parameters."""
    return ContrarianExtremeStrategy()
