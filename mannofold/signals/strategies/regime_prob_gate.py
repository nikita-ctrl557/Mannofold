"""Regime-probability-gated strategy variant.

Follows drift (Sharpe-based tanh sizing) but scales exposure by regime_prob**2
times (1-anomaly_score). Low-confidence regimes and anomalous states receive
near-zero or zero exposure.
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

NAME = "regime_prob_gate"
DESCRIPTION = (
    "Drift-following strategy gated by regime_prob**2 * (1-anomaly_score); "
    "low-confidence regimes collapse to flat."
)

_EPS = 1e-9
_GAIN = 60.0          # tanh amplification on the Sharpe proxy
_ANOMALY_GATE = 0.6   # anomaly_score threshold above which -> flat
_DEADBAND = 0.04      # |weight| below this -> 0


class RegimeProbGateStrategy:
    """Size positions only in high-confidence regimes.

    weight = tanh(gain * tanh(sharpe)) * regime_prob**2 * (1 - anomaly_score)

    Flat when:
      - regime_id == ANOMALY_REGIME
      - anomaly_score > _ANOMALY_GATE
      - |weight| < _DEADBAND  (dead-band)
    Clamped to [-1, 1].
    """

    def __init__(
        self,
        gain: float = _GAIN,
        anomaly_gate: float = _ANOMALY_GATE,
        deadband: float = _DEADBAND,
    ) -> None:
        self._gain = gain
        self._anomaly_gate = anomaly_gate
        self._deadband = deadband

    def signals(self, state: ManifoldState) -> SignalSet:
        sharpe = state.fwd_return_mean / (state.fwd_return_std + _EPS)
        # Confidence: high only when regime is stable AND anomaly is low.
        confidence = max(0.0, state.regime_prob ** 2 * (1.0 - state.anomaly_score))
        return SignalSet(
            ts=state.ts,
            symbol=state.symbol,
            momentum=math.tanh(sharpe),
            expected_return=state.fwd_return_mean,
            anomaly=state.anomaly_score,
            regime_id=state.regime_id,
            confidence=confidence,
        )

    def target(self, signals: SignalSet) -> TargetPosition:
        flat = TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=0.0)

        # Hard gates.
        if signals.regime_id == ANOMALY_REGIME or signals.anomaly > self._anomaly_gate:
            return flat

        sharpe = signals.expected_return / (signals.fwd_return_std + _EPS) if hasattr(signals, "fwd_return_std") else signals.expected_return / (abs(signals.momentum) + _EPS)  # noqa: E501
        # signals.momentum == tanh(sharpe), so reconstruct sharpe proxy via atanh.
        # Clamp momentum to avoid atanh blowup.
        mom = max(-1.0 + _EPS, min(1.0 - _EPS, signals.momentum))
        sharpe_proxy = math.atanh(mom)

        base = math.tanh(self._gain * math.tanh(sharpe_proxy))
        weight = base * signals.confidence

        # Dead-band.
        if abs(weight) < self._deadband:
            return flat

        weight = max(-1.0, min(1.0, weight))
        return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=weight)


def build() -> Strategy:
    return RegimeProbGateStrategy()
