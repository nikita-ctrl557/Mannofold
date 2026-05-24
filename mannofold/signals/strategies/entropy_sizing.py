"""Info-theoretic strategy: size positions by regime-assignment certainty.

Treats regime_prob as the confidence of a binary outcome and computes binary
entropy H = -p*log2(p) - (1-p)*log2(1-p). Certainty = 1 - H in [0,1].
High-entropy (uncertain) states produce near-zero exposure; crisp regime
assignments (p near 0 or 1) allow full expression of the trend conviction.
Anomaly score is folded in as a second filter so off-manifold states are flat.
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

NAME = "entropy_sizing"
DESCRIPTION = (
    "Info-theoretic sizing: weight = tanh(gain*tanh(sharpe)) * certainty "
    "* (1-anomaly_score), where certainty = 1 - binary_entropy(regime_prob). "
    "High-entropy (uncertain) states collapse to flat; ANOMALY_REGIME always flat."
)

_EPS = 1e-9
_P_CLAMP = 1e-6            # clamp regime_prob away from 0/1 before log
_GAIN = 60.0               # tanh amplification on the Sharpe proxy
_DEADBAND = 0.04           # |weight| below this -> 0


def _binary_entropy(p: float) -> float:
    """Binary entropy H(p) in bits, p clamped to [_P_CLAMP, 1-_P_CLAMP]."""
    p = max(_P_CLAMP, min(1.0 - _P_CLAMP, p))
    q = 1.0 - p
    return -(p * math.log2(p) + q * math.log2(q))


class EntropySizingStrategy:
    """Size by certainty derived from binary entropy of regime_prob.

    weight = tanh(gain * tanh(sharpe)) * certainty * (1 - anomaly_score)

    where:
        sharpe    = fwd_return_mean / (fwd_return_std + eps)
        H         = -p*log2(p) - (1-p)*log2(1-p),  p = regime_prob (clamped)
        certainty = 1 - H   in [0, 1]

    Flat when:
      - regime_id == ANOMALY_REGIME
      - |weight| < _DEADBAND  (dead-band suppresses noise)
    Clamped to [-1, 1].
    """

    def __init__(
        self,
        gain: float = _GAIN,
        deadband: float = _DEADBAND,
    ) -> None:
        self._gain = gain
        self._deadband = deadband

    def signals(self, state: ManifoldState) -> SignalSet:
        sharpe = state.fwd_return_mean / (state.fwd_return_std + _EPS)
        H = _binary_entropy(state.regime_prob)
        certainty = 1.0 - H  # 0 = max uncertainty, 1 = perfectly certain
        confidence = certainty * max(0.0, 1.0 - state.anomaly_score)
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

        # Hard gate: anomalous regime is always flat.
        if signals.regime_id == ANOMALY_REGIME:
            return flat

        # Reconstruct sharpe proxy via atanh(momentum); clamp to avoid blowup.
        mom = max(-1.0 + _EPS, min(1.0 - _EPS, signals.momentum))
        sharpe_proxy = math.atanh(mom)

        # Trend conviction modulated by certainty and anomaly filter.
        weight = (
            math.tanh(self._gain * math.tanh(sharpe_proxy))
            * signals.confidence
        )

        # Dead-band: tiny weights -> flat to avoid churn.
        if abs(weight) < self._deadband:
            return flat

        weight = max(-1.0, min(1.0, weight))
        return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=weight)


def build() -> Strategy:
    return EntropySizingStrategy()
