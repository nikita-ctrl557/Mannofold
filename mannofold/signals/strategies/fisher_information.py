"""Information-geometry strategy: size positions by Fisher information (precision).

For a Gaussian drift estimate, the Fisher information about the mean is
I = 1/sigma^2.  High I means the drift estimate is PRECISE and trustworthy.
Positions are weighted by precision: confident low-variance states get more
capital; noisy high-variance states get little.
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

NAME = "fisher_information"
DESCRIPTION = (
    "Information-geometry sizing: weight = tanh(gain * fwd_return_mean * I_norm) "
    "* confidence, where I_norm = clamp(1/(fwd_return_std^2 + eps), 0, 1) and "
    "confidence = regime_prob * (1 - anomaly_score). "
    "High-precision (low-variance) drift estimates receive more capital; "
    "ANOMALY_REGIME and anomaly_score > 0.6 are always flat."
)

_EPS = 1e-8
_GAIN = 40.0           # tanh amplification on the precision-weighted drift
_DEADBAND = 0.04       # |weight| below this -> flat to suppress churn
_ANOMALY_GATE = 0.6    # hard anomaly-score threshold for flat
# Scale factor so that fwd_return_std ~ 0.005 (50 bps daily) normalises I to ~1
_I_SCALE = 0.005 ** 2  # = 2.5e-5; I_norm = clamp(I * _I_SCALE, 0, 1)


class FisherInformationStrategy:
    """Size by Fisher information (precision) of the forward-return estimate.

    For a Gaussian with unknown mean mu and known variance sigma^2, the
    Fisher information about mu is I(mu) = 1/sigma^2.

    I_normalized = clamp(1 / (fwd_return_std^2 + eps) * _I_SCALE, 0, 1)
    confidence   = regime_prob * (1 - anomaly_score)
    weight       = tanh(gain * fwd_return_mean * I_normalized) * confidence

    Flat when:
      - regime_id == ANOMALY_REGIME
      - anomaly_score > _ANOMALY_GATE
      - |weight| < _DEADBAND
    Clamped defensively to [-1, 1].
    """

    def __init__(
        self,
        gain: float = _GAIN,
        deadband: float = _DEADBAND,
    ) -> None:
        self._gain = gain
        self._deadband = deadband

    def signals(self, state: ManifoldState) -> SignalSet:
        std = max(state.fwd_return_std, _EPS)
        fisher_info = 1.0 / (std ** 2 + _EPS)
        i_norm = min(fisher_info * _I_SCALE, 1.0)
        confidence = state.regime_prob * max(0.0, 1.0 - state.anomaly_score)
        return SignalSet(
            ts=state.ts,
            symbol=state.symbol,
            momentum=math.tanh(self._gain * state.fwd_return_mean * i_norm),
            expected_return=state.fwd_return_mean,
            anomaly=state.anomaly_score,
            regime_id=state.regime_id,
            confidence=confidence,
        )

    def target(self, signals: SignalSet) -> TargetPosition:
        flat = TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=0.0)

        # Hard gates: anomalous regime or high anomaly score -> flat.
        if signals.regime_id == ANOMALY_REGIME:
            return flat
        if signals.anomaly > _ANOMALY_GATE:
            return flat

        # momentum already encodes tanh(gain * mean * I_norm).
        weight = signals.momentum * signals.confidence

        # Dead-band: suppress tiny/noisy weights.
        if abs(weight) < self._deadband:
            return flat

        weight = max(-1.0, min(1.0, weight))
        return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=weight)


def build() -> Strategy:
    return FisherInformationStrategy()
