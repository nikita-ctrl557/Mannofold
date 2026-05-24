"""Mean-reversion (contrarian fade) strategy variant.

Fades strong neighbourhood drift: if the local manifold neighbourhood has been
trending hard in one direction, bet on reversion. Confidence is scaled by regime
stability and reduced when the state is anomalous. Goes flat in anomalous regimes
or when the anomaly score is high.
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

NAME = "mean_reversion"
DESCRIPTION = "Contrarian fade: short strong neighbourhood drift expecting reversion, gated by regime confidence."

_EPS = 1e-9
_GAIN = 3.0          # outer tanh gain — scales the faded Sharpe into weight
_ANOMALY_GATE = 0.6  # anomaly_score above this -> go flat
_DEADBAND = 0.04     # |weight| below this collapses to 0


class MeanReversionStrategy:
    """Contrarian fade of neighbourhood forward-return Sharpe.

    target_weight = -tanh(_GAIN * tanh(sharpe)) * confidence

    where sharpe = fwd_return_mean / (fwd_return_std + eps)
    and   confidence = regime_prob * (1 - anomaly_score)

    Goes flat when:
      - regime_id == ANOMALY_REGIME
      - anomaly_score > _ANOMALY_GATE
    """

    def signals(self, state: ManifoldState) -> SignalSet:
        sharpe = state.fwd_return_mean / (state.fwd_return_std + _EPS)
        # Momentum from the neighbourhood's perspective (positive = uptrend there)
        momentum = math.tanh(sharpe)
        confidence = max(0.0, state.regime_prob * (1.0 - state.anomaly_score))
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
        # Hard gates: anomalous regime or high anomaly score -> flat
        if signals.regime_id == ANOMALY_REGIME or signals.anomaly > _ANOMALY_GATE:
            return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=0.0)

        # Contrarian: negate the neighbourhood Sharpe direction
        sharpe_tanh = signals.momentum  # tanh(sharpe) already computed in signals()
        raw_fade = -math.tanh(_GAIN * sharpe_tanh)

        # Scale by confidence (regime_prob * (1 - anomaly))
        weight = raw_fade * signals.confidence

        # Dead-band: suppress micro positions
        if abs(weight) < _DEADBAND:
            weight = 0.0

        weight = max(-1.0, min(1.0, weight))
        return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=weight)


def build() -> Strategy:
    return MeanReversionStrategy()
