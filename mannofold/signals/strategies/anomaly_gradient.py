"""Anomaly-gradient strategy: trade the RATE OF CHANGE of anomaly_score.

Re-enters as the state returns to the manifold (falling gradient) and trims
toward flat as it drifts further off-manifold (rising gradient).
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

NAME = "anomaly_gradient"
DESCRIPTION = (
    "Trades the gradient (rate of change) of anomaly_score: sizes up when the "
    "state is re-converging to the manifold (grad < 0) and trims to flat when "
    "it is drifting further off (grad > 0). Flat on ANOMALY_REGIME."
)

# Tunable knobs
_GAIN = 2.5     # amplifier inside the double-tanh: tanh(gain * tanh(sharpe))
_K = 3.0        # gradient sensitivity: clamp(0.2 - k*grad, 0, 1)
_DEAD_BAND = 0.04  # collapse |weight| below this to 0


class AnomalyGradientStrategy:
    """Size by trend conviction scaled by how fast anomaly is resolving.

    Per-symbol state: previous anomaly_score so we can compute
    grad = anomaly_score - prev_anomaly_score.

    weight = tanh(gain * tanh(sharpe))
             * clamp(0.2 - k * grad, 0, 1)
             * regime_prob

    When grad < 0 (anomaly falling, state returning to manifold) the gradient
    factor exceeds 0.2 and approaches 1, amplifying the trend signal.
    When grad > 0 (anomaly rising, drifting off-manifold) the factor shrinks
    toward 0, reducing exposure.
    """

    def __init__(self) -> None:
        # per-symbol: last seen anomaly_score
        self._prev_anomaly: dict[str, float] = {}

    def signals(self, state: ManifoldState) -> SignalSet:
        sym = state.symbol

        # Gradient of anomaly_score (0 on first bar for each symbol)
        prev = self._prev_anomaly.get(sym, state.anomaly_score)
        grad = state.anomaly_score - prev
        self._prev_anomaly[sym] = state.anomaly_score

        # Sharpe: expected forward return per unit of its spread
        sharpe = state.fwd_return_mean / (state.fwd_return_std + 1e-9)

        # Encode gradient into momentum field for use in target()
        # We embed both sharpe and grad via a tuple-like encoding in momentum.
        # momentum = tanh(gain * tanh(sharpe))  (pre-computed for cleanliness)
        momentum = math.tanh(_GAIN * math.tanh(sharpe))

        # Pass gradient via confidence channel (scaled to [-1, 1] range)
        # We'll pass raw gradient as anomaly field since we need it in target()
        # Use confidence for regime_prob and anomaly for the gradient
        confidence = max(0.0, min(1.0, state.regime_prob))

        return SignalSet(
            ts=state.ts,
            symbol=state.symbol,
            momentum=momentum,
            expected_return=state.fwd_return_mean,
            anomaly=grad,           # repurposed: raw gradient of anomaly_score
            regime_id=state.regime_id,
            confidence=confidence,  # regime_prob
        )

    def target(self, signals: SignalSet) -> TargetPosition:
        sym = signals.symbol

        # Hard flat on anomalous regime sentinel
        if signals.regime_id == ANOMALY_REGIME:
            return TargetPosition(ts=signals.ts, symbol=sym, target_weight=0.0)

        grad = signals.anomaly  # gradient of anomaly_score (from signals())

        # Gradient factor: clamp(0.2 - k*grad, 0, 1)
        # grad < 0 -> factor > 0.2 (anomaly resolving -> amplify)
        # grad > 0 -> factor < 0.2 (anomaly rising -> shrink)
        grad_factor = max(0.0, min(1.0, 0.2 - _K * grad))

        # Final weight: double-tanh conviction * gradient factor * regime_prob
        weight = signals.momentum * grad_factor * signals.confidence

        # Dead-band: suppress small noisy weights
        if abs(weight) < _DEAD_BAND:
            weight = 0.0

        return TargetPosition(ts=signals.ts, symbol=sym, target_weight=weight)


def build() -> Strategy:
    """Return a fresh AnomalyGradientStrategy instance."""
    return AnomalyGradientStrategy()
