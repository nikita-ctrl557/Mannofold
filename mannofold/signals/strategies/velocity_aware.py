"""Velocity-aware manifold strategy.

Uses the trajectory VELOCITY of the manifold state — how fast the embedding
coordinates are changing — to scale conviction. A decisively-moving manifold
(fast regime transition) gets full weight; a stagnant manifold gets trimmed.
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

NAME = "velocity_aware"
DESCRIPTION = (
    "Scales Sharpe-derived conviction by manifold trajectory speed: "
    "lean in on fast regime transitions, trim on stagnation."
)

_EPS = 1e-9

# Typical L2-norm of velocity when the manifold is 'moving decisively'.
# Empirically ~median speed across a typical run is ~1.7; we use 2.0 so that
# roughly half the bars are below full conviction and the scaling is active.
# The clamp keeps it bounded in [0, 1] regardless of scale choice.
_SPEED_SCALE = 2.0

# Gain that maps neighbourhood-Sharpe into a weight via tanh.
_GAIN = 60.0

# Anomaly above this threshold forces flat.
_ANOMALY_GATE = 0.6

# Dead-band: weights whose |w| < this collapse to 0.
_DEADBAND = 0.04


def _speed(velocity: list[float]) -> float:
    """L2 norm of the velocity vector; returns 0 if empty or too short."""
    if not velocity:
        return 0.0
    return math.sqrt(sum(v * v for v in velocity))


def _f_speed(speed: float, scale: float = _SPEED_SCALE) -> float:
    """Clamp(speed / scale, 0, 1) — velocity scaling factor in [0, 1]."""
    if scale <= 0:
        return 0.0
    return max(0.0, min(1.0, speed / scale))


class _SpeedAwareSignalSet(SignalSet):
    """SignalSet subclass that carries the manifold speed for target()."""

    speed: float = 0.0

    model_config = {"extra": "allow"}


class VelocityAwareStrategy:
    """Manifold strategy that gates conviction on trajectory speed."""

    def __init__(
        self,
        gain: float = _GAIN,
        anomaly_gate: float = _ANOMALY_GATE,
        deadband: float = _DEADBAND,
        speed_scale: float = _SPEED_SCALE,
    ):
        self._gain = gain
        self._anomaly_gate = anomaly_gate
        self._deadband = deadband
        self._speed_scale = speed_scale

    def signals(self, state: ManifoldState) -> SignalSet:
        """Compute signals, embedding trajectory speed into the returned set."""
        sharpe = state.fwd_return_mean / (state.fwd_return_std + _EPS)
        momentum = math.tanh(sharpe)
        confidence = max(0.0, state.regime_prob * (1.0 - state.anomaly_score))
        spd = _speed(state.velocity)
        return _SpeedAwareSignalSet(
            ts=state.ts,
            symbol=state.symbol,
            momentum=momentum,
            expected_return=state.fwd_return_mean,
            anomaly=state.anomaly_score,
            regime_id=state.regime_id,
            confidence=confidence,
            speed=spd,
        )

    def target(self, signals: SignalSet) -> TargetPosition:
        """Compute target weight, scaling by trajectory speed."""
        # Hard-off conditions: anomalous regime or high anomaly score.
        if signals.regime_id == ANOMALY_REGIME or signals.anomaly > self._anomaly_gate:
            return TargetPosition(
                ts=signals.ts, symbol=signals.symbol, target_weight=0.0
            )

        # signals.momentum is tanh(sharpe) from signals().
        sharpe_tanh = signals.momentum

        # confidence = regime_prob * (1 - anomaly_score)
        confidence = signals.confidence

        # Retrieve speed (0 if signals is a plain SignalSet with no speed attr).
        spd = getattr(signals, "speed", 0.0)

        # weight = tanh(gain * tanh(sharpe)) * f(speed) * confidence
        w = math.tanh(self._gain * sharpe_tanh) * _f_speed(spd, self._speed_scale) * confidence

        # Dead-band.
        if abs(w) < self._deadband:
            w = 0.0

        w = max(-1.0, min(1.0, w))
        return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=w)


def build() -> Strategy:
    """Construct and return a VelocityAwareStrategy with default parameters."""
    return VelocityAwareStrategy()
