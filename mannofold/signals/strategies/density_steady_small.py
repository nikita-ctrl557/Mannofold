"""Density-steady-small strategy.

Month-over-month CONSISTENCY: trade ONLY in the highest-density (most typical,
reliable) manifold states with a small capped position size and light EMA
smoothing, for steady predictable monthly returns.

Signal:
    sharpe         = fwd_return_mean / (fwd_return_std + eps)
    base           = tanh(gain * tanh(sharpe))
    high_density_gate  ~ 1 only when density is in the top band, else 0
                     implemented as hard threshold + soft sigmoid taper
    confidence     = regime_prob * (1 - anomaly_score)
    raw_weight     = clamp(tanh(gain * tanh(sharpe)), -0.35, 0.35)
                     * high_density_gate * confidence
    weight         = EMA-smoothed raw_weight (light alpha)

Gates:
    flat on ANOMALY_REGIME or anomaly_score > 0.6
    dead-band |w| < 0.04 -> 0
    max absolute weight capped at 0.35
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

NAME = "density_steady_small"
DESCRIPTION = (
    "Trades only in the highest-density (most typical) manifold states with a "
    "small capped position size and light EMA smoothing for steady, consistent "
    "month-over-month returns.  Flat in anomalous, sparse, or uncertain regimes."
)

_EPS = 1e-9
_GAIN = 2.5
_ANOMALY_GATE = 0.6
_DEADBAND = 0.04
_MAX_WEIGHT = 0.35

# Hard lower-bound density threshold: below this the gate is zero.
_DENSITY_HARD_THRESHOLD = 0.55
# Above _DENSITY_TOP the gate is fully open; between threshold and top it tapers.
_DENSITY_TOP = 0.70
_DENSITY_SIGMOID_SCALE = 20.0

# EMA smoothing alpha: small alpha = smoother / slower response.
_EMA_ALPHA = 0.25


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


def _high_density_gate(density: float) -> float:
    """Returns ~1 only in the top density band, ~0 for typical/low density."""
    if density < _DENSITY_HARD_THRESHOLD:
        return 0.0
    # Soft sigmoid taper between _DENSITY_HARD_THRESHOLD and _DENSITY_TOP.
    mid = (_DENSITY_HARD_THRESHOLD + _DENSITY_TOP) * 0.5
    return _sigmoid(_DENSITY_SIGMOID_SCALE * (density - mid))


class DensitySteadySmallStrategy:
    """Steady small-size carry in the highest-density manifold regions only."""

    def __init__(self) -> None:
        self._ema: float | None = None

    def signals(self, state: ManifoldState) -> SignalSet:
        sharpe = state.fwd_return_mean / (state.fwd_return_std + _EPS)
        sharpe = max(-10.0, min(10.0, sharpe))

        gate = _high_density_gate(state.density)
        confidence = max(0.0, state.regime_prob * (1.0 - state.anomaly_score))

        return SignalSet(
            ts=state.ts,
            symbol=state.symbol,
            momentum=math.tanh(sharpe),
            expected_return=state.fwd_return_mean,
            anomaly=state.anomaly_score,
            regime_id=state.regime_id,
            confidence=confidence * gate,
        )

    def target(self, signals: SignalSet) -> TargetPosition:
        sym = signals.symbol

        # Hard gates: anomalous regime or elevated anomaly score -> flat.
        if signals.regime_id == ANOMALY_REGIME or signals.anomaly > _ANOMALY_GATE:
            self._ema = 0.0
            return TargetPosition(ts=signals.ts, symbol=sym, target_weight=0.0)

        # Base weight: tanh(gain * tanh(sharpe)); signals.momentum == tanh(sharpe).
        base = math.tanh(_GAIN * signals.momentum)

        # Apply max-size cap first, then density gate * confidence.
        raw = max(-_MAX_WEIGHT, min(_MAX_WEIGHT, base)) * signals.confidence

        # EMA smoothing for month-over-month steadiness.
        if self._ema is None:
            self._ema = raw
        else:
            self._ema = _EMA_ALPHA * raw + (1.0 - _EMA_ALPHA) * self._ema

        weight = self._ema

        # Dead-band: collapse small weights to flat.
        if abs(weight) < _DEADBAND:
            weight = 0.0

        weight = max(-_MAX_WEIGHT, min(_MAX_WEIGHT, weight))
        return TargetPosition(ts=signals.ts, symbol=sym, target_weight=weight)


def build() -> Strategy:
    return DensitySteadySmallStrategy()
