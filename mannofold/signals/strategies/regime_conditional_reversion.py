"""Regime-conditional mean-reversion strategy.

Mean-reverts ONLY in low-density (atypical) states where extremes tend to
revert, and only when dispersion is high enough to suggest over-extension.
In typical / trending states the position is flat — no fighting trends.
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

NAME = "regime_conditional_reversion"
DESCRIPTION = (
    "Mean-reverts only in low-density atypical states with high dispersion; "
    "flat in typical/trending regimes to avoid fighting trends."
)

_EPS = 1e-9
_GAIN = 3.0           # outer tanh gain on the faded Sharpe
_ANOMALY_GATE = 0.6   # anomaly_score above this -> flat
_DEADBAND = 0.04      # |weight| below this -> 0

# Reversion gate parameters.
# Gate rises as density FALLS (atypical) and dispersion RISES (overextended).
_DENSITY_MID = 1.0     # density value at which density factor = 0.5
_DENSITY_SCALE = 2.0   # sigmoid steepness for density component
_DENSITY_CLAMP = 50.0  # defensive upper clamp
_DISP_MID = 0.005      # fwd_return_std at which dispersion factor = 0.5
_DISP_SCALE = 200.0    # sigmoid steepness for dispersion component


def _low_density_factor(density: float) -> float:
    """Returns ~1 when density is LOW (atypical), ~0 when density is HIGH (typical).

    This is 1 - sigmoid(density), so sparsely populated regions activate it.
    """
    d = max(0.0, min(density, _DENSITY_CLAMP))
    high_density = 1.0 / (1.0 + math.exp(-_DENSITY_SCALE * (d - _DENSITY_MID)))
    return 1.0 - high_density


def _high_dispersion_factor(fwd_return_std: float) -> float:
    """Returns ~1 when dispersion is HIGH (overextended), ~0 when low."""
    s = max(0.0, fwd_return_std)
    return 1.0 / (1.0 + math.exp(-_DISP_SCALE * (s - _DISP_MID)))


def _reversion_gate(density: float, fwd_return_std: float) -> float:
    """Gate in [0, 1]: grows as density falls and dispersion rises."""
    return _low_density_factor(density) * _high_dispersion_factor(fwd_return_std)


class RegimeConditionalReversionStrategy:
    """Contrarian fade only in atypical, high-dispersion manifold regions.

    target_weight = -tanh(gain * tanh(sharpe)) * reversion_gate * confidence

    where:
      sharpe          = fwd_return_mean / (fwd_return_std + eps)
      reversion_gate  = low_density_factor * high_dispersion_factor  in [0, 1]
      confidence      = regime_prob * (1 - anomaly_score)

    Goes flat when:
      - regime_id == ANOMALY_REGIME
      - anomaly_score > _ANOMALY_GATE
      - |weight| < _DEADBAND
    """

    def signals(self, state: ManifoldState) -> SignalSet:
        sharpe = state.fwd_return_mean / (state.fwd_return_std + _EPS)
        gate = _reversion_gate(state.density, state.fwd_return_std)
        confidence = max(0.0, state.regime_prob * (1.0 - state.anomaly_score))
        return SignalSet(
            ts=state.ts,
            symbol=state.symbol,
            momentum=math.tanh(sharpe),
            expected_return=state.fwd_return_mean,
            anomaly=state.anomaly_score,
            regime_id=state.regime_id,
            confidence=confidence * gate,  # gate embedded for target()
        )

    def target(self, signals: SignalSet) -> TargetPosition:
        # Hard gates: anomalous regime or high anomaly score -> flat
        if signals.regime_id == ANOMALY_REGIME or signals.anomaly > _ANOMALY_GATE:
            return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=0.0)

        # Contrarian fade: negate the neighbourhood Sharpe direction
        raw_fade = -math.tanh(_GAIN * signals.momentum)

        # Weight is scaled by gated confidence (gate already folded in)
        weight = raw_fade * signals.confidence

        # Dead-band: suppress micro positions
        if abs(weight) < _DEADBAND:
            weight = 0.0

        weight = max(-1.0, min(1.0, weight))
        return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=weight)


def build() -> Strategy:
    return RegimeConditionalReversionStrategy()
