"""Month-over-month consistency mean-reversion strategy.

Fades SMALL deviations ONLY in calm, range-bound conditions: high density
(typical manifold region), low forward-return dispersion, and low anomaly.
Stays completely flat in trending, volatile, or anomalous states to avoid the
big losing months that destroy consistency.

weight = -clamp(tanh(gain * tanh(sharpe)), -0.4, 0.4) * calm_gate * confidence

where:
    sharpe       = fwd_return_mean / (fwd_return_std + eps)
    calm_gate    = density_sigmoid * vol_gate    (~1 only when calm/range-bound)
    confidence   = regime_prob * (1 - anomaly_score)
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

NAME = "mean_revert_calm_only"
DESCRIPTION = (
    "Consistency-first contrarian fade: enters ONLY in calm, high-density, "
    "low-dispersion regimes; caps weight at 0.4 for many small wins; goes "
    "flat whenever trending, volatile, or anomalous."
)

_EPS = 1e-9

# inner tanh gain — keeps individual positions small (many tiny wins)
_GAIN = 2.5

# density gate: sigmoid centred at _DENSITY_MID
# typical density range 0.18..0.77; mid ~0.40
_DENSITY_MID = 0.40
_DENSITY_SCALE = 10.0
_DENSITY_CLAMP = 50.0

# vol gate: fwd_return_std above _VOL_HIGH -> gate collapses to 0
# typical fwd_std ~0.01..0.037; only trade in genuinely calm bands
_VOL_HIGH = 0.018
_VOL_SCALE = 250.0

# hard anomaly cutoff — go flat above this
_ANOMALY_GATE = 0.5

# weight cap — never take more than ±40% exposure (consistency over size)
_WEIGHT_CAP = 0.4

# dead-band — suppress micro positions (< 4 % -> exactly 0)
_DEADBAND = 0.04


def _density_sigmoid(density: float) -> float:
    """Smooth gate [0, 1]: low density -> ~0, high density -> ~1."""
    d = max(0.0, min(density, _DENSITY_CLAMP))
    return 1.0 / (1.0 + math.exp(-_DENSITY_SCALE * (d - _DENSITY_MID)))


def _vol_gate(fwd_return_std: float) -> float:
    """Smooth gate [0, 1]: high vol -> ~0, low vol -> ~1."""
    return 1.0 / (1.0 + math.exp(_VOL_SCALE * (fwd_return_std - _VOL_HIGH)))


class MeanRevertCalmOnlyStrategy:
    """Calm-gated, capped contrarian fade for month-over-month consistency."""

    def signals(self, state: ManifoldState) -> SignalSet:
        sharpe = state.fwd_return_mean / (state.fwd_return_std + _EPS)
        density_g = _density_sigmoid(state.density)
        vol_g = _vol_gate(state.fwd_return_std)
        calm_gate = density_g * vol_g
        confidence = max(0.0, state.regime_prob * (1.0 - state.anomaly_score))
        return SignalSet(
            ts=state.ts,
            symbol=state.symbol,
            momentum=math.tanh(sharpe),
            expected_return=state.fwd_return_mean,
            anomaly=state.anomaly_score,
            regime_id=state.regime_id,
            confidence=confidence * calm_gate,
        )

    def target(self, signals: SignalSet) -> TargetPosition:
        # Hard gates: anomalous regime or elevated anomaly score -> flat
        if signals.regime_id == ANOMALY_REGIME or signals.anomaly > _ANOMALY_GATE:
            return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=0.0)

        # Contrarian fade: negate the neighbourhood Sharpe direction
        raw_fade = -math.tanh(_GAIN * signals.momentum)
        # Clamp to [-0.4, 0.4] BEFORE applying confidence (consistency cap)
        clamped = max(-_WEIGHT_CAP, min(_WEIGHT_CAP, raw_fade))
        weight = clamped * signals.confidence  # confidence already embeds calm_gate

        # Dead-band: collapse negligible weights to flat
        if abs(weight) < _DEADBAND:
            weight = 0.0

        weight = max(-1.0, min(1.0, weight))
        return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=weight)


def build() -> Strategy:
    return MeanRevertCalmOnlyStrategy()
