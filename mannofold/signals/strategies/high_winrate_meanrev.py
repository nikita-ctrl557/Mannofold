"""High-win-rate mean-reversion strategy.

Fades SMALL deviations ONLY in calm, predictable conditions — high density
(typical manifold region) and low forward-return dispersion — where reversion
is reliable and produces many small wins. Stays flat in trending, anomalous, or
high-volatility states where contrarian bets lose big.

weight = -tanh(gain * tanh(sharpe)) * calm_gate * confidence

where:
    sharpe       = fwd_return_mean / (fwd_return_std + eps)
    calm_gate    = density_sigmoid * vol_gate   (~1 only when calm)
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

NAME = "high_winrate_meanrev"
DESCRIPTION = (
    "High-win-rate contrarian fade: only trades in calm, high-density, low-dispersion "
    "regimes where mean reversion is reliable; stays flat in trending/anomalous states."
)

_EPS = 1e-9

# --- inner tanh gain: keeps weight small (many tiny wins, not rare big ones)
_GAIN = 2.0

# --- density gate: sigmoid centred on _DENSITY_MID; high density -> gate near 1
#     density values typically range 0.18..0.77 in practice; mid at ~0.45
_DENSITY_MID = 0.45
_DENSITY_SCALE = 8.0
_DENSITY_CLAMP = 50.0

# --- volatility gate: fwd_return_std above _VOL_HIGH -> gate collapses to 0
#     uses a soft inverted sigmoid so the transition is smooth
#     typical fwd_std range ~0.01..0.037; low-vol threshold ~0.018
_VOL_HIGH = 0.020      # std above which calm_gate ~ 0
_VOL_SCALE = 200.0     # steepness of the vol gate

# --- hard anomaly cut-off
_ANOMALY_GATE = 0.6

# --- dead-band: suppress micro positions
_DEADBAND = 0.04


def _density_sigmoid(density: float) -> float:
    """Smooth gate [0,1]: low density -> ~0, high density -> ~1."""
    d = max(0.0, min(density, _DENSITY_CLAMP))
    return 1.0 / (1.0 + math.exp(-_DENSITY_SCALE * (d - _DENSITY_MID)))


def _vol_gate(fwd_return_std: float) -> float:
    """Smooth gate [0,1]: high vol -> ~0, low vol -> ~1."""
    return 1.0 / (1.0 + math.exp(_VOL_SCALE * (fwd_return_std - _VOL_HIGH)))


class HighWinrateMeanRevStrategy:
    """Calm-gated contrarian fade producing many small wins, few big losses."""

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
        if signals.regime_id == ANOMALY_REGIME or signals.anomaly > _ANOMALY_GATE:
            return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=0.0)

        # Contrarian fade: negate the neighbourhood Sharpe direction, scaled by calm confidence
        raw_fade = -math.tanh(_GAIN * signals.momentum)
        weight = raw_fade * signals.confidence  # confidence already embeds calm_gate

        if abs(weight) < _DEADBAND:
            weight = 0.0

        weight = max(-1.0, min(1.0, weight))
        return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=weight)


def build() -> Strategy:
    return HighWinrateMeanRevStrategy()
