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

Additional guard: only fade SMALL deviations (|sharpe| < MAX_SHARPE_ABS) to
avoid fighting strong trends. Large neighbourhood Sharpe -> skip, not fade.
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
    "High-win-rate contrarian fade: trades only in calm, high-density, low-dispersion "
    "regimes on SMALL deviations (|sharpe|<1) where mean reversion is reliable; "
    "stays flat in trending/anomalous/high-vol states."
)

_EPS = 1e-9

# Outer tanh gain — pushes weight toward ±1 for decisive but bounded positions.
_GAIN = 15.0

# Density gate: steep sigmoid so only truly high-density bars pass.
# density range ~0.18..0.77; mid=0.50 means the top half trades.
_DENSITY_MID = 0.50
_DENSITY_SCALE = 20.0   # sharp transition
_DENSITY_CLAMP = 50.0

# Volatility gate: inverted sigmoid; low fwd_return_std -> gate~1.
# fwd_std range ~0.010..0.037; cutoff at 0.018 keeps only the calmest bars.
_VOL_HIGH = 0.018       # std above which vol_gate -> 0
_VOL_SCALE = 250.0      # steepness of the inverted sigmoid

# Small-deviation guard: skip bars where neighbourhood Sharpe is large.
# A large |sharpe| means the regime is trending hard — reversion is risky.
_MAX_SHARPE_ABS = 1.0

# Hard anomaly cut-off: go flat when state is off-manifold.
_ANOMALY_GATE = 0.6

# Dead-band: suppress micro positions that add noise without return.
_DEADBAND = 0.04


def _density_sigmoid(density: float) -> float:
    """Smooth gate [0,1]: low density -> ~0, high density -> ~1."""
    d = max(0.0, min(density, _DENSITY_CLAMP))
    return 1.0 / (1.0 + math.exp(-_DENSITY_SCALE * (d - _DENSITY_MID)))


def _vol_gate(fwd_return_std: float) -> float:
    """Smooth gate [0,1]: high dispersion -> ~0, low dispersion -> ~1."""
    return 1.0 / (1.0 + math.exp(_VOL_SCALE * (fwd_return_std - _VOL_HIGH)))


class HighWinrateMeanRevStrategy:
    """Calm-gated contrarian fade of small deviations for maximum win rate.

    Trades only when:
      - manifold density is high (typical, well-mapped region)
      - fwd_return_std is low (calm, predictable regime)
      - neighbourhood Sharpe is modest (small deviation — reversion likely)
      - regime_id != ANOMALY_REGIME and anomaly_score < 0.6

    Stays flat otherwise, avoiding the large losses that destroy expectancy.
    """

    def signals(self, state: ManifoldState) -> SignalSet:
        sharpe = state.fwd_return_mean / (state.fwd_return_std + _EPS)
        density_g = _density_sigmoid(state.density)
        vol_g = _vol_gate(state.fwd_return_std)
        calm_gate = density_g * vol_g
        confidence = max(0.0, state.regime_prob * (1.0 - state.anomaly_score))
        # Small-deviation guard: zero confidence when Sharpe is large (trending).
        if abs(sharpe) > _MAX_SHARPE_ABS:
            confidence = 0.0
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
        # Hard gates: anomalous regime or elevated anomaly score -> flat.
        if signals.regime_id == ANOMALY_REGIME or signals.anomaly > _ANOMALY_GATE:
            return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=0.0)

        # Contrarian fade: negate the neighbourhood Sharpe direction.
        # confidence already embeds calm_gate * small-deviation guard.
        raw_fade = -math.tanh(_GAIN * signals.momentum)
        weight = raw_fade * signals.confidence

        # Dead-band: collapse negligible weights to flat.
        if abs(weight) < _DEADBAND:
            weight = 0.0

        weight = max(-1.0, min(1.0, weight))
        return TargetPosition(ts=signals.ts, symbol=signals.symbol, target_weight=weight)


def build() -> Strategy:
    return HighWinrateMeanRevStrategy()
